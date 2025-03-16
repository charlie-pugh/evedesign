"""
Wrapper class around ESM2 model
"""
from os import PathLike
from typing import Self, Tuple, Sequence, List
from contextlib import contextmanager

import numpy as np
import pandas as pd
from loguru import logger
import torch
from typing import Literal, List, Sequence

from protdesign.model import (
    BaseModel, Scorer, Generator, RequiredResources, MutationScorer, ConditionalMutationScorer
)
from protdesign.entity import System, SystemInstance, EntityInstance, EntityPosList, Mutant
from protdesign.constants import MASK, VALID_AA_OR_GAP_SORTED
from protdesign.utils import ensure_sequence, model_param_context
from protdesign.types import DeviceType, StatusCallback, BatchSize
from protdesign.samplers.gibbs import GibbsSampler

try:
    import torch
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class ESM2(BaseModel, Scorer, MutationScorer, ConditionalMutationScorer, Generator):
    """
    Wrapper class around ESM2 model
    """
    available = IMPORT_AVAILABLE
    name: str = "ESM2"

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = False
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    # molecular model properties
    requires_heavy_build: bool = False
    requires_seqs: bool = False  # ESM2 doesn't require MSA
    requires_msa: bool = False
    requires_3d: bool = False

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",  # Default model
        decoder_batch_size: BatchSize = 64,
        num_samples: int = 16,
        keep_model_after_build: bool = False,
        device: DeviceType = "cpu",
    ):
        """
        Instantiate new ESM2 model

        Parameters
        ----------
        model_name
            Name of the ESM2 model to load (e.g., "esm2_t33_650M_UR50D")
        decoder_batch_size
            Maximum number of sequences to process concurrently
        num_samples
            Number of samples to generate when sampling sequences
        keep_model_after_build
            If True, keep model parameters associated to instance after build step
            to avoid reloading when scoring/generating. If serializing model, set to
            False to avoid storing model parameters repeatedly.
        device
            Device to use for computations
        """
        if not self.available:
            raise ValueError(
                "ESM2 package could not be imported. Is it installed already?")

        self.model_name = model_name
        self.keep_model_after_build = keep_model_after_build

        # by default, keep parameters loaded once loaded for prediction purposes to avoid reloading
        self.keep_model_after_pred = True
        self.device = device

        # modelled system
        self._system = None
        # lazy-load model when needed
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        # model parameters for inference
        self.decoder_batch_size = decoder_batch_size
        self.num_samples = num_samples

        if self.num_samples < 1:
            raise ValueError("num_samples must be > 0")

        if self.decoder_batch_size != "auto" and self.decoder_batch_size < 1:
            raise ValueError(
                "decoder_batch_size must be at least 1 or 'auto'"
            )

        if self.decoder_batch_size == "auto":
            raise NotImplementedError(
                "Automatic batch_size not yet implemented")

        # encodings created when calling build() method
        self.encoding = None
        self.token_ids = None

    @property
    def ready(self):
        return self.system is not None and self.encoding is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> Tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"

        if len(system) != 1 or system[0].type_ != "protein":
            return False, "Can only handle single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have defined rep sequence"

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: System,
        data: None = None,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        # TODO: implement meaningful requirements depending on target size
        return RequiredResources(
            min_gpu_cores=1,
            min_gpu_memory_per_core=16000,
            min_cpu_cores=1,
            min_cpu_memory_per_core=16000,
            max_batch_size=512,
            time=1,
        )

    def _load_model(self):
        # avoid reloading if already loaded
        if self.model is not None:
            return

        # Load ESM-2 model
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", self.model_name)
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # switch to evaluation mode
        self.model.eval()

    def _release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self):
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._release_cache()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None
    ) -> Self:
        # verify if we can model the system
        self.can_model_or_raise(system, data)

        # store system with this instance
        self._system = system
        target = self.system[0]

        # Load model if needed
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            # Prepare sequence for ESM2
            data = [("protein", "".join(target.rep))]

            _, _, token_ids = self.batch_converter(data)
            token_ids = token_ids.to(self.device)

            # Compute representations (no grad needed for inference)
            with torch.no_grad():
                results = self.model(token_ids, repr_layers=[
                                     self.model.num_layers])

                # Store the last layer representation
                representations = results["representations"][self.model.num_layers]

                # Store token IDs for later use
                self.token_ids = token_ids.cpu()

                # Store the encoding
                self.encoding = representations.cpu()

        # return self to allow method chaining
        return self

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> List[Tuple[int, int]]:
        self.ready_or_raise()

        # implementation here is very simple: we model all positions of exactly one target
        # protein sequence of fixed length (i.e. can ignore the passed instance)
        target = self.system[0]
        return [
            (0, pos) for pos, _ in enumerate(target.rep, start=target.first_index)
        ]

    def _validate_instances(
        self,
        instances: Sequence[SystemInstance],
    ) -> None:
        # validate instance sequences; must all have the same length
        [
            self.system.valid_instance(
                instance,
                validate_reps=True,
                fixed_length=True,
                allow_deletions=False,
                raise_invalid=True,
            ) for instance in instances
        ]

    @contextmanager
    def _reps_on_device(self, keep: bool = True):
        """
        Helper to move necessary information to target device
        when calling inference methods

        Parameters
        ----------
        keep
            If True, keep representations on device after exiting the manager;
            if False, remove them and clear cache where applicable
        """
        # Move representations to device
        encoding_on_device = None
        token_ids_on_device = None

        try:
            # reload representations if anything is missing
            if self.encoding is not None:
                encoding_on_device = self.encoding.to(self.device)
            if self.token_ids is not None:
                token_ids_on_device = self.token_ids.to(self.device)

            yield encoding_on_device, token_ids_on_device
        finally:
            # If not keeping representations, release them and clear cache
            if not keep:
                encoding_on_device = None
                token_ids_on_device = None
                self._release_cache()

    '''
    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        deletions: bool = False,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
        """
        Generate protein sequences using the ESM2 model
        """
        self.ready_or_raise()
    
        # verify validity of entity selection, even if not used since
        # at this point method can only handle single entity
        if entities is not None:
            entities = ensure_sequence(entities)
            if len(entities) != 1 or entities[0] != 0:
                raise ValueError("Can only design single entity (entities = [0] | None)")
        else:
            entities = [0]
    
        # Ensure num_designs is a multiple of batch_size
        if rem := num_designs % self.decoder_batch_size:
            num_designs_adj = num_designs + (self.decoder_batch_size - rem)
            logger.warning(f"Adjusting num_designs from {num_designs} to {num_designs_adj} to be a multiple of batch_size")
            num_designs = num_designs_adj
            
        target = self.system[0]
    
        # Extract fixed positions for the chain
        if fixed_pos is not None:
            if len(fixed_pos) != 1 or list(fixed_pos)[0] != 0:
                raise ValueError("Only accepting position mapping for entity 0")
    
            # verify if all positions are valid
            self.valid_positions(fixed_pos[0], entities=0, raise_invalid=True)
            fixed_pos = set(fixed_pos[0])
        else:
            fixed_pos = set()
    
        if len(fixed_pos) == len(target.rep):
            raise ValueError("All positions fixed, need to sample at least one position")
    
        # Mark which positions to design (with mask symbol)
        base_seq = [
            symbol if pos in fixed_pos else MASK
            for pos, symbol in enumerate(
                target.rep, start=target.first_index
            )
        ]
        
        # Convert to string
        base_seq = "".join(base_seq)
    
        with (
            model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred),
            self._reps_on_device(self.keep_model_after_pred) as (encoding_device, token_ids_device)
        ):
            logger.info(f"Generating {num_designs} designs with ESM2")
            
            # Generate sequences using ESM2
            designs = []
            
            # Create mask for positions to be designed
            mask = torch.ones_like(token_ids_device)
            for i, char in enumerate(base_seq):
                if char != MASK:
                    # +1 because of cls token at beginning
                    mask[0, i+1] = 0
            
            with torch.no_grad():
                for batch_start in range(0, num_designs, self.decoder_batch_size):
                    batch_size = min(self.decoder_batch_size, num_designs - batch_start)
                    
                    # Create batch of token IDs
                    batch_tokens = token_ids_device.repeat(batch_size, 1)
                    batch_mask = mask.repeat(batch_size, 1)
                    
                    # For all masked positions, sample from the model distribution
                    for pos in range(1, len(base_seq) + 1):  # Skip cls token at pos 0
                        if base_seq[pos-1] == MASK:
                            # Forward pass to get logits
                            logits = self.model(batch_tokens)["logits"][:, pos, :]
                            
                            # Apply temperature
                            if temperature > 0:
                                logits = logits / temperature
                            
                            # Sample from the distribution
                            probs = torch.softmax(logits, dim=-1)
                            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
                            
                            # Update tokens at this position
                            batch_tokens[:, pos] = sampled_tokens
                    
                    # Convert back to amino acid sequences
                    for i in range(batch_size):
                        # Skip cls and eos tokens, convert to amino acids
                        tokens = batch_tokens[i].cpu().tolist()[1:-1]  # Remove cls and eos tokens
                        seq = "".join([self.alphabet.get_tok(token) for token in tokens])
                        designs.append(seq)
    
        # score the designs relative to entity sequence
        ref_and_designs = ["".join(target.rep)] + designs[:num_designs]  # Ensure we only return the requested number
        
        # Make sure sequences are properly converted to strings
        instances = [
            SystemInstance(
                EntityInstance(rep=seq if isinstance(seq, str) else "".join(seq))
            ) for seq in ref_and_designs
        ]
    
        # Score and attach to instances (normalize by reference score)
        scores = self.score(instances)
        ref_score = scores[0]
    
        # Remove reference in first position
        instances_with_score = [
            SystemInstance(
                EntityInstance(rep=seq),
                score=score - ref_score
            ) for seq, score in zip(ref_and_designs[1:], scores[1:])
        ]
    
        assert len(instances_with_score) >= num_designs, "Not returning minimum guaranteed number of designs"
        return instances_with_score[:num_designs]  # Return exactly the number requested
    '''

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        deletions: bool = False,
        status_callback: StatusCallback | None = None,
        num_sweeps: int = 10,  # Number of full Gibbs sweeps across all positions
        scan_order: Literal["random", "sequential"] = "random"
    ) -> List[SystemInstance]:
        """
        Generate protein sequences using the ESM2 model with Gibbs sampling

        Parameters
        ----------
        num_designs
            Number of designs to generate
        entities
            Which entities to design (defaults to all)
        fixed_pos
            Positions to keep fixed during design
        temperature
            Temperature parameter for sampling (higher = more diversity)
        deletions
            Whether to allow gap characters
        status_callback
            Callback for progress reporting
        num_sweeps
            Number of full Gibbs sweeps over all positions
        scan_order
            Whether to visit positions in random or sequential order
        """
        self.ready_or_raise()

        # verify validity of entity selection, even if not used since
        # at this point method can only handle single entity
        if entities is not None:
            entities = ensure_sequence(entities)
            if len(entities) != 1 or entities[0] != 0:
                raise ValueError(
                    "Can only design single entity (entities = [0] | None)")
        else:
            entities = [0]

        # Ensure num_designs is a multiple of batch_size
        if rem := num_designs % self.decoder_batch_size:
            num_designs_adj = num_designs + (self.decoder_batch_size - rem)
            logger.warning(
                f"Adjusting num_designs from {num_designs} to {num_designs_adj} to be a multiple of batch_size")
            num_designs = num_designs_adj

        target = self.system[0]

        # Extract fixed positions for the chain
        if fixed_pos is not None:
            if len(fixed_pos) != 1 or list(fixed_pos)[0] != 0:
                raise ValueError(
                    "Only accepting position mapping for entity 0")

            # verify if all positions are valid
            self.valid_positions(fixed_pos[0], entities=0, raise_invalid=True)
            fixed_pos = set(fixed_pos[0])
        else:
            fixed_pos = set()

        if len(fixed_pos) == len(target.rep):
            raise ValueError(
                "All positions fixed, need to sample at least one position")

        # Determine positions to sample (those not in fixed_pos)
        design_positions = [
            pos for pos in range(target.first_index, target.first_index + len(target.rep))
            if pos not in fixed_pos
        ]

        # Initialize random number generator
        rng = np.random.default_rng()

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            logger.info(
                f"Generating {num_designs} designs with ESM2 using Gibbs sampling")

            # Initialize sequences either randomly or from template
            sequences = []
            for _ in range(num_designs):
                # Start with wildtype sequence
                seq_chars = list(target.rep)

                # Randomize all design positions
                valid_aa = VALID_AA_OR_GAP_SORTED if deletions else [
                    aa for aa in VALID_AA_OR_GAP_SORTED if aa != '-']
                for pos in design_positions:
                    idx = pos - target.first_index
                    seq_chars[idx] = rng.choice(valid_aa)

                sequences.append("".join(seq_chars))

            # Convert to SystemInstance objects and ensure sequences are strings
            instances = []
            for seq in sequences:
                # Make sure sequence is a string
                if isinstance(seq, np.ndarray):
                    seq = "".join(seq)
                instances.append(SystemInstance(EntityInstance(rep=seq)))

            # Status update function
            def update_status(progress):
                if status_callback:
                    status_callback(progress)

            # Perform Gibbs sampling
            total_steps = num_sweeps * len(design_positions)
            step_count = 0

            for sweep in range(num_sweeps):
                # Determine position order for this sweep
                if scan_order == "random":
                    rng.shuffle(design_positions)

                # For each position in the sweep
                for pos in design_positions:
                    pos_idx = pos - target.first_index  # Convert to 0-indexed

                    # Process designs in batches
                    for batch_start in range(0, num_designs, self.decoder_batch_size):
                        batch_end = min(
                            batch_start + self.decoder_batch_size, num_designs)
                        batch_instances = instances[batch_start:batch_end]
                        batch_size = len(batch_instances)

                        # Make sure batch instances have string sequences
                        for i, instance in enumerate(batch_instances):
                            if isinstance(instance[0].rep, np.ndarray):
                                batch_instances[i][0].rep = "".join(
                                    instance[0].rep)

                        # Get conditional probabilities for this position across all sequences
                        conditionals = self.score_conditional(
                            batch_instances,
                            entities=[0] * batch_size,
                            positions=[pos] * batch_size
                        )

                        # Filter conditionals to only valid amino acids
                        # Define gap character
                        GAP = '-'
                        if not deletions:
                            conditionals = conditionals.drop(
                                GAP, axis=1, errors='ignore')

                        # Apply temperature scaling
                        prob_cols = [col for col in conditionals.columns if col !=
                                     'entity' and col != 'instance' and col != 'pos']

                        for i, row_idx in enumerate(conditionals.index):
                            # Get probabilities for this instance
                            probs = conditionals.loc[row_idx, prob_cols].values

                            # Apply temperature scaling
                            if temperature != 0:
                                probs = np.exp(np.log(probs) / temperature)
                                probs = probs / probs.sum()  # Renormalize

                            # Sample a new amino acid
                            new_aa = rng.choice(prob_cols, p=probs)

                            # Update the sequence
                            instance_idx = batch_start + i
                            seq_chars = list(instances[instance_idx][0].rep)
                            seq_chars[pos_idx] = new_aa
                            instances[instance_idx][0].rep = "".join(seq_chars)

                    # Update progress
                    step_count += 1
                    update_status(step_count / total_steps)

            # Score the designs relative to entity sequence
            ref_seq = "".join(target.rep)
            all_sequences = [ref_seq] + \
                [instance[0].rep for instance in instances]

            all_instances = [
                SystemInstance(EntityInstance(rep=seq))
                for seq in all_sequences
            ]

            # Score and normalize by reference
            scores = self.score(all_instances)
            ref_score = scores[0]

            # Attach scores to the instances
            scored_instances = [
                SystemInstance(
                    EntityInstance(rep=instance[0].rep),
                    score=score - ref_score
                ) for instance, score in zip(instances, scores[1:])
            ]

        # Return exactly the number requested
        return scored_instances[:num_designs]

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # Validate sequences
        self._validate_instances(instances)

        # Make sure sequences are strings, not numpy arrays
        sequences = []
        for instance in instances:
            seq = instance[0].rep
            # Convert numpy array to string if needed
            if isinstance(seq, np.ndarray):
                seq = "".join(seq)
            sequences.append(seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            scores = []

            # Process in batches
            for batch_start in range(0, len(sequences), self.decoder_batch_size):
                batch_end = min(
                    batch_start + self.decoder_batch_size, len(sequences))
                batch_seqs = sequences[batch_start:batch_end]

                # Prepare batch data
                batch_data = [(f"seq_{i}", seq)
                              for i, seq in enumerate(batch_seqs)]
                _, _, batch_tokens = self.batch_converter(batch_data)
                batch_tokens = batch_tokens.to(self.device)

                # Compute log-likelihoods
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[])
                    logits = results["logits"]

                    # Calculate log-likelihood for each sequence
                    for i, seq in enumerate(batch_seqs):
                        # Skip first token (BOS) and calculate log-likelihood
                        token_probs = torch.log_softmax(logits[i, :-1], dim=-1)
                        # Shifted right by 1
                        target_tokens = batch_tokens[i, 1:]

                        # Get probability of each correct token
                        seq_log_probs = torch.gather(
                            token_probs,
                            dim=1,
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)

                        # Sum log probabilities (excluding padding)
                        seq_log_likelihood = seq_log_probs.sum().item()
                        scores.append(seq_log_likelihood)

        # return as numpy vector
        return np.array(scores)

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Perform a single mutation scan for the given instance
        """
        self.ready_or_raise()

        # Check instance against molecular system
        self._validate_instances([instance])

        if entity is None:
            entity = 0

        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        # Extract target entity from system and get sequence from instance
        target = self.system[0]
        instance_seq = instance[0].rep

        # Convert numpy array to string if necessary
        # Check if it's a numpy array
        if hasattr(instance_seq, 'dtype') and hasattr(instance_seq, 'tolist'):
            if instance_seq.dtype.kind == 'U':  # Unicode strings
                instance_seq = ''.join(instance_seq.tolist())
            else:
                instance_seq = ''.join(instance_seq.astype(str).tolist())

        # Validate positions
        if positions is not None:
            self.valid_positions(positions, entities=0, raise_invalid=True)
        else:
            positions = list(
                range(target.first_index, target.first_index + len(target.rep)))

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            # Data structure to hold mutation effects
            mutation_effects = []

            # Prepare the reference sequence
            ref_data = [("protein", instance_seq)]
            _, _, ref_tokens = self.batch_converter(ref_data)
            ref_tokens = ref_tokens.to(self.device)

            # Calculate reference sequence score with a forward pass
            with torch.no_grad():
                ref_results = self.model(ref_tokens, repr_layers=[])
                ref_logits = ref_results["logits"][0]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

                # For each position
                for pos in positions:
                    pos_idx = pos - target.first_index  # Convert to 0-based indexing in seq

                    # Get wildtype aa at this position
                    wt_aa = instance_seq[pos_idx]

                    # For each possible amino acid substitution
                    mut_scores = {}
                    for aa in VALID_AA_OR_GAP_SORTED:
                        if aa == '-':  # Skip gap character for ESM2
                            continue

                        # If same as wildtype, effect is 0
                        if aa == wt_aa:
                            mut_scores[aa] = 0.0
                            continue

                        # Create mutant sequence
                        mut_seq = instance_seq[:pos_idx] + \
                            aa + instance_seq[pos_idx+1:]
                        mut_data = [("mutant", mut_seq)]
                        _, _, mut_tokens = self.batch_converter(mut_data)
                        mut_tokens = mut_tokens.to(self.device)

                        # Score the mutant
                        mut_results = self.model(mut_tokens, repr_layers=[])
                        mut_logits = mut_results["logits"][0]
                        mut_log_probs = torch.log_softmax(mut_logits, dim=-1)

                        # Calculate difference in log-likelihood
                        target_tokens = mut_tokens[0, 1:]
                        mut_seq_log_probs = torch.gather(
                            mut_log_probs[:-1],
                            dim=1,
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)

                        target_tokens_ref = ref_tokens[0, 1:]
                        ref_seq_log_probs = torch.gather(
                            ref_log_probs[:-1],
                            dim=1,
                            index=target_tokens_ref.unsqueeze(1)
                        ).squeeze(1)

                        # Calculate difference in log-likelihood (ddG-like score)
                        score_diff = mut_seq_log_probs.sum().item() - ref_seq_log_probs.sum().item()
                        mut_scores[aa] = score_diff

                    # Store results for this position
                    mutation_effects.append({
                        'pos': pos,
                        'ref': wt_aa,
                        **mut_scores
                    })

        # Convert to dataframe
        df = pd.DataFrame(mutation_effects)
        df = df.set_index(['pos', 'ref'])

        # Add entity 0 to index
        df = pd.concat({entity: df}, names=["entity"])

        assert (
            (positions is None and len(df) == len(target.rep)) or
            (positions is not None and len(df) == len(positions))
        ), "Invalid number of positions in output dataframe"

        return df

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # Check instance against molecular system
        self._validate_instances([instance])

        # Verify if mutants are valid relative to system and instance
        self.system.valid_mutants(
            instance, mutants, deletions=False, insertions=False, raise_invalid=True
        )

        # Extract target entity from system, and get sequence from instance
        target = self.system[0]
        instance_seq = instance[0].rep

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            # Prepare the reference sequence
            ref_data = [("protein", instance_seq)]
            _, _, ref_tokens = self.batch_converter(ref_data)
            ref_tokens = ref_tokens.to(self.device)

            # Score reference sequence
            with torch.no_grad():
                ref_results = self.model(ref_tokens, repr_layers=[])
                ref_logits = ref_results["logits"][0]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

                target_tokens_ref = ref_tokens[0, 1:]
                ref_seq_log_probs = torch.gather(
                    ref_log_probs[:-1],
                    dim=1,
                    index=target_tokens_ref.unsqueeze(1)
                ).squeeze(1)
                ref_score = ref_seq_log_probs.sum().item()

                # Score each mutant
                mutant_scores = []
                for mutant in mutants:
                    # Apply mutations to create mutant sequence
                    mut_seq = list(instance_seq)
                    for sub in mutant:
                        # Convert to 0-based indexing in sequence
                        pos_idx = sub.pos - target.first_index
                        mut_seq[pos_idx] = sub.to
                    mut_seq = "".join(mut_seq)

                    # Score the mutant
                    mut_data = [("mutant", mut_seq)]
                    _, _, mut_tokens = self.batch_converter(mut_data)
                    mut_tokens = mut_tokens.to(self.device)

                    mut_results = self.model(mut_tokens, repr_layers=[])
                    mut_logits = mut_results["logits"][0]
                    mut_log_probs = torch.log_softmax(mut_logits, dim=-1)

                    target_tokens = mut_tokens[0, 1:]
                    mut_seq_log_probs = torch.gather(
                        mut_log_probs[:-1],
                        dim=1,
                        index=target_tokens.unsqueeze(1)
                    ).squeeze(1)

                    mut_score = mut_seq_log_probs.sum().item()
                    mutant_scores.append(mut_score - ref_score)

        return np.array(mutant_scores)

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Score conditional probabilities for specified positions in the sequences
        """
        self.ready_or_raise()

        # Validate instance sequences
        self._validate_instances(instances)

        # Validate entity specification (only handle single entity for now)
        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")

        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length")

        target = self.system[0]

        # Validate positions
        self.valid_positions(positions, entities=0, raise_invalid=True)

        # Extract sequences and ensure they're strings, not numpy arrays
        seqs = []
        for instance in instances:
            seq = instance[0].rep
            # Convert numpy array to string if needed
            if isinstance(seq, np.ndarray):
                seq = "".join(seq)
            seqs.append(seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            conditionals_list = []

            # Process each sequence and position pair
            for i, (seq, pos) in enumerate(zip(seqs, positions)):
                # Convert position to 0-based indexing in sequence
                pos_idx = pos - target.first_index

                # Prepare the sequence
                seq_data = [("protein", seq)]
                _, _, seq_tokens = self.batch_converter(seq_data)
                seq_tokens = seq_tokens.to(self.device)

                with torch.no_grad():
                    # Forward pass to get logits
                    results = self.model(seq_tokens, repr_layers=[])
                    logits = results["logits"][0]

                    # Get conditional probabilities at the specified position
                    # ESM2 uses position+1 because of the BOS token
                    pos_logits = logits[pos_idx + 1]
                    pos_probs = torch.softmax(pos_logits, dim=-1)

                    # Convert to amino acid probabilities
                    aa_probs = {}
                    for aa in VALID_AA_OR_GAP_SORTED:
                        if aa == '-':  # Skip gap character for ESM2
                            aa_probs[aa] = 0.0
                        else:
                            # Get token ID for this amino acid
                            aa_token = self.alphabet.get_idx(aa)
                            aa_probs[aa] = pos_probs[aa_token].item()

                # Store results
                conditionals_list.append({
                    'instance': i,
                    'pos': pos,
                    **aa_probs
                })

        # Convert to dataframe
        conditionals = pd.DataFrame(conditionals_list)
        conditionals = conditionals.set_index(['instance', 'pos'])

        # Add entity 0 to index
        conditionals = pd.concat({0: conditionals}, names=["entity"])

        assert len(conditionals) == len(
            entities), "Length mismatch between output and input"

        return conditionals
