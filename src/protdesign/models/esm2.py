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

from protdesign.model import BaseModel, Scorer, Generator, RequiredResources
from protdesign.entity import System, SystemInstance, EntityInstance, EntityPosList, Mutant
from protdesign.constants import MASK, VALID_AA_OR_GAP_SORTED
from protdesign.sequence import valid_protein_sequence
from protdesign.utils import ensure_sequence, model_param_context
from protdesign.types import DeviceType, StatusCallback, BatchSize
import esm
try:
    import torch
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class ESM2(BaseModel, Scorer, Generator):
    """
    Wrapper class around ESM2 model
    """
    available = IMPORT_AVAILABLE
    name: str = "ESM2"

    requires_heavy_build: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = True
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    requires_target: bool = True
    requires_seqs: bool = False  # ESM2 doesn't require MSA
    requires_msa: bool = False
    requires_3d: bool = False
    requires_fixed_length: bool = True
    handles_deletions: bool = False  # ESM2 doesn't handle gaps as well as EVmutation2

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
        super().__init__()
        self.model_name = model_name
        self.keep_model_after_build = keep_model_after_build

        # by default, keep parameters loaded once loaded for prediction purposes to avoid reloading
        self.keep_model_after_pred = True
        self.device = device

        # modelled system
        self.system = None

        # lazy-load model when needed
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        # model parameters for inference
        self.decoder_batch_size = decoder_batch_size
        self.num_samples = num_samples

        if self.decoder_batch_size != "auto" and self.decoder_batch_size < 1:
            raise ValueError(
                "decoder_batch_size must be at least 1 or 'auto'"
            )

        if self.decoder_batch_size == "auto":
            raise NotImplementedError("Automatic batch_size not yet implemented")

        # encodings created when calling build() method
        self.encoding = None
        self.token_ids = None

    @property
    def ready(self):
        return self.system is not None and self.encoding is not None

    @classmethod
    def can_model(cls, system: System) -> Tuple[bool, str]:
        if len(system) != 1 or system[0].type_ != "protein":
            return False, "Can only handle single-component protein system"

        target = system[0]

        # this should be ensured by construction of system but check again to be safe
        if not valid_protein_sequence(
            target.rep, allow_mask=True, allow_gap=False, allow_ambiguous=True
        ):
            return False, "Input sequence may only contain AA symbols or mask (no gaps)"

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: System,
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
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
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
        status_callback: StatusCallback | None = None
    ) -> Self:
        # verify if we can model the system
        self.can_model_or_raise(system)

        # store system with this instance
        self.system = system
        target = self.system[0]

        # Load model if needed
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            # Prepare sequence for ESM2
            data = [("protein", target.rep)]
            _, _, token_ids = self.batch_converter(data)
            token_ids = token_ids.to(self.device)

            # Compute representations (no grad needed for inference)
            with torch.no_grad():
                results = self.model(token_ids, repr_layers=[self.model.num_layers])
                
                # Store the last layer representation
                representations = results["representations"][self.model.num_layers]
                
                # Store token IDs for later use
                self.token_ids = token_ids.cpu()
                
                # Store the encoding
                self.encoding = representations.cpu()

        # return self to allow method chaining
        return self

    def positions(
        self
    ) -> List[Tuple[int, int]]:
        self.ready_or_raise()

        # We model all positions of the target protein sequence
        target = self.system[0]
        return [
            (0, idx) for idx, _ in enumerate(target.rep, start=target.first_index)
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
        encoding_on_device = self.encoding.to(self.device)
        token_ids_on_device = self.token_ids.to(self.device)
        
        try:
            yield encoding_on_device, token_ids_on_device
        finally:
            # If not keeping representations, release them and clear cache
            if not keep:
                encoding_on_device = None
                token_ids_on_device = None
                self._release_cache()

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None
    ) -> List[SystemInstance]:
        """
        Generate protein sequences using the ESM2 model
        """
        self.ready_or_raise()
    
        # Verify validity of entity selection
        if entities is not None:
            entities = ensure_sequence(entities)
            if len(entities) != 1 or entities[0] != 0:
                raise ValueError("Can only design single entity (entities = [0] | None)")
        else:
            entities = [0]
    
        # Ensure num_designs is a multiple of batch_size
        if num_designs % self.decoder_batch_size != 0:
            logger.warning(f"Adjusting num_designs from {num_designs} to {self.decoder_batch_size * ((num_designs + self.decoder_batch_size - 1) // self.decoder_batch_size)} to be a multiple of batch_size")
            num_designs = self.decoder_batch_size * ((num_designs + self.decoder_batch_size - 1) // self.decoder_batch_size)
            
        target = self.system[0]
    
        # Extract fixed positions for the chain
        if fixed_pos is not None:
            if len(fixed_pos) != 1 or list(fixed_pos)[0] != 0:
                raise ValueError("Only accepting position mapping for entity 0")
    
            fixed_pos = set(fixed_pos[0])
            # Verify all positions are valid
            self.valid_positions(fixed_pos, entity=0, raise_invalid=True)
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
    
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            with self._reps_on_device(self.keep_model_after_pred) as (encoding_device, token_ids_device):
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
                            # Fix: Use convert_tokens_to_string instead of decode
                            tokens = batch_tokens[i].cpu().tolist()[1:-1]  # Remove cls and eos tokens
                            seq = "".join([self.alphabet.get_tok(token) for token in tokens])
                            designs.append(seq)
    
        # Score the designs
        ref_and_designs = [target.rep] + designs
        instances = [
            SystemInstance(
                EntityInstance(rep=rep)
            ) for rep in ref_and_designs
        ]
    
        # Score and attach to instances (normalize by reference score)
        scores = self.score(instances)
        ref_score = scores[0]
    
        # Remove reference in first position
        instances_with_score = [
            SystemInstance(
                EntityInstance(rep=seq),
                score=score - ref_score
            ) for seq, score in zip(ref_and_designs, scores)
        ][1:]
    
        return instances_with_score

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # Validate sequences
        _ = (
            self.system.valid_instance(
                instance, fixed_length=True, validate_reps=True, raise_invalid=True
            ) for instance in instances
        )

        sequences = [instance[0].rep for instance in instances]
        
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            scores = []
            
            # Process in batches
            for batch_start in range(0, len(sequences), self.decoder_batch_size):
                batch_end = min(batch_start + self.decoder_batch_size, len(sequences))
                batch_seqs = sequences[batch_start:batch_end]
                
                # Prepare batch data
                batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(batch_seqs)]
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
                        target_tokens = batch_tokens[i, 1:]  # Shifted right by 1
                        
                        # Get probability of each correct token
                        seq_log_probs = torch.gather(
                            token_probs, 
                            dim=1, 
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)
                        
                        # Sum log probabilities (excluding padding)
                        seq_log_likelihood = seq_log_probs.sum().item()
                        scores.append(seq_log_likelihood)

        return np.array(scores)

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int = 0,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Perform a single mutation scan for the given instance
        """
        self.ready_or_raise()

        # Check instance against molecular system
        self.system.valid_instance(
            instance, fixed_length=True, validate_reps=True, raise_invalid=True,
        )

        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        # Extract target entity from system and get sequence from instance
        target = self.system[0]
        instance_seq = instance[0].rep

        # Validate positions
        if positions is not None:
            positions = self.valid_positions(positions, raise_invalid=True)
        else:
            positions = list(range(target.first_index, target.first_index + len(target.rep)))

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
                        mut_seq = instance_seq[:pos_idx] + aa + instance_seq[pos_idx+1:]
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
        
        return df

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        # Check instance against molecular system
        self.system.valid_instance(
            instance, fixed_length=True, validate_reps=True, raise_invalid=True,
        )

        # Verify if mutants are valid relative to system and instance
        self.system.valid_mutants(
            instance, mutants, allow_gap=False, raise_invalid=True
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
        [
            self.system.valid_instance(
                instance, fixed_length=True, validate_reps=True, raise_invalid=True
            ) for instance in instances
        ]

        # Validate entity specification (only handle single entity for now)
        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")

        if not len(instances) == len(entities) == len(positions):
            raise ValueError("Sequences for instances, entities and positions must all have same length")

        target = self.system[0]

        # Validate positions
        self.valid_positions(positions, entity=0, raise_invalid=True)

        # Extract sequences
        seqs = [instance[0].rep for instance in instances]

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
                    'seq': i,
                    'pos': pos,
                    **aa_probs
                })
        
        # Convert to dataframe
        conditionals = pd.DataFrame(conditionals_list)
        conditionals = conditionals.set_index(['seq', 'pos'])
        
        # Add entity 0 to index
        conditionals = pd.concat({0: conditionals}, names=["entity"])
        
        assert len(conditionals) == len(entities), "Length mismatch between output and input"
        
        return conditionals