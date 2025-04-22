from os import PathLike
from typing import Self, Tuple, Sequence, List
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import torch

from protdesign.model import (
    BaseModel, Scorer, Generator, RequiredResources, MutationScorer, ConditionalMutationScorer
)
from protdesign.entity import System, SystemInstance, EntityInstance, EntityPosList, Mutant
from protdesign.utils import model_param_context
from protdesign.types import DeviceType, StatusCallback, BatchSize
from protdesign.samplers.gibbs import GibbsSampler, ScanOrder, InitStrategy, TemperatureSchedule

try:
    from transformers import EsmForMaskedLM, AutoTokenizer  # noqa
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
    requires_seqs: bool = False
    requires_msa: bool = False
    requires_3d: bool = False

    def __init__(
        self,
        model_name: str | None = None,
        model_dir_path: str | PathLike | None = None,
        decoder_batch_size: BatchSize = 64,
        keep_model_after_build: bool = False,
        device: DeviceType = "cpu",
        # GibbsSampler hyperparameters
        num_sweeps: int = 1000,
        init_strategy: InitStrategy = "system",
        scan_order: ScanOrder = "random",
        temperature_schedule: TemperatureSchedule | None = None
    ):
        if not self.available:
            raise ValueError(
                "transformers package could not be imported. Is it installed already?"
            )

        # Validate model specification parameters
        if (model_name is None and model_dir_path is None) or (model_name is not None and model_dir_path is not None):
            raise ValueError(
                "Must specify exactly one of model_name or model_file_path, but not both"
            )

        self.model_name = model_name
        self.model_dir_path = Path(model_dir_path) if model_dir_path is not None else None
        self.keep_model_after_build = keep_model_after_build
        self.keep_model_after_pred = True
        self.device = device

        # Define maximum sequence length for ESM2 models (1024 tokens - 2 for special tokens)
        self.max_seq_length = 1022

        self._system = None
        self.model = None
        self.tokenizer = None  # Changed from alphabet to tokenizer

        self.decoder_batch_size = decoder_batch_size

        # Store GibbsSampler hyperparameters
        self.num_sweeps = num_sweeps
        self.init_strategy = init_strategy
        self.scan_order = scan_order
        self.temperature_schedule = temperature_schedule

        if self.decoder_batch_size != "auto" and self.decoder_batch_size < 1:
            raise ValueError(
                "decoder_batch_size must be at least 1 or 'auto'"
            )

        if self.decoder_batch_size == "auto":
            raise NotImplementedError(
                "Automatic batch_size not yet implemented"
            )

        self.token_ids = None
        self.encoding = None

    @property
    def ready(self):
        return self._system is not None

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

        # Add check for sequence length
        max_seq_length = 1022  # 1024 - 2 for special tokens
        if len(target.rep) > max_seq_length:
            return False, f"Sequence length ({len(target.rep)}) exceeds maximum allowed ({max_seq_length})"

        return True, ""

    @classmethod
    def required_resources(
        cls,
        system: System,
        data: None = None,
        use_gpu: bool = True,
        build: bool = True,
    ) -> RequiredResources:
        raise NotImplementedError(
            "Resource estimation not yet implemented"
        )
        # return RequiredResources(
        #     min_gpu_cores=1,
        #     min_gpu_memory_per_core=16000,
        #     min_cpu_cores=1,
        #     min_cpu_memory_per_core=16000,
        #     max_batch_size=512,
        #     time=1,
        # )

    def _load_model(self):
        if self.model is not None:
            return

        if self.model_name is not None:
            # Load from HuggingFace hub
            try:
                # For remote loading from HuggingFace
                self.model = EsmForMaskedLM.from_pretrained(
                    f"facebook/{self.model_name}"
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f"facebook/{self.model_name}"
                )
            except Exception as e:
                logger.error(f"Error loading model from HuggingFace: {e}")
                raise ValueError(
                    f"Failed to load model {self.model_name} from HuggingFace: {e}"
                )
        elif self.model_dir_path is not None:
            # Load from local file path
            try:
                # For local loading from a directory
                self.model = EsmForMaskedLM.from_pretrained(
                    self.model_dir_path
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir_path
                )
            except Exception as e:
                logger.error(f"Error loading model from local path: {e}")

                raise ValueError(
                    f"Failed to load model from {self.model_dir_path}: {e}"
                )

        self.model.eval()

    def _release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self):
        self.model = None
        self.tokenizer = None  # Changed from alphabet to tokenizer

        self._release_cache()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system

        # Additional check for sequence length
        target = system[0]
        if len(target.rep) > self.max_seq_length:
            raise ValueError(
                f"Sequence length ({len(target.rep)}) exceeds maximum allowed by ESM2 ({self.max_seq_length})"
            )

        self.encoding = None
        self.token_ids = None

        return self

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> List[Tuple[int, int]]:
        self.ready_or_raise()
        target = self.system[0]
        return [(0, pos) for pos, _ in enumerate(target.rep, start=target.first_index)]

    def _validate_instances(
        self,
        instances: Sequence[SystemInstance],
    ) -> None:
        # Validate all instances in a single loop
        for instance in instances:
            # First validate the instance with system validation
            self.system.valid_instance(
                instance,
                validate_reps=True,
                fixed_length=True,
                allow_deletions=False,
                raise_invalid=True,
            )

            # Now that we know the instance is valid
            seq = instance[0].rep
            seq_len = len(seq)

            # Check sequence length
            if seq_len > self.max_seq_length:
                raise ValueError(
                    f"Sequence length ({seq_len}) exceeds maximum allowed by ESM2 ({self.max_seq_length})"
                )

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        deletions: bool = False,
        status_callback: StatusCallback | None = None,
    ) -> List[SystemInstance]:
        """
        Generate protein sequences using the ESM2 model with the GibbsSampler

        Parameters
        ----------
        num_designs
            Number of protein sequences to generate
        entities
            Indices of entities to redesign (default: [0])
        fixed_pos
            Positions to keep fixed during design
        temperature
            Initial temperature for sampling
        deletions
            Whether to allow deletions
        status_callback
            Optional callback function for progress updates

        Returns
        -------
        List[SystemInstance]
            Generated protein sequence instances
        """
        self.ready_or_raise()

        # Add validation for deletions parameter
        if deletions:
            raise ValueError(
                "ESM2 model does not support deletions (gaps)"
            )

        entities = entities if entities is not None else [0]
        if len(entities) != 1 or entities[0] != 0:
            raise ValueError(
                "Can only design single entity (entities = [0] | None)"
            )

        # Adjust num_designs to be a multiple of batch_size
        if rem := num_designs % self.decoder_batch_size:
            num_designs_adj = num_designs + (self.decoder_batch_size - rem)
            num_designs = num_designs_adj

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            logger.info(
                f"Generating {num_designs} designs with ESM2 using GibbsSampler"
            )

            # Create a GibbsSampler using the configured hyperparameters
            sampler = GibbsSampler(
                scorers=[self],
                weights=None,
                num_sweeps=self.num_sweeps,
                init_strategy=self.init_strategy,
                scan_order=self.scan_order,
                temperature_schedule=self.temperature_schedule,
                require_strict_pos=True,
                record_full_chain=False
            )

            # Generate designs
            instances = sampler.generate(
                num_designs=num_designs,
                entities=entities,
                fixed_pos=fixed_pos,
                temperature=temperature,
                deletions=deletions,  # This will now be False because of the validation
                status_callback=status_callback
            )

        # Score designs relative to reference
        target = self.system[0]
        ref_instance = SystemInstance(EntityInstance(rep="".join(target.rep)))
        all_instances = [ref_instance] + instances

        logger.info(f"Scoring {len(instances)} generated designs")
        scores = self.score(all_instances)
        ref_score = scores[0]

        # Attach normalized scores to instances
        for i, instance in enumerate(instances):
            instance.score = (scores[i+1] - ref_score)

        return instances

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()
        self._validate_instances(instances)

        # Convert any sequence arrays to strings
        sequences = []
        for instance in instances:
            seq = instance[0].rep
            seq = "".join(seq)
            sequences.append(seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            scores = []

            # Process in batches
            for batch_start in range(0, len(sequences), self.decoder_batch_size):
                batch_end = min(
                    batch_start + self.decoder_batch_size, len(sequences)
                )
                batch_seqs = sequences[batch_start:batch_end]

                # Prepare batch data with tokenizer
                inputs = self.tokenizer(
                    batch_seqs, return_tensors="pt", padding=True
                ).to(self.device)

                # Compute log-likelihoods
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # Calculate log-likelihood for each sequence
                    for i, seq in enumerate(batch_seqs):
                        # Get sequence length (excluding padding)
                        # -2 for special tokens
                        seq_len = len(self.tokenizer.encode(seq)) - 2

                        # Extract logits for the actual sequence (excluding padding and the last token)
                        # Skip the first special token
                        seq_logits = logits[i, 1:seq_len+1]

                        # Get target tokens (shifted by one position)
                        # +2 to include one more token as target
                        target_tokens = inputs.input_ids[i, 2:seq_len+2]

                        # Calculate log probabilities
                        token_probs = torch.log_softmax(seq_logits, dim=-1)

                        # Gather log probs for the target tokens
                        seq_log_probs = torch.gather(
                            token_probs,
                            dim=1,
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)

                        # Sum log probs to get sequence log likelihood
                        seq_log_likelihood = seq_log_probs.sum().item()

                        scores.append(seq_log_likelihood)

        return np.array(scores)

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None
    ) -> pd.DataFrame:
        """
        Perform a single mutation scan for the given instance using the Wildtype marginal probability approach
        """
        self.ready_or_raise()
        self._validate_instances([instance])

        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        # Get sequence and convert to string if needed
        target = self.system[0]
        instance_seq = instance[0].rep
        instance_seq = "".join(instance_seq)

        # Validate positions
        if positions is not None:
            self.valid_positions(positions, entities=0, raise_invalid=True)
        else:
            positions = list(
                range(target.first_index, target.first_index + len(target.rep))
            )

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            mutation_effects = []

            # Prepare reference sequence with tokenizer
            inputs = self.tokenizer(
                instance_seq, return_tensors="pt").to(self.device)

            # Calculate reference probabilities with a single forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                ref_logits = outputs.logits[0]

                # Convert logits to log probabilities
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

                # For each position to scan
                for pos in positions:
                    pos_idx = pos - target.first_index
                    wt_aa = instance_seq[pos_idx]

                    # Adjust for tokenizer offsets (assuming 1-to-1 mapping + 1 for start token)

                    token_idx = pos_idx + 1

                    # Extract log probabilities for all amino acids at this position
                    pos_log_probs = ref_log_probs[token_idx]

                    # Score each possible substitution
                    mut_scores = {}
                    for aa in target.alphabet(include_gap=False):
                        if aa == '-':  # Skip gap character
                            continue

                        # If same as wildtype, effect is 0
                        if aa == wt_aa:
                            mut_scores[aa] = 0.0
                            continue

                        # Get the token index for this amino acid
                        aa_token = self.tokenizer.convert_tokens_to_ids(aa)
                        wt_token = self.tokenizer.convert_tokens_to_ids(wt_aa)

                        # For wildtype marginal probability, calculate:
                        # -log(p(mut_aa)) + log(p(wt_aa))
                        score_diff = (pos_log_probs[aa_token].item() -
                                      pos_log_probs[wt_token].item())

                        mut_scores[aa] = score_diff

                    # Store results for this position
                    mutation_effects.append({
                        'pos': pos,
                        'ref': wt_aa,
                        **mut_scores
                    })

        # Convert to dataframe with proper index format
        df = pd.DataFrame(mutation_effects)
        df = df.set_index(['pos', 'ref'])
        df = pd.concat({entity: df}, names=["entity"])

        return df

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()
        self._validate_instances([instance])
        self.system.valid_mutants(
            instance, mutants, deletions=False, insertions=False, raise_invalid=True
        )

        # Get instance sequence
        target = self.system[0]
        instance_seq = instance[0].rep
        instance_seq = "".join(instance_seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            # Score reference sequence with a single forward pass
            inputs = self.tokenizer(
                instance_seq, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                ref_logits = outputs.logits[0]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

                # Calculate scores for all mutants using the reference probabilities
                mutant_scores = []
                for mutant in mutants:
                    total_score = 0.0

                    for sub in mutant:
                        pos_idx = sub.pos - target.first_index
                        wt_aa = instance_seq[pos_idx]
                        mut_aa = sub.to

                        if wt_aa == mut_aa:
                            continue  # No change in score for unchanged positions

                        # Adjust for tokenizer offsets
                        token_pos = pos_idx + 1  # +1 for start token

                        # Get token IDs
                        wt_token = self.tokenizer.convert_tokens_to_ids(wt_aa)
                        mut_token = self.tokenizer.convert_tokens_to_ids(
                            mut_aa)

                        # Calculate score difference for this mutation
                        wt_log_prob = ref_log_probs[token_pos, wt_token].item()
                        mut_log_prob = ref_log_probs[token_pos, mut_token].item()

                        score_diff = (mut_log_prob - wt_log_prob)
                        total_score += score_diff

                    mutant_scores.append(total_score)

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
        with batching for efficiency
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        # Validate input parameters
        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")

        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length"
            )

        # Validate positions
        target = self.system[0]
        self.valid_positions(positions, entities=0, raise_invalid=True)

        # Convert sequences to strings if needed
        seqs = []
        for instance in instances:
            seq = instance[0].rep
            seq = "".join(seq)
            seqs.append(seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            conditionals_list = []

            # Process in batches
            for batch_start in range(0, len(seqs), self.decoder_batch_size):
                batch_end = min(
                    batch_start + self.decoder_batch_size, len(seqs)
                )
                batch_seqs = seqs[batch_start:batch_end]
                batch_positions = positions[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                batch_entities = entities[batch_start:batch_end]

                # Prepare batch data with tokenizer
                inputs = self.tokenizer(
                    batch_seqs, return_tensors="pt", padding=True
                ).to(self.device)

                with torch.no_grad():
                    # Forward pass for the entire batch
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # Process each sequence in the batch
                    for batch_idx, (orig_idx, pos, entity) in enumerate(
                            zip(batch_indices, batch_positions, batch_entities)
                    ):
                        pos_idx = pos - target.first_index

                        # Get logits for this position (adjust for tokenizer offsets)
                        token_idx = pos_idx + 1  # +1 for start token
                        pos_logits = logits[batch_idx, token_idx]

                        # Convert to amino acid probabilities
                        aa_probs = {}
                        for aa in target.alphabet(include_gap=False):
                            if aa == '-':  # Skip gap character
                                aa_probs[aa] = 0.0
                            else:
                                aa_token = self.tokenizer.convert_tokens_to_ids(aa)
                                aa_probs[aa] = pos_logits[aa_token].item()

                        # Store results
                        conditionals_list.append({
                            'instance': orig_idx,
                            'entity': entity,
                            'pos': pos,
                            **aa_probs
                        })

        # Create dataframe with proper index format
        conditionals = pd.DataFrame(conditionals_list)
        conditionals = conditionals.set_index(['instance', 'entity', 'pos'])

        return conditionals

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None   # noqa
    ) -> List[SystemInstance]:
        """
        Transform system instances by adding embeddings from the ESM2 model
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        # Default to entity 0 if not specified
        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            transformed_instances = []

            # Process in batches
            for batch_start in range(0, len(instances), self.decoder_batch_size):
                batch_end = min(
                    batch_start + self.decoder_batch_size, len(instances)
                )
                batch_instances = instances[batch_start:batch_end]

                # Prepare batch sequences
                sequences = []
                for instance in batch_instances:
                    seq = instance[0].rep
                    seq = "".join(seq)
                    sequences.append(seq)

                # Tokenize sequences
                inputs = self.tokenizer(
                    sequences, return_tensors="pt", padding=True
                ).to(self.device)

                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # Get the hidden states from the last layer
                    # Note: For EsmForMaskedLM, the hidden states are typically accessed as:
                    # hidden_states = outputs.hidden_states[-1]
                    # Last layer hidden states
                    hidden_states = outputs.hidden_states[-1]

                    # Process each instance in the batch
                    for i, instance in enumerate(batch_instances):
                        # Create new entity instance
                        entity_instance = instance[0]

                        # Create new entity with proper initialization
                        new_entity = EntityInstance(
                            rep=entity_instance.rep,
                            models=entity_instance.models  # Copy over 3D structures
                        )

                        # Copy structure attribute if it exists
                        if hasattr(entity_instance, 'structure'):
                            new_entity.structure = entity_instance.structure

                        # Copy confidence attribute if it exists
                        if hasattr(entity_instance, 'confidence'):
                            new_entity.confidence = entity_instance.confidence

                        # Copy metadata attribute if it exists
                        if hasattr(entity_instance, 'metadata'):
                            new_entity.metadata = entity_instance.metadata

                        # Create a new SystemInstance with this entity
                        new_instance = SystemInstance([new_entity])

                        # Get sequence length (excluding padding)
                        # -2 for special tokens
                        seq_len = len(self.tokenizer.encode(sequences[i])) - 2

                        # Store the embedding (excluding the first token which is the start token)
                        embedding = hidden_states[i, 1:seq_len+1].cpu().numpy()
                        new_entity.embedding = embedding

                        # Calculate and store score
                        logits = outputs.logits[i, :-1]  # exclude last token
                        token_probs = torch.log_softmax(logits, dim=-1)

                        # Get the target tokens (shifted by one)
                        target_tokens = inputs.input_ids[i, 1:seq_len+1]

                        # Calculate log probabilities for target tokens
                        seq_log_probs = torch.gather(
                            token_probs,
                            dim=1,
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)

                        new_instance.score = seq_log_probs.sum().item()

                        # Copy over original instance score if needed

                        if hasattr(instance, 'score') and not hasattr(new_instance, 'score'):
                            new_instance.score = instance.score

                        # Copy any other SystemInstance attributes
                        if hasattr(instance, 'metadata'):
                            new_instance.metadata = instance.metadata

                        transformed_instances.append(new_instance)

        return transformed_instances
