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
    requires_seqs: bool = False
    requires_msa: bool = False
    requires_3d: bool = False

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
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
        device
            Device to use for computations
        """
        if not self.available:
            raise ValueError(
                "ESM2 package could not be imported. Is it installed already?")

        self.model_name = model_name
        self.keep_model_after_build = keep_model_after_build
        self.keep_model_after_pred = True
        self.device = device

        self._system = None
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        self.decoder_batch_size = decoder_batch_size
        self.num_samples = num_samples

        if self.num_samples < 1:
            raise ValueError("num_samples must be > 0")

        if self.decoder_batch_size != "auto" and self.decoder_batch_size < 1:
            raise ValueError("decoder_batch_size must be at least 1 or 'auto'")

        if self.decoder_batch_size == "auto":
            raise NotImplementedError(
                "Automatic batch_size not yet implemented")

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
        return RequiredResources(
            min_gpu_cores=1,
            min_gpu_memory_per_core=16000,
            min_cpu_cores=1,
            min_cpu_memory_per_core=16000,
            max_batch_size=512,
            time=1,
        )

    def _load_model(self):
        if self.model is not None:
            return

        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", self.model_name)
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
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
        self.can_model_or_raise(system, data)
        self._system = system
        target = self.system[0]

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            data = [("protein", "".join(target.rep))]
            _, _, token_ids = self.batch_converter(data)
            token_ids = token_ids.to(self.device)

            with torch.no_grad():
                results = self.model(token_ids, repr_layers=[
                                     self.model.num_layers])
                representations = results["representations"][self.model.num_layers]
                self.token_ids = token_ids.cpu()
                self.encoding = representations.cpu()

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
        encoding_on_device = None
        token_ids_on_device = None

        try:
            if self.encoding is not None:
                encoding_on_device = self.encoding.to(self.device)
            if self.token_ids is not None:
                token_ids_on_device = self.token_ids.to(self.device)

            yield encoding_on_device, token_ids_on_device
        finally:
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
        deletions: bool = False,
        status_callback: StatusCallback | None = None,
        num_sweeps: int = 10
    ) -> List[SystemInstance]:
        """
        Generate protein sequences using the ESM2 model with the GibbsSampler
        """
        self.ready_or_raise()

        entities = entities if entities is not None else [0]
        if len(entities) != 1 or entities[0] != 0:
            raise ValueError(
                "Can only design single entity (entities = [0] | None)")

        # Adjust num_designs to be a multiple of batch_size
        if rem := num_designs % self.decoder_batch_size:
            num_designs_adj = num_designs + (self.decoder_batch_size - rem)
            logger.warning(
                f"Adjusting num_designs from {num_designs} to {num_designs_adj} to be a multiple of batch_size")
            num_designs = num_designs_adj

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            logger.info(
                f"Generating {num_designs} designs with ESM2 using GibbsSampler")

            # Create a GibbsSampler with this ESM2 model as the scorer
            sampler = GibbsSampler(
                scorers=[self],
                weights=None,
                num_sweeps=num_sweeps,
                init_strategy="random",
                scan_order="random",
                temperature_schedule=lambda init_temp, *
                args: init_temp,  # Constant temperature
                require_strict_pos=True,
                record_full_chain=False
            )

            # Generate designs
            instances = sampler.generate(
                num_designs=num_designs,
                entities=entities,
                fixed_pos=fixed_pos,
                temperature=temperature,
                deletions=deletions,
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
            instance.score = scores[i+1] - ref_score

        return instances[:num_designs]

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
                        token_probs = torch.log_softmax(logits[i, :-1], dim=-1)
                        target_tokens = batch_tokens[i, 1:]

                        seq_log_probs = torch.gather(
                            token_probs,
                            dim=1,
                            index=target_tokens.unsqueeze(1)
                        ).squeeze(1)

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
        Perform a single mutation scan for the given instance
        """
        self.ready_or_raise()
        self._validate_instances([instance])

        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        # Get sequence and convert to string if needed
        target = self.system[0]
        instance_seq = instance[0].rep
        if isinstance(instance_seq, np.ndarray):
            instance_seq = "".join(instance_seq)

        # Validate positions
        if positions is not None:
            self.valid_positions(positions, entities=0, raise_invalid=True)
        else:
            positions = list(
                range(target.first_index, target.first_index + len(target.rep)))

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            mutation_effects = []

            # Prepare reference sequence
            ref_data = [("protein", instance_seq)]
            _, _, ref_tokens = self.batch_converter(ref_data)
            ref_tokens = ref_tokens.to(self.device)

            # Calculate reference score
            with torch.no_grad():
                ref_results = self.model(ref_tokens, repr_layers=[])
                ref_logits = ref_results["logits"][0]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

                # For each position to scan
                for pos in positions:
                    pos_idx = pos - target.first_index
                    wt_aa = instance_seq[pos_idx]

                    # Score each possible substitution
                    mut_scores = {}
                    for aa in VALID_AA_OR_GAP_SORTED:
                        if aa == '-':  # Skip gap character
                            continue

                        # If same as wildtype, effect is 0
                        if aa == wt_aa:
                            mut_scores[aa] = 0.0
                            continue

                        # Create and score mutant sequence
                        mut_seq = instance_seq[:pos_idx] + \
                            aa + instance_seq[pos_idx+1:]
                        mut_data = [("mutant", mut_seq)]
                        _, _, mut_tokens = self.batch_converter(mut_data)
                        mut_tokens = mut_tokens.to(self.device)

                        mut_results = self.model(mut_tokens, repr_layers=[])
                        mut_logits = mut_results["logits"][0]
                        mut_log_probs = torch.log_softmax(mut_logits, dim=-1)

                        # Calculate log-likelihood difference
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

                        score_diff = mut_seq_log_probs.sum().item() - ref_seq_log_probs.sum().item()
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
        if isinstance(instance_seq, np.ndarray):
            instance_seq = "".join(instance_seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            # Score reference sequence
            ref_data = [("protein", instance_seq)]
            _, _, ref_tokens = self.batch_converter(ref_data)
            ref_tokens = ref_tokens.to(self.device)

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
                    # Apply mutations to sequence
                    mut_seq = list(instance_seq)
                    for sub in mutant:
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
        self._validate_instances(instances)

        # Validate input parameters
        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")

        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length")

        # Validate positions
        target = self.system[0]
        self.valid_positions(positions, entities=0, raise_invalid=True)

        # Convert sequences to strings if needed
        seqs = []
        for instance in instances:
            seq = instance[0].rep
            if isinstance(seq, np.ndarray):
                seq = "".join(seq)
            seqs.append(seq)

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_pred):
            conditionals_list = []

            # Process each sequence and position pair
            for i, (seq, pos) in enumerate(zip(seqs, positions)):
                pos_idx = pos - target.first_index

                # Prepare the sequence
                seq_data = [("protein", seq)]
                _, _, seq_tokens = self.batch_converter(seq_data)
                seq_tokens = seq_tokens.to(self.device)

                with torch.no_grad():
                    # Get conditional probabilities at the specified position
                    results = self.model(seq_tokens, repr_layers=[])
                    logits = results["logits"][0]
                    pos_logits = logits[pos_idx + 1]  # +1 for BOS token
                    pos_probs = torch.softmax(pos_logits, dim=-1)

                    # Convert to amino acid probabilities
                    aa_probs = {}
                    for aa in VALID_AA_OR_GAP_SORTED:
                        if aa == '-':  # Skip gap character
                            aa_probs[aa] = 0.0
                        else:
                            aa_token = self.alphabet.get_idx(aa)
                            aa_probs[aa] = pos_probs[aa_token].item()

                # Store results
                conditionals_list.append({
                    'instance': i,
                    'entity': entities[i],
                    'pos': positions[i],
                    **aa_probs
                })

        # Create dataframe with proper index format
        conditionals = pd.DataFrame(conditionals_list)
        conditionals = conditionals.set_index(['instance', 'entity', 'pos'])

        return conditionals
