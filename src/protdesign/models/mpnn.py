import tempfile
import os
from typing import List, Dict, Sequence, Tuple, Optional, Self, Literal
import copy
import urllib.request

import numpy as np
import torch
from loguru import logger

from protdesign.model import (
    BaseModel, Scorer, Generator, RequiredResources
)
from protdesign.entity import System, SystemInstance, EntityInstance, EntityPosList
from protdesign.structure import Model
from protdesign.utils import ensure_sequence, model_param_context
from protdesign.constants import MASK
from protdesign.types import DeviceType, StatusCallback, BatchSize

# Import the LigandMPNN modules
from protdesign.models.ligandmpnn.data_utils import (
    featurize,
    parse_PDB,
    restype_str_to_int,
    restype_int_to_str,
    get_score,
)
from protdesign.models.ligandmpnn.model_utils import ProteinMPNN
try:
    import prody
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

MODEL_BASE_URL = "https://files.ipd.uw.edu/pub/ligandmpnn"

# Model checkpoint URLs
MODEL_URLS = {
    # Original ProteinMPNN weights
    "proteinmpnn_v_48_002": f"{MODEL_BASE_URL}/proteinmpnn_v_48_002.pt",
    "proteinmpnn_v_48_010": f"{MODEL_BASE_URL}/proteinmpnn_v_48_010.pt",
    "proteinmpnn_v_48_020": f"{MODEL_BASE_URL}/proteinmpnn_v_48_020.pt",
    "proteinmpnn_v_48_030": f"{MODEL_BASE_URL}/proteinmpnn_v_48_030.pt",
    # LigandMPNN with num_edges=32; atom_context_num=25
    "ligandmpnn_v_32_005_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_005_25.pt",
    "ligandmpnn_v_32_010_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_010_25.pt",
    "ligandmpnn_v_32_020_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_020_25.pt",
    "ligandmpnn_v_32_030_25": f"{MODEL_BASE_URL}/ligandmpnn_v_32_030_25.pt",
    # Per residue label membrane ProteinMPNN
    "per_residue_label_membrane_mpnn_v_48_020": f"{MODEL_BASE_URL}/per_residue_label_membrane_mpnn_v_48_020.pt",
    # Global label membrane ProteinMPNN
    "global_label_membrane_mpnn_v_48_020": f"{MODEL_BASE_URL}/global_label_membrane_mpnn_v_48_020.pt",
    # SolubleMPNN
    "solublempnn_v_48_002": f"{MODEL_BASE_URL}/solublempnn_v_48_002.pt",
    "solublempnn_v_48_010": f"{MODEL_BASE_URL}/solublempnn_v_48_010.pt",
    "solublempnn_v_48_020": f"{MODEL_BASE_URL}/solublempnn_v_48_020.pt",
    "solublempnn_v_48_030": f"{MODEL_BASE_URL}/solublempnn_v_48_030.pt",
    # LigandMPNN for side-chain packing (multi-step denoising model)
    "ligandmpnn_sc_v_32_002_16": f"{MODEL_BASE_URL}/ligandmpnn_sc_v_32_002_16.pt",
}

def download_checkpoint(model_name: str, save_dir: str) -> str:
    """
    Download model checkpoint from URL if not already present.

    Args:
        model_name: Name of the model to download
        save_dir: Directory to save the checkpoint

    Returns:
        Path to the downloaded checkpoint

    Raises:
        ValueError: If model_name is not recognized
        RuntimeError: If download fails
    """
    if model_name not in MODEL_URLS:
        available_models = ", ".join(MODEL_URLS.keys())
        raise ValueError(
            f"Model '{model_name}' not recognized. Available models: {available_models}"
        )

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Construct file path
    checkpoint_path = os.path.join(save_dir, f"{model_name}.pt")

    # Download if not already present
    if not os.path.exists(checkpoint_path):
        url = MODEL_URLS[model_name]
        logger.info(f"Downloading {model_name} from {url}...")
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"Successfully downloaded to {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name}: {str(e)}")
    else:
        logger.info(f"Using cached checkpoint at {checkpoint_path}")

    return checkpoint_path


class LigandMPNNWrapper(BaseModel, Scorer, Generator):
    """
    evedesign wrapper for LigandMPNN

    # TODO: extend to also handle ligand entities
    # TODO: implement specialized scoring methods that move known positions to front to score all substitutions at once
    """
    available = IMPORT_AVAILABLE
    name: str = "LigandMPNN"
    citations: list[str] = ["doi: 10.1038/s41592-025-02626-1"]

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
    requires_3d: bool = True

    def __init__(
        self,
        model_name: Literal[tuple(MODEL_URLS.keys())],  # noqa
        checkpoint_path: str | None = None,
        batch_size: BatchSize = 1,
        use_ligand_context: bool = True,
        ligand_cutoff: float = 6.0,
        keep_model_after_build: bool = False,
        cache_dir: str | None = "./model_params",
        device: DeviceType = "cpu"
    ):
        """
        Initialize the LigandMPNN wrapper

        Parameters
        ----------
        model_name
            Name of MPNN model. If checkpoint_path is specified, must match the loaded model.
        checkpoint_path
            Path to checkpoint file to load. If None, will attempt to download from web.
        batch_size
            Batch sized used for generation. Will not be used while scoring due to implementation limitations
            inside original MPNN code.
        use_ligand_context
            If True, ligand atoms will be included during calculations
        keep_model_after_build
            If True, keep model parameters asssociated to instance after build step
            to avoid reloading when scoring/generating. If serializing model, set to
            False to avoid storing model parameters repeatedly.
        ligand_cutoff
            Cutoff distance in angstroms to select residues that are considered to be close to ligand atoms
        cache_dir
            Directory to use for storing downloaded model parameters (only relevant if checkpoint_path is None)
        device
            Device to use for computations
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_ligand_context = use_ligand_context
        self.keep_model_after_build = keep_model_after_build
        self.ligand_cutoff = ligand_cutoff

        # Determine model type from model_name
        if "ligand" in model_name.lower():
            self.model_type = "ligand_mpnn"
        else:
            self.model_type = "protein_mpnn"

       # Handle checkpoint path
        if checkpoint_path is None:
            # Download from web using model_name
            self.checkpoint_path = download_checkpoint(model_name, cache_dir)
        else:
            self.checkpoint_path = checkpoint_path

        self.model = None

        # State that gets set during build()
        self._system = None
        self._feature_dict = None
        self._entity_lengths = None
        self._symmetry_residues = None
        self._symmetry_weights = None
        self._native_seq = None
        self._pdb_path = None
        self._pdb_to_entity_mapping = None  # Map PDB positions to entity positions
        self._entity_to_pdb_chains = None  # Map entity_idx to list of PDB chain IDs

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

        # Check that all entities are proteins with structures
        for entity in system:
            if entity.type_ != "protein":
                return False, "Can only handle protein entities"
            if not entity.structures or len(entity.structures) == 0:
                return False, "All entities must have 3D structures"

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

    def _load_model(self):
        """
        Load the model from checkpoint
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract model parameters
        if self.model_type == "ligand_mpnn":
            atom_context_num = checkpoint.get("atom_context_num", 25)
            k_neighbors = checkpoint.get("num_edges", 32)
        else:
            atom_context_num = 1
            k_neighbors = checkpoint.get("num_edges", 48)

        # Initialize model
        self.model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type=self.model_type,
            ligand_mpnn_use_side_chain_context=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def _delete_model(self):
        self.model = None
        self._release_cache()

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system

        # Get entity sequence lengths
        self._entity_lengths = [
            (idx, len(entity.rep) if entity.rep is not None else 0)
            for idx, entity in enumerate(system)
        ]

        # Convert system to PDB file and build mappings simultaneously
        self._pdb_path, self._pdb_to_entity_mapping, self._entity_to_pdb_chains = (
            self._system_to_pdb_file(system)
        )

        print(self._pdb_path)  # TODO: remove again

        # Parse PDB with LigandMPNN
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            self._pdb_path,
            device=self.device,
            chains=[],
            parse_all_atoms=True,
            parse_atoms_with_zero_occupancy=False,
        )

        # Build symmetry constraints directly from entity_to_pdb_chains
        self._symmetry_residues, self._symmetry_weights = (
            self._build_symmetry_from_chains(protein_dict)
        )

        # Set up chain mask (which residues to design)
        chain_mask = torch.ones_like(protein_dict["mask"], dtype=torch.float32)
        protein_dict["chain_mask"] = chain_mask

        # Featurize the protein
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            self._feature_dict = featurize(
                protein_dict,
                cutoff_for_score=self.ligand_cutoff,
                use_atom_context=self.use_ligand_context,
                number_of_ligand_atoms=getattr(self.model, 'atom_context_num', 25),
                model_type=self.model_type,
            )

        # Store native sequence
        self._native_seq = "".join([
            restype_int_to_str[aa] for aa in self._feature_dict["S"][0].cpu().numpy()
        ])

        return self

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> List[Tuple[int, int]]:
        """Get all designable positions in the system."""
        self.ready_or_raise()
        positions = []
        for entity_idx, entity in enumerate(self.system):
            if entity.rep is not None:
                first_idx = entity.first_index if entity.first_index is not None else 0
                for pos in range(first_idx, first_idx + len(entity.rep)):
                    positions.append((entity_idx, pos))
        return positions

    def _system_to_pdb_file(self, system: System) -> Tuple[str, Dict, Dict]:
        """
        Convert a System object to a temporary PDB file.
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdb')
        os.close(temp_fd)

        structure_keys = set()
        for entity in system:
            structure_keys.update(entity.structures.keys()
                                  ) if entity.structures else None

        if len(structure_keys) > 1:
            raise NotImplementedError(
                "Multi-state design not currently supported")

        structure_key = list(structure_keys)[0] if structure_keys else None

        # Track which entity each chain belongs to
        entity_to_pdb_chains = {i: [] for i in range(len(system))}
        pdb_to_entity_mapping = {}

        # Use single letter chain IDs: A, B, C, ..., Z, AA, AB, etc.
        def get_chain_id(chain_num: int) -> str:
            """Generate chain ID: A-Z, then AA, AB, AC, ..."""
            if chain_num < 26:
                return chr(65 + chain_num)  # A-Z
            else:
                # AA, AB, AC, ...
                first = chr(65 + (chain_num - 26) // 26)
                second = chr(65 + (chain_num - 26) % 26)
                return first + second

        chain_counter = 0
        current_pdb_pos = 0
        models_to_concat = []

        for entity_idx, entity in enumerate(system):
            if entity.structures and structure_key in entity.structures:
                entity_chains = entity.structures[structure_key]
                if not isinstance(entity_chains, list):
                    entity_chains = [entity_chains]

                for chain_obj in entity_chains:

                    # Perform deep copy
                    model_copy = Model(copy.deepcopy(chain_obj.atom_array))

                    # Assign new chain ID
                    new_chain_id = get_chain_id(chain_counter)
                    entity_to_pdb_chains[entity_idx].append(new_chain_id)

                    # Modify chain_id directly in the Model's atom array
                    model_copy.atom_array.chain_id[:] = new_chain_id

                    # Build position mapping using residue table
                    res_df = model_copy.res_df()
                    for entity_pos in range(len(res_df)):
                        pdb_to_entity_mapping[current_pdb_pos] = (
                            entity_idx, entity_pos)
                        current_pdb_pos += 1

                    models_to_concat.append(model_copy)
                    chain_counter += 1

        # Use Model.concat() to merge all models
        if models_to_concat:
            combined_model = Model.concat(models_to_concat)
            # Use Model.to_file() to write to PDB
            combined_model.to_file(temp_path, format='pdb')

        return temp_path, pdb_to_entity_mapping, entity_to_pdb_chains

    def _replace_chain_id(self, pdb_content: str, new_chain_id: str) -> str:
        """Replace chain IDs in PDB content."""
        lines = []
        for line in pdb_content.split('\n'):
            if line.startswith(('ATOM', 'HETATM', 'TER')):
                # Chain ID is at position 21 (0-indexed: 21)
                if len(line) > 21:
                    line = line[:21] + new_chain_id + line[22:]
            lines.append(line)
        return '\n'.join(lines)

    def _build_symmetry_from_chains(self, protein_dict: Dict) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Build symmetry constraints from entity_to_pdb_chains mapping.
        """
        symmetry_residues = []
        symmetry_weights = []

        # Get actual chain lengths from parsed PDB
        chain_list = protein_dict['chain_list']
        mask_c = protein_dict['mask_c']

        # Build chain_id to length mapping
        chain_lengths = {}
        for i, chain_id in enumerate(chain_list):
            chain_length = mask_c[i].sum().item()
            chain_lengths[chain_id] = chain_length

        # For each entity with multiple chains (homo-oligomer)
        for entity_idx, pdb_chain_ids in self._entity_to_pdb_chains.items():
            if len(pdb_chain_ids) > 1:
                # Get lengths of all chains for this entity
                entity_chain_lengths = [chain_lengths.get(
                    cid, 0) for cid in pdb_chain_ids]

                # Use minimum length to avoid out-of-bounds
                min_chain_length = min(
                    entity_chain_lengths) if entity_chain_lengths else 0

                # Calculate starting position for each chain in concatenated sequence
                chain_start_positions = []
                cumulative_pos = 0
                for chain_id in chain_list:
                    if chain_id in pdb_chain_ids:
                        chain_start_positions.append(cumulative_pos)
                    cumulative_pos += chain_lengths.get(chain_id, 0)

                # Create symmetry groups
                num_chains = len(pdb_chain_ids)
                for pos in range(min_chain_length):
                    residue_group = []
                    weight_group = []

                    for start_pos in chain_start_positions:
                        residue_idx = start_pos + pos
                        residue_group.append(residue_idx)
                        weight_group.append(1.0 / num_chains)

                    symmetry_residues.append(residue_group)
                    symmetry_weights.append(weight_group)

        return symmetry_residues, symmetry_weights

    def _split_concatenated_sequences(self, concatenated_sequences: List[str],
                                      entity_lengths: List[Tuple[int, int]]) -> Dict[int, List[str]]:
        """
        Split concatenated sequences back into per-entity sequences.
        Handles cases where PDB sequence != entity sequence due to missing residues.
        """
        separated_sequences = {entity_idx: []
                               for entity_idx, _ in entity_lengths}

        for concat_seq in concatenated_sequences:
            # Initialize entity sequences with mask character for missing density
            entity_seqs = {entity_idx: [MASK] * length  # Changed from ['X']
                           for entity_idx, length in entity_lengths}

            # Fill in positions that exist in PDB
            for pdb_pos, aa in enumerate(concat_seq):
                if pdb_pos in self._pdb_to_entity_mapping:
                    mapped_entity_idx, entity_pos = self._pdb_to_entity_mapping[pdb_pos]
                    entity_length = dict(entity_lengths).get(
                        mapped_entity_idx, 0)
                    if entity_pos < entity_length:
                        entity_seqs[mapped_entity_idx][entity_pos] = aa

            # Add to results
            for entity_idx, _ in entity_lengths:
                separated_sequences[entity_idx].append(
                    ''.join(entity_seqs[entity_idx]))

        return separated_sequences

    def _create_chain_mask(self, fixed_pos: EntityPosList | None) -> torch.Tensor:
        """
        Create chain mask from fixed positions.

        """
        chain_mask = torch.ones_like(
            self._feature_dict["mask"], dtype=torch.float32)

        if fixed_pos is not None:
            # Use PDB-to-entity mapping to convert entity positions to PDB positions
            for entity_idx, positions in fixed_pos.items():
                for entity_pos in positions:
                    # Find corresponding PDB position(s)
                    for pdb_pos, (mapped_entity_idx, mapped_entity_pos) in self._pdb_to_entity_mapping.items():
                        if mapped_entity_idx == entity_idx and mapped_entity_pos == entity_pos:
                            chain_mask[0, pdb_pos] = 0.0

        return chain_mask

    def _create_bias_tensor(self, amino_acid_bias: Dict[str, float]) -> torch.Tensor:
        """Create bias tensor from amino acid bias dictionary."""
        bias_tensor = torch.zeros(
            [21], device=self.device, dtype=torch.float32)
        for aa, bias in amino_acid_bias.items():
            if aa in restype_str_to_int:
                bias_tensor[restype_str_to_int[aa]] = bias
        return bias_tensor

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 0.1,
        deletions: bool = False,
        status_callback: StatusCallback | None = None,
        amino_acid_bias: Optional[Dict[str, float]] = None,
        omit_amino_acids: Optional[str] = None,
        use_ligand_context: bool = True,
    ) -> List[SystemInstance]:
        """
        Generate new sequences for the built structure and optionally score them.
        # TODO: additional parameter documentation
        """
        # 1. Check model is ready
        self.ready_or_raise()

        if deletions:
            raise ValueError("LigandMPNN does not support deletions")

        # Use batch_size from constructor
        batch_size = self.batch_size

        # Random seed was already set in constructor if provided

        # 2. Validate entity selection
        if entities is not None:
            entities = ensure_sequence(entities)
            # Validate entities exist in system
            max_entity = len(self.system) - 1
            for entity_idx in entities:
                if entity_idx > max_entity:
                    raise ValueError(
                        f"Entity index {entity_idx} out of range (max: {max_entity})")
        else:
            entities = list(range(len(self.system)))

        # 3. Process fixed_pos into chain_mask
        chain_mask = self._create_chain_mask(fixed_pos)

        # 4. Update feature_dict with generation parameters
        feature_dict_copy = self._feature_dict.copy()
        feature_dict_copy["chain_mask"] = chain_mask
        feature_dict_copy["batch_size"] = batch_size
        feature_dict_copy["temperature"] = temperature
        feature_dict_copy["symmetry_residues"] = self._symmetry_residues or [[]]
        feature_dict_copy["symmetry_weights"] = self._symmetry_weights or [[]]

        # 5. Apply amino acid biases (always set bias tensor)
        B, L, _, _ = feature_dict_copy["X"].shape
        if amino_acid_bias:
            bias_tensor = self._create_bias_tensor(amino_acid_bias)
        else:
            bias_tensor = torch.zeros(
                [21], device=self.device, dtype=torch.float32)
        feature_dict_copy["bias"] = bias_tensor[None, None, :].repeat(1, L, 1)

        # 6. Generate sequences using the model
        L = feature_dict_copy["X"].shape[1]
        generated_sequences = []

        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            with torch.no_grad():
                num_batches = (num_designs + batch_size - 1) // batch_size
                for batch_idx in range(num_batches):
                    if status_callback:
                        progress = ((batch_idx + 1) / num_batches) * 100
                        status_callback(
                            "running", progress, f"Generating batch {batch_idx + 1}/{num_batches}")

                    feature_dict_copy["randn"] = torch.randn(
                        [batch_size, L], device=self.device)
                    output_dict = self.model.sample(feature_dict_copy)
                    generated_sequences.append(output_dict["S"])

        S_stack = torch.cat(generated_sequences, 0)[:num_designs]

        # 7. Convert to sequences and split by entity
        concatenated_sequences = [
            "".join([restype_int_to_str[aa]
                    for aa in S_stack[i].cpu().numpy()])
            for i in range(S_stack.shape[0])
        ]

        separated_sequences = self._split_concatenated_sequences(
            concatenated_sequences, self._entity_lengths
        )

        # 8. Create SystemInstance objects
        system_instances = []
        for design_idx in range(num_designs):
            entity_instances = []

            for entity_idx, (entity_id, length) in enumerate(self._entity_lengths):
                generated_seq = separated_sequences[entity_idx][design_idx]

                # Create EntityInstance
                entity_instance = EntityInstance(
                    rep=generated_seq
                )
                entity_instances.append(entity_instance)

            # Create SystemInstance with None score/confidence (will be filled in next step)
            system_instance = SystemInstance(
                entity_instances=entity_instances,
                score=None,
                confidence=None
            )
            system_instances.append(system_instance)

        # 9. Score the generated instances
        scores = self.score(system_instances, status_callback=status_callback)

        # 10. Attach scores and confidence to instances
        for instance, raw_score in zip(system_instances, scores):
            instance.score = raw_score
            instance.confidence = raw_score

        return system_instances

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray:
        """
        Score sequences against the built structure.

        Args:
            instances: Sequence of SystemInstance objects to score
            status_callback: Optional callback for status updates

        Returns:
            Numpy array of scores (log probabilities)
        """
        # 1. Check model is ready
        self.ready_or_raise()

        # 2. Validate instance sequence lengths match the built system
        for instance_idx, instance in enumerate(instances):
            for entity_idx, (entity_instance, system_entity) in enumerate(zip(instance, self.system)):
                if entity_instance.rep is not None and system_entity.rep is not None:
                    # Get the sequence length
                    entity_seq = ''.join(entity_instance.rep) if isinstance(
                        entity_instance.rep, np.ndarray) else str(entity_instance.rep)
                    expected_length = len(system_entity.rep)

                    if len(entity_seq) != expected_length:
                        raise ValueError(
                            f"Instance {instance_idx}, entity {entity_idx} has length {len(entity_seq)}, "
                            f"but system entity has length {expected_length}"
                        )

        # 3. Extract sequences from instances and convert to PDB positions
        sequences = []
        for instance in instances:
            # Reconstruct full PDB sequence (length = total PDB residues)
            pdb_length = len(self._native_seq)
            pdb_seq = [MASK] * pdb_length  # Changed from ['X']

            # Fill in the PDB sequence from entity sequences
            for entity_idx, entity_instance in enumerate(instance):
                entity_seq = ''.join(entity_instance.rep) if isinstance(
                    entity_instance.rep, np.ndarray) else str(entity_instance.rep)

                # Map entity positions back to PDB positions
                for pdb_pos, (mapped_entity_idx, entity_pos) in self._pdb_to_entity_mapping.items():
                    if mapped_entity_idx == entity_idx and entity_pos < len(entity_seq):
                        pdb_seq[pdb_pos] = entity_seq[entity_pos]

            sequences.append(''.join(pdb_seq))

        # 4. Score each sequence
        scores = []
        with model_param_context(self._load_model, self._delete_model, self.keep_model_after_build):
            with torch.no_grad():
                for seq_idx, seq in enumerate(sequences):
                    if status_callback:
                        progress = ((seq_idx + 1) / len(sequences)) * 100
                        status_callback(
                            "running", progress, f"Scoring sequence {seq_idx + 1}/{len(sequences)}")

                    # Convert sequence to tensor
                    S_tensor = torch.tensor(
                        [restype_str_to_int.get(aa, 20) for aa in seq],
                        device=self.device,
                        dtype=torch.int64
                    )[None, :]

                    # Create feature dict for this sequence
                    feature_dict_copy = self._feature_dict.copy()
                    feature_dict_copy["S"] = S_tensor
                    feature_dict_copy["batch_size"] = 1
                    feature_dict_copy["randn"] = torch.randn(
                        [1, len(seq)], device=self.device)
                    feature_dict_copy["symmetry_residues"] = [[]]
                    feature_dict_copy["symmetry_weights"] = [[]]

                    # Score the sequence
                    output_dict = self.model.score(
                        feature_dict_copy, use_sequence=True)

                    # Calculate loss (negative log probability)
                    loss, _ = get_score(
                        output_dict["S"],
                        output_dict["log_probs"],
                        self._feature_dict["mask"][:1]
                    )

                    # Convert to positive log likelihood
                    scores.append(-loss.item())

        # 5. Return as numpy array
        return np.array(scores)

    def __del__(self):
        """Cleanup temporary files"""
        if hasattr(self, 'pdb_path') and self._pdb_path and os.path.exists(self._pdb_path):
            os.unlink(self._pdb_path)
