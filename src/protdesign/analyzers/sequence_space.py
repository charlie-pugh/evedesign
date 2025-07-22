"""
Functionality for reducing design dimensionality to analyze relationships between generated and natural sequences
"""
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
from numba import prange, jit
from protdesign.analysis import Analyzer
from sklearn.manifold import MDS

try:
    from umap import UMAP  # noqa
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from protdesign.entity import System, SystemInstance
from protdesign.types import EntityType
from protdesign.utils import str_to_np_char_view, map_array, index_map

# each element is (entity, list of sequences, number of system sequences at end of list)
CollectedSequences = list[tuple[int, list[str], int]]

SEQSPACE_PROJECTION_COMPONENT_KEY = "sequence_space_projection"


@jit(nopython=True, parallel=True)
def hamming_distance_no_gaps(matrix, exclude_value):
    """
    Calculate pairwise sequence distance matrix for a set of sequences

    The function will by default use the available number of threads
    (as returned by numba.get_num_threads()). If a different number should
    be used, the caller is responsible to set the number of threads with
    numba.set_num_threads

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols)
    exclude_value : int
        Value >= 0 in matrix that will be excluded from identity calculation, e.g. gap or lowercase character.
        Set to -1 to enable legacy behaviour num_cluster_members_legacy which includes gaps in identity calculation.

    Returns
    -------
    np.array
        Symmetric distance matrix normalized to range 0 to 1
    """
    N, L = matrix.shape  # noqa

    # minimal cluster size is 1 (self) but for parallelization we set the self-hit below inside the loop
    # and initialize to zero here
    dist_matrix = np.zeros((N, N))

    # compare all pairs of sequences; we cannot assume symmetry of the resulting matrix here due to exclusion of
    # gaps (this is also convenient for parallelizing the outer loop); no speedup from using a separate function
    # with regular range(N) in single-thread case so can always use this function

    for i in prange(N):
        # compare to all other sequences
        for j in range(i + 1, N):
            # differences
            dist = 0

            # total number of pairs compared
            pairs = 0

            # compare all positions
            for k in range(L):
                if matrix[i, k] != exclude_value and matrix[j, k] != exclude_value:
                    pairs += 1
                    if matrix[i, k] != matrix[j, k]:
                        dist += 1

            # avoid potential division by zero
            if pairs == 0:
                pairs = 1

            dist_norm = dist / pairs
            dist_matrix[i, j] = dist_norm
            dist_matrix[j, i] = dist_norm

    return dist_matrix


class SequenceSpaceProjection(Analyzer, ABC):
    """
    Project sequences into lower-dimensional space for visual inspection

    Note: may want to re-express this as implementation of Transformer interface;
     however this is not compatible with analyzing sequences in underlying System
    """
    def __init__(
        self,
        acceptable_entity_types: list[EntityType],
        num_components: int = 2,
        include_system_sequences: bool = True,
    ):
        """
        Create new MDS-based sequence space projector

        Parameters
        ----------
        acceptable_entity_types
            List of entity types that projector implementation can handle
        num_components
            Number of components to project sequences down to
        include_system_sequences
            If true, include system sequences besides designed sequences
        """
        self.num_components = num_components
        self.include_system_sequences = include_system_sequences
        self.acceptable_entity_types = acceptable_entity_types

    def _select_entities(
        self,
        system: System,
        entity: int | None,
    ) -> list[int]:
        """
        Helper method to determine entities used for computation

        Parameters
        ----------
        system
            System for which instances/natural sequences will be projected
        entity
            If None, use all entities, if specified, use particular entity.

        Returns
        -------
        List of selected entities
        """
        all_entities = list(range(0, len(system)))

        # either use all entities if unspecified, or restrict to selected entity
        if entity is None:
            entities = all_entities
        else:
            if entity not in all_entities:
                raise ValueError(
                    f"Invalid entity selection, valid options are {' '.join(map(str, all_entities))}"
                )

            entities = [entity]

        # make sure only entities are selected that projection method can handle (protein, DNA, ...)
        for checked_entity in entities:
            entity_type = system[checked_entity].type_
            if entity_type not in self.acceptable_entity_types:
                raise ValueError(
                    f"Entity {checked_entity} is of type {entity_type} but only the following are "
                    f"allowed: {', '.join(self.acceptable_entity_types)} "
                )

        return entities

    def _collect_sequences(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entities: list[int],
        require_aligned: bool = True,
    ) -> CollectedSequences:
        """
        Collect rep sequences for all analyzed entities

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances from which sequences will be collected
        entities
            Index of entities for which sequences will be collected
        require_aligned
            If True, requires that all instance and system sequences are aligned (same number of match states)

        Returns
        -------
        List of collected sequence reps per entity
        """
        if self.include_system_sequences:
            if len(entities) != 1:
                raise ValueError(
                    "Must specify a single entity for inclusion of system sequences as mapping may be ambiguous; "
                    "this feature may be implemented at a later time"
                )

            system_sequences = system.data[entities[0]].sequences
            if system_sequences is not None and require_aligned and not system_sequences.aligned:
                raise ValueError(
                    "System sequences must be aligned for analysis"
                )
        else:
            pass

        # assemble sequences on per-entity basis; this allows us to look at entity-specific information
        # like alphabets when performing actual computation
        all_seqs = []
        for entity in entities:
            if self.include_system_sequences and system.data[entity].sequences is not None:
                # if requiring alignment, remove dealigned positions (otherwise will rarely find an MSA
                # that could be handled)
                system_seqs = [
                    (entry.remove_insertions().seq if require_aligned else  entry.dealign().seq)
                    for entry in system.data[entity].sequences.seqs
                ]
            else:
                system_seqs = []

            # ensure all instances have a defined rep
            if any([
                instance[entity].rep is None for instance in instances
            ]):
                raise ValueError(
                    "Entity instance contains rep that is None; "
                    "for sequence space projection all instances must have specified rep"
                )

            instance_seqs = [
                "".join(instance[entity].rep if require_aligned else instance[entity].normalized_rep())
                for instance in instances
            ]

            # if requiring alignment, need to verify all sequences now have same length
            merged_seqs = system_seqs + instance_seqs

            if require_aligned:
                seq_lengths = {
                    len(seq) for seq in merged_seqs
                }
                if len(seq_lengths) != 1:
                    raise ValueError(
                        f"Aligned sequences required but input sequences have differing lengths for "
                        f"entity {entity}: {seq_lengths}"
                    )

            all_seqs.append((
                entity, merged_seqs, len(system_seqs)
            ))

        # return assembled sequences per entity
        return all_seqs

    def _add_projections(
        self,
        system,
        instances,
        projections
    ) -> tuple[System, Sequence[SystemInstance]]:
        """
        Add projections as metadata to system and instances

        Parameters
        ----------
        system
            System to which analysis results will be attached
        instances
            Instances to which analysis results will be attached
        projections
            Projections that will be attached to system and instances

        Returns
        -------
        Tuple containing results from analysis in
        (i) System
        (ii) SystemInstances
        """
        if self.include_system_sequences:
            system_projections = projections[:-len(instances)]
            instance_projections = projections[-len(instances):]
        else:
            system_projections = None
            instance_projections = projections

        # shallow copy of instances, then attach metadata
        updated_instances = [
            inst.copy() for inst in instances
        ]
        for idx, inst in enumerate(updated_instances):
            if inst.metadata is None:
                inst.metadata = {}

            inst.metadata[SEQSPACE_PROJECTION_COMPONENT_KEY] = instance_projections[idx, :].tolist()

        updated_system = system  # TODO: perform copy
        if system_projections is not None:
            print(system_projections.shape) # TODO: add projection

        return updated_system, updated_instances


class SequenceSpaceProjectionAligned(SequenceSpaceProjection):
    """
    Sequence space projection, Assuming sequences are aligned and have same length of match states;
    will discard any inserts relative to consensus from analysis
    """
    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
    ):
        """
        Initialize new sequence space projector

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        """
        super().__init__(
            acceptable_entity_types=["protein", "dna", "rna"],
            num_components=num_components,
            include_system_sequences=include_system_sequences,
        )

    @classmethod
    def _distance_matrix(
        cls,
        system: System,
        collected_sequences: CollectedSequences,
        default_value: int = -1
    ) -> np.ndarray:
        """
        Compute distance matrix from set of instance/system sequences
        (potentially for multiple entities)

        Parameters
        ----------
        system
            System for which sequences are analyzed
        collected_sequences
            Extracted rep sequences for all entities
        default_value
            Default value to map gaps/non-standard symbols to

        Returns
        -------
        Distance matrix of shape len(collected_sequences) x num_components
        """

        # map sequences for each entity to integer array for numba computation
        entity_arrays = [
            map_array(
                str_to_np_char_view(seqs),
                # do not map gaps so we can easily exclude with same value in numba calculation
                index_map(system[entity].alphabet(include_gap=False), default_value=default_value),
            )
            for (entity, seqs, _) in collected_sequences
        ]

        # merge array together across entities
        array_merged = np.concatenate(entity_arrays + entity_arrays, axis=1)

        # compute distance matrix with numba
        dist_matrix = hamming_distance_no_gaps(array_merged, default_value)
        return dist_matrix

    @abstractmethod
    def _project(self, dist_matrix: np.ndarray):
        pass

    def distances_and_projection(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform sequence space projection analysis, returning results directly
        (e.g. for interactive analysis)

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which codon-optimized DNA sequences should be created
        entity
            Index of protein entity for which DNA sequence should be created

        Returns
        -------
        Tuple containing main results from analysis:
        (i) Distance matrix
        (ii) Projection of shape num_sequences x num_components; system sequences will be first
        """
        # validate entities, in particular fixed length requirement for this class
        [
            system.valid_instance(
                instance,
                validate_reps=True,
                fixed_length=True,
                allow_deletions=True,
                raise_invalid=True,
            ) for instance in instances
        ]

        # determine selected entities (for regular MDS, can do all types of biopolymers)
        entities = self._select_entities(
            system, entity
        )

        # assemble instance data as needed, also verify they are aligned for all methods but landmark_mds_mmseqs
        sequences = self._collect_sequences(
            system, instances, entities, require_aligned=True
        )

        # compute distance matirx
        dist_matrix = self._distance_matrix(system, sequences)

        # perform projection
        projections = self._project(dist_matrix)

        return dist_matrix, projections

    def analyze(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int | None = None
    ) -> tuple[System, Sequence[SystemInstance]]:
        """
        Perform sequence space projection analysis

        Parameters
        ----------
        system
            System for which instances are provided
        instances
            Instances for which codon-optimized DNA sequences should be created
        entity
            Index of protein entity for which DNA sequence should be created

        Returns
        -------
        Tuple containing results from sequence space analysis in
        (i) System (only updated if include_system_sequences is True)
        (ii) SystemInstances
        """
        # validate entities, in particular fixed length requirement for this class
        dist_matrix, projections = self.distances_and_projection(
            system, instances, entity
        )

        # add projection to shallow copy of system and instances
        return self._add_projections(
            system, instances, projections
        )


class SequenceSpaceMDS(SequenceSpaceProjectionAligned):
    """
    Sequence space projection with multidimensional scaling, following https://github.com/debbiemarkslab/sequenceMDS
    """
    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
        mds_kwargs: dict | None = None
    ):
        """
        Initialize new sequence space projector using multidimensional scaling (MDS)

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        mds_kwargs
            Keyword arguments forwarded to constructor of sklearn.manifold.MDS
        """
        super().__init__(
            num_components=num_components,
            include_system_sequences=include_system_sequences,
        )
        self.mds_kwargs = mds_kwargs

    def _project(self, dist_matrix: np.ndarray):
        if self.mds_kwargs is None:
            params = {
                "normalized_stress": "auto",
            }
        else:
            params = self.mds_kwargs

        # following https://github.com/debbiemarkslab/sequenceMDS
        embedding = MDS(
            n_components=self.num_components,
            dissimilarity="precomputed",
            **params,
        )

        return embedding.fit_transform(dist_matrix)


class SequenceSpaceUMAP(SequenceSpaceProjectionAligned):
    available = UMAP_AVAILABLE

    def __init__(
        self,
        num_components: int = 2,
        include_system_sequences: bool = True,
        umap_kwargs: dict | None = None
    ):
        """
        Initialize new sequence space projector using multidimensional scaling (MDS)

        Parameters
        ----------
        num_components
            Number of components to project sequences to (typically 2)
        include_system_sequences
            If True, include sequences from system for analyzing designs in context of
            available sequence information
        umap_kwargs
            Keyword arguments forwarded to constructor of umap.UMAP
        """
        if not self.available:
            raise ValueError(
                "umap package is not available, please install"
            )

        super().__init__(
            num_components=num_components,
            include_system_sequences=include_system_sequences,
        )
        self.umap_kwargs = umap_kwargs

    def _project(self, dist_matrix: np.ndarray):
        if self.umap_kwargs is None:
            params = {
                # TODO: investigate sensible defaults
                "n_neighbors": 50
            }
        else:
            params = self.umap_kwargs

        # following https://github.com/debbiemarkslab/sequenceMDS
        embedding = UMAP(
            n_components=self.num_components,
            metric="precomputed",
            **params,
        )

        return embedding.fit_transform(dist_matrix)
