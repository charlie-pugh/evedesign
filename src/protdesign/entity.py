"""
Specification of components of molecular design system (proteins, nucleic acids, ligands, etc.)
"""
from collections import UserList
from collections.abc import Sequence
from typing import Mapping, NamedTuple
import numpy as np
from protdesign.sequence import valid_protein_sequence, Sequences
from protdesign.structure import StructureChainMap
from protdesign.types import EntityType, Metadata
from protdesign.constants import VALID_AA_OR_GAP_SORTED, VALID_AA_SORTED
from protdesign.utils import ensure_sequence, shorten

# Data structures/types for providing mutation information in structured format
Mutation = NamedTuple(
    "Mutation", [("entity", int), ("pos", int), ("ref", str), ("to", str)]
)

# Mutant is comprised of one or more mutations
Mutant = Sequence[Mutation]


class Entity:
    def __init__(
        self,
        type: EntityType,  # noqa
        rep: str | None = None,
        id: str | None = None,  # noqa
        copies: int | None = None,
        first_index: int | None = None,
        sequences: Sequences | None = None,
        structures: StructureChainMap | None = None,
    ):
        """
        Create new generic entity for molecular system.

        Note: For clarity, preferentially use subclasses for specific types
        of entities (e.g. Protein class)

        # TODO: additional attributes to be added right away
            * mapping to sequences/structures - implement right away

        # TODO: parameters to be added at later point
            * modifications
            * different states / conformations
            * designable or not?
            * hotspots, pair restraints / constraints

        Parameters
        ----------
        type
            Type of entity (protein, nucleotide, ligand, ...)
        id
            Unique identifier of entity
        rep
            Representation of entity (sequence, atom name, etc.)
        first_index
            Sequence index of first residue; must be specified
            for polymer types (protein, nucleotide, ...)
        copies
            Number of entity copies in molecular system. Set to None
            to leave variable.
                sequences
        sequences
            Sequence record (e.g. multiple sequence alignment of homologs) of the target
            sequence represented by this entity (only applies to proteins and nucleotides)
        structures
            Structure chains representing this entity. Use dict with structure identifiers
            as keys to supply multiple different structures; use list to supply multiple copies
            of the chain within the structure (homooligomer)
        """
        self.type_ = type
        self.rep = rep
        self.id_ = id
        self.copies = copies

        # TODO: also allow for nucleotide entities once implemented
        if self.type_ != "protein" and sequences is not None:
            raise ValueError(
                "Sequence record only supported for biopolymer entities"
            )

        self.sequences = sequences
        self.structures = structures

        if self.type_== "protein" and first_index is None:
            raise ValueError(
                f"first_index must be specified for type {self.type_}"
            )

        self.first_index = first_index

    def __eq__(self, other):
        # only ever accept other entities for equality
        if not isinstance(other, Entity):
            return False

        # do not compare sequences and structures are these are auxiliary resources
        # for modeling the entity
        return (
            self.type_ == other.type_ and
            self.rep == other.rep and
            self.id_ == other.id_ and
            self.copies == other.copies and
            self.first_index == other.first_index
        )

    def defined_sequence(self) -> bool:
        """
        Check if entity corresponds to a biopolymer (protein, ...)
        and has a defined representative with non-zero length

        Returns
        -------
        True if protein/nucleotide sequence with some defined length
        """
        return (
            self.type_ == "protein" and
            self.rep is not None and
            len(self.rep) > 0 and
            self.first_index is not None
        )

    def alphabet(self, include_gap: bool=True) -> list[str]:
        """
        Return sequence alphabet for biopolymer entities

        Parameters
        ----------
        include_gap
            If true, add gap symbol to alphabet

        Returns
        -------
        Alphabet for representing primary sequence of entity
        """
        if self.type_ == "protein":
            if include_gap:
                return VALID_AA_OR_GAP_SORTED
            else:
                return VALID_AA_SORTED
        else:
            raise NotImplementedError("Non-protein alphabets not yet implemented")

Embedding = np.ndarray[tuple[int, int], np.dtype[float]] | np.ndarray[tuple[int], np.dtype[float]]

class EntityInstance:
    """
    Instantiation of a single entity in a system
    """
    def __init__(
        self,
        rep: str | None = None,
        embedding:  Embedding | None = None,
        models: StructureChainMap | None = None,
    ):
        """
        Create new instantiation of an entity in a sequence

        Parameters
        ----------
        rep
            Uniquely defining representation (e.g. primary sequence) of entity. Set to None if no
            representation is yet available (e.g. just structural backbone but no sequence)
        embedding
            Transformation of entity instance into per-residue embedding (2D array) or
            per-entity embedding (1D array) space
        models
            Structural models associated with each of the entities in the system.
            Set to None if no structural models are available.
        """
        self.rep = rep
        self.embedding = embedding
        self.models = models

    def __repr__(self):
        if self.models is not None:
            structure_info = len(self.models)
        else:
            structure_info = self.models

        return f"EntityInstance(rep={shorten( self.rep)}, structure_models={structure_info})"


class SystemInstance(UserList):
    """
    Result designing the representations of the entity/entities
    in a system, comprised of individual EntityInstances (one per entity),
    mirroring the "System" class comprised of entities
    """
    def __init__(
        self,
        entity_instances: EntityInstance | Sequence[EntityInstance],
        score: float | None = None,
        confidence: float | None = None,
        metadata: Metadata | None = None,
    ):
        """
        Create new entity system instance

        # TODO: activate metadata attribute once needed

        Parameters
        ----------
        entity_instances
            One or more entity instances (must match entities in corresponding System)
        score
            Score describing quality/likelihood of the designed system instance
            (higher is better, ideally in logits)
        confidence
            Reliability of model score from 0 (lowest confidence) to 1 (highest confidence)
        """
        # turn single instance into list of instances
        entity_instances = ensure_sequence(entity_instances)
        super().__init__(entity_instances)

        self.score = score
        self.confidence = confidence
        self.metadata = metadata

    def __repr__(self):
        return f"SystemInstance({self.data} score={self.score})"


class System(UserList):
    def __init__(self, entities: Entity | Sequence[Entity]):
        """
        Create new biomolecular system for modeling/design

        Parameters
        ----------
        entities
            One or more entities comprising the system
        """
        # turn single entity into list of entities
        entities = ensure_sequence(entities)
        super().__init__(entities)

    def __eq__(self, other):
        # only ever accept other systems for equality
        if not isinstance(other, System):
            return False

        # systems must have same length
        if not len(self) == len(other):
            return False

        # two systems are equal if all contained entities are equal
        # (in same order)
        for ent_self, ent_other in zip(self, other):
            if ent_self != ent_other:
                return False

        return True

    def valid_instance(
        self,
        instance: SystemInstance,
        fixed_length: bool=False,
        validate_reps: bool=False,
        raise_invalid: bool=False,
    ) -> bool:
        """
        Verify if instance is valid representation of this biomolecular system

        Parameters
        ----------
        instance
            System instance to validate
        fixed_length
            If True, require that length of instance sequence matches the system entity representation length
            (only sensible for fixed-length models and biopolymers)
        validate_reps
            If True, verify if sequence representations are comprised of valid amino acids/nucleotides
        raise_invalid
            If True, raise ValueError if instance is invalid w.r.t. system

        Returns
        -------
        True if valid instance, False otherwise
        """
        # instance representations always must have same length as number of entities
        # in system by convention
        valid = len(self.data) == len(instance)

        for entity, entity_instance in zip(self.data, instance):
            # TODO: also implement comparison for nucleotides eventually
            if entity.type_ == "protein":
                if fixed_length:
                    valid = valid and (entity.rep is None or len(entity.rep) == len(entity_instance.rep))

                if validate_reps:
                    # allow gaps for deletion modeling, and mask for leaving parts of representation unspecified
                    is_valid_seq = valid_protein_sequence(
                        entity_instance.rep, allow_mask=True, allow_gap=True, allow_ambiguous=False
                    )

                    valid = valid and is_valid_seq

        if not valid and raise_invalid:
            raise ValueError("Provided instance is not valid for biomolecular system")

        return valid

    def valid_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        allow_gap: bool = False,
        raise_invalid: bool = False,
    ) -> tuple[bool, list[tuple[int, Mutation]]]:
        """
        Validate mutants against a system instance

        Parameters
        ----------
        instance
            System instance to check against; assuming this has been previously validated with valid_instance().
        mutants
            Verify these mutants against system instance
        allow_gap
            If True, consider gap symbol a valid substitution
        raise_invalid
            Raise ValueError if any invalid mutants are detected

        Returns
        -------
        invalid
            True if all mutants are valid, False otherwise
        invalid_subs
            Tuple of mutant indies and invalid mutations in these mutants (empty if all mutants are valid)
        """
        # create mapping of valid position and reference symbol in each biopolymer entity instance with defined
        # sequence and first_index
        entity_to_pos = {
            entity_idx: {
                pos: ref_symbol for (pos, ref_symbol) in enumerate(
                    instance[entity_idx].rep, start=entity.first_index
                )
            } for entity_idx, entity in enumerate(self.data)
            if entity.defined_sequence()
        }

        entity_to_valid_subs = {
            entity_idx: set(entity.alphabet(include_gap=allow_gap))
            for entity_idx, entity in enumerate(self.data)
        }

        invalid_subs = [
            (i, subs) for (i, mutant) in enumerate(mutants) for subs in mutant if (
                (subs.entity not in entity_to_pos) or  # valid entity index
                (subs.pos not in entity_to_pos[subs.entity]) or  # valid position in entity
                (subs.ref != entity_to_pos[subs.entity][subs.pos]) or  # invalid reference symbol
                (subs.to not in entity_to_valid_subs[subs.entity])
            )
        ]

        invalid = len(invalid_subs) > 0

        if invalid and raise_invalid:
            raise ValueError(f"Invalid mutants: {invalid_subs}")

        return invalid, invalid_subs


class Protein(Entity):
    """
    Single protein chain entity
    """
    def __init__(
        self,
        id: str | None,  # noqa
        rep: str | None = None,
        first_index: int = 1,
        copies: int | None = None,
        sequences: Sequences | None = None,
        structures: StructureChainMap | None = None,
    ):
        """
        Create new protein entity

        Parameters
        ----------
        id
            Unique identifier of protein
        rep
            Sequence of protein (if None, auto-infer or leave open as needed for model).
            May contain any valid amino acid or the mask symbol.
        first_index
            Sequence index of first residue (1-based numbering)
        copies
            Number of copies of protein chain in system (None to leave unspecified/variable)
        sequences
            Sequence record (e.g. multiple sequence alignment of homologs) of the target
            sequence represented by this entity
        structures
            Structure chains representing this entity. Use dict with structure identifiers
            as keys to supply multiple different structures; use list to supply multiple copies
            of the chain within the structure (homooligomer)
        """
        # verify that protein sequence is valid if specified (including mask)
        if rep is not None:
            valid_seq, invalid_aa = valid_protein_sequence(
                rep, allow_mask=True, allow_gap=False, allow_ambiguous=True
            )

            if not valid_seq:
                raise ValueError(f"Invalid protein sequence: {invalid_aa}")

        super().__init__(
            type="protein",
            id=id,
            rep=rep,
            first_index=first_index,
            copies=copies,
            sequences=sequences,
            structures=structures,
        )

# mapping from entity index to positions in entity (e.g. for fixing positions)
EntityPosList = Mapping[int, Sequence[int]]
