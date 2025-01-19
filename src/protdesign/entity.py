"""
Specification of components of molecular design system (proteins, nucleic acids, ligands, etc.)
"""
from collections import UserList
from collections.abc import Sequence
from typing import Mapping, NamedTuple
from protdesign.sequence import valid_protein_sequence, Sequences
from protdesign.structure import StructureChainMap
from protdesign.types import EntityType  #, Metadata
from protdesign.utils import ensure_sequence, shorten


class Entity:
    def __init__(
        self,
        type: EntityType,
        rep: str | None = None,
        id: str | None = None,
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


class EntityInstance:
    """
    Instantiation of a single entity in a system
    """
    def __init__(
        self,
        rep: str | None = None,
        structure_models: StructureChainMap | None = None,
    ):
        """
        Create new instantiation of an entity in a sequence

        Parameters
        ----------
        rep
            Representation (e.g. sequence) of entity. Set to None if no
            representation is yet available (e.g. just structural backbone but no sequence)
        structure_models
            Structural models associated with each of the entities in the system.
            Set to None if no structural models are available.
        """
        self.rep = rep
        self.structure_models = structure_models

    def __repr__(self):
        if self.structure_models is not None:
            structure_info = len(self.structure_models)
        else:
            structure_info = self.structure_models

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
        # metadata: Metadata | None = None,
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
        # self.metadata = metadata

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
        # TODO: implement equality of molecular systems: same length, and equality of all attributes per entity
        raise NotImplementedError()

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
                    valid = valid and len(entity.rep) == len(entity_instance.rep)

                if validate_reps:
                    # allow gaps for deletion modeling, and mask for leaving parts of representation unspecified
                    is_valid_seq = valid_protein_sequence(
                        entity_instance.rep, allow_mask=True, allow_gap=True, allow_ambiguous=False
                    )

                    valid = valid and is_valid_seq

        if not valid and raise_invalid:
            raise ValueError("Provided instance is not valid for biomolecular system")

        return valid



class Protein(Entity):
    """
    Single protein chain entity
    """
    def __init__(
        self,
        id: str | None,
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

Mutation = NamedTuple(
    "Mutation", [("entity", int), ("pos", int), ("ref", str), ("to", str)]
)

Mutant = Sequence[Mutation]