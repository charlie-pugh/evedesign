"""
Specification of components of molecular design system (proteins, nucleic acids, ligands, etc.)
"""
from typing import List
from protdesign.sequence import valid_protein_sequence, Sequences
from protdesign.structure import StructureChainMap
from protdesign.types import EntityType, Metadata


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


class SystemInstance:
    """
    Result designing the representation (structure/sequence) of the entity or entities
    in a system
    """
    def __init__(
        self,
        reps: List[str | None],
        structures: List[StructureChainMap | None ] = None,
        score: float | None = None,
        confidence: float | None = None,
        # metadata: Metadata | None = None,
    ):
        """
        Create new entity system instance

        Parameters
        ----------
        reps
            List of representations for different entities in system.
            If only structural model but no sequence available for any entity, set
            element to None
        structures
            List of structural models associated with each of the entities in the system.
            For any entity, set list element to None if no structural models are available.
        score
            Score describing quality/likelihood of the designed system instance
            (higher is better, ideally in logits)
        confidence
            Reliability of model score from 0 (lowest confidence) to 1 (highest confidence)
        """
        self.reps = reps
        self.structures = structures
        self.score = score
        self.confidence = confidence
        # self.metadata = metadata


EntityList = List[Entity]
EntityOrEntityList = Entity | EntityList


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
