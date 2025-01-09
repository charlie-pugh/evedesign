"""
Specification of components of molecular design system (proteins, nucleic acids, ligands, etc.)
"""
from typing import Literal, List
from protdesign.utils import valid_protein_sequence

PROTEIN = "protein"
EntityType = Literal[PROTEIN]


class Entity:
    def __init__(
        self,
        entity_type: EntityType,
        repr: str | None = None,
        id: str | None = None,
        copies: int | None = None,
        first_index: int | None = None,
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
        entity_type
            Type of entity (protein, nucleotide, ligand, ...)
        id
            Unique identifier of entity
        repr
            Representation of entity (sequence, atom name, etc.)
        first_index
            Sequence index of first residue; must be specified
            for polymer types (protein, nucleotide, ...)
        copies
            Number of entity copies in molecular system. Set to None
            to leave variable.
        """
        self.entity_type = entity_type
        self.repr = repr
        self.id = id
        self.copies = copies

        if entity_type == PROTEIN and first_index is None:
            raise ValueError(
                f"first_index must be specified for entity_type {entity_type}"
            )
        self.first_index = first_index


EntityList = List[Entity]
EntityOrEntityList = Entity | EntityList


class Protein(Entity):
    """
    Single protein chain entity
    """
    def __init__(
        self,
        id: str | None,
        seq: str | None = None,
        first_index: int = 1,
        copies: int | None = None,
    ):
        """
        Create new protein entity

        Parameters
        ----------
        id
            Unique identifier of protein
        seq
            Sequence of protein (if None, auto-infer or leave open as needed for model).
            May contain any valid amino acid or the mask symbol.
        first_index
            Sequence index of first residue (1-based numbering)
        copies
            Number of copies of protein chain in system (None to leave unspecified/variable)
        """
        # verify that protein sequence is valid if specified (including mask)
        if seq is not None:
            valid_seq, invalid_aa = valid_protein_sequence(
                seq, allow_mask=True, allow_gap=False, allow_ambiguous=True
            )

            if not valid_seq:
                raise ValueError(f"Invalid protein sequence: {invalid_aa}")

        super().__init__(
            entity_type=PROTEIN,
            id=id,
            repr=seq,
            first_index=first_index,
            copies=copies,
        )
