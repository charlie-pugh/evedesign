"""
Biopolymer sequence functionality (protein sequences etc.)
"""

from typing import Literal, Tuple, List

from protdesign.constants import AA_TO_INDEX, MASK, GAP, INDEX_TO_AA
from protdesign.types import BioPolymer


class Sequence:
    """
    Single biopolymer sequence

    # TODO: add methods for sequence verification and transformation
    """
    def __init__(
        self,
        seq: str,
        seq_id: str | None = None,
        key: str | None = None,
        seq_type: BioPolymer = "protein",
    ):
        """
        Create new sequence object

        Parameters
        ----------
        seq
            Sequence (can contain lowercase characters and gaps)
        seq_id
            Identifier of sequence
        key
            Key for matching sequence to other resources (e.g. paired alignment)
        seq_type
            Type of biopolymer (protein, rna, dna, ...)
        """
        self.seq = seq
        self.seq_id = seq_id,
        self.key = key
        self.seq_type = seq_type


class Sequences:
    """
    Collection of one or more biopolymer sequences, can be aligned or unaligned

    # TODO: method to turn into different formats of alignments, and to dealign
    # TODO: make this class a list? probably not good for extra attributes
    #   can this be serialized?
    #   (running alignment probably out of scope...)
    """
    def __init__(self):
        # TODO: store if aligned and what type of alignment
        pass



def valid_protein_sequence(
    seq: str,
    allow_mask: bool = False,
    allow_gap: bool = False,
    allow_ambiguous: bool = False,
) -> Tuple[bool, List[Tuple[int, str]]]:
    """
    Check if a given sequence is a valid protein sequence

    Parameters
    ----------
    seq
        Protein seqeunce
    allow_mask
        Consider mask character as valid symbol (default: False)
    allow_gap
        Consider gap character as valid symbol (default: False)
    allow_ambiguous
        Consider ambiguous amino acids as valid symbol (default: False)

    Returns
    -------
    bool
        True if valid sequence, False otherwise
    str
        Invalid characters and their indices in sequence
    """
    invalid = [
        (i, aa) for i, aa in enumerate(seq) if not (
            aa in AA_TO_INDEX or
            (allow_mask and aa == MASK) or
            (allow_gap and aa == GAP)
        ) or (
            not allow_ambiguous and aa in AA_TO_INDEX and INDEX_TO_AA[AA_TO_INDEX[aa]] != "aa"
        )
    ]

    return len(invalid) == 0, invalid
