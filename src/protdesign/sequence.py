"""
Biopolymer sequence functionality (protein sequences etc.)
"""
from typing import List, Literal, Self, TextIO, Tuple
from collections import abc
from protdesign.constants import AA_TO_INDEX, MASK, GAP, INDEX_TO_AA
from protdesign.types import BioPolymer
from protdesign.utils import shorten


class Sequence:
    """
    Single biopolymer sequence (may include gaps and inserts in lowercase)

    # TODO: add methods for sequence verification and transformation
    # TODO: add attributes for description and any other relevant metadata
    """
    def __init__(
        self,
        seq: str,
        id: str | None = None,
        key: str | None = None,
        type: BioPolymer = "protein",
    ):
        """
        Create new sequence object

        Parameters
        ----------
        seq
            Sequence (can contain lowercase characters and gaps)
        id
            Identifier of sequence
        key
            Key for matching sequence to other resources (e.g. paired alignment)
        type
            Type of biopolymer sequence (protein, rna, dna, ...)
        """
        self.seq = seq
        self.id_ = id
        self.key = key
        self.type_ = type

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.id_} key={self.key} type={self.type_} seq={shorten(self.seq)})"
        )


class Sequences:
    """
    Collection of one or more biopolymer sequences, can be aligned or unaligned

    This class only intends to be a thin wrapper around different alignment formats
    to connect input sequences to the different types of formats expected by individual methods,
    rather than a full-fledged class for computations on sequence alignments
    """
    def __init__(
        self,
        seqs: abc.Sequence[Sequence],
        aligned: bool = False,
        type: BioPolymer = "protein",
        weights: List[float] | None = None,
        format: Literal["a3m", "a2m", "fasta"] | None = None,
    ):

        self.seqs = seqs
        self.aligned = aligned
        self.type_ = type
        self.weights = weights
        self.format_ = format
        # TODO: check alignment integrity and/or autodetect properties/format

    @classmethod
    def from_file(cls, f: TextIO):
         # TODO: parameter for different format types
         # TODO: callback param for header parsing
        raise NotImplementedError(
            "Loading from file not yet implemented"
        )

    def dealign(self) -> Self:
        # remove gaps from sequences and return new
        raise NotImplementedError(
            "Sequence dealigning not yet implemented"
        )

    def to_a3m(self) -> Self:
        # return sequences in a3m format
        if self.format_ == "a3m":
            return self
        else:
            raise NotImplementedError(
                "Conversion to a3m format not yet implemented"
            )

    def to_a2m(self) -> Self:
        # return sequences in a2m format
        # TODO: add parameter to specify strategy how to deal with inserts (drop or fully expand sequences)
        #  cf. https://github.com/debbiemarkslab/EVcouplings/blob/75bfc9677fc9412ddb7089a9f26c7a01f65bfa12/evcouplings/align/alignment.py#L236
        if self.format_ == "a2m":
            return self
        else:
            raise NotImplementedError(
                "Conversion into a2m format not yet implemented"
            )

    def to_fasta(self) -> Self:
        if self.format_ == "fasta":
            return self
        else:
            raise NotImplementedError(
                "Conversion into fasta format not yet implemented"
            )

def valid_sequence(
    seq: str,
    alphabet: list[str],
    allow_mask: bool = False,
) -> Tuple[bool, List[Tuple[int, str]]]:
    """
    Check if a given sequence is valid according to some alphabet

    Parameters
    ----------
    seq
        Sequence to validate
    alphabet
        Valid symbols (may contain GAP and insert symbols)
    allow_mask
        If true, allow masked positions in the sequence

    Returns
    -------
    bool
        True if valid sequence, False otherwise
    list[tuple[int, str]]
        Invalid characters and their zero-based indices in sequence
    """
    alphabet = set(alphabet)

    invalid = [
        (i, symbol) for i, symbol in enumerate(seq) if not (
            symbol in alphabet or
            (allow_mask and symbol == MASK)
        )
    ]

    return len(invalid) == 0, invalid


# TODO: following is legacy function superseded by valid_sequence(), remove eventually
# def valid_protein_sequence(
#     seq: str,
#     allow_mask: bool = False,
#     allow_gap: bool = False,
#     allow_ambiguous: bool = False,
# ) -> Tuple[bool, List[Tuple[int, str]]]:
#     """
#     Check if a given sequence is a valid protein sequence
#
#     Parameters
#     ----------
#     seq
#         Protein seqeunce
#     allow_mask
#         Consider mask character as valid symbol (default: False)
#     allow_gap
#         Consider gap character as valid symbol (default: False)
#     allow_ambiguous
#         Consider ambiguous amino acids as valid symbol (default: False)
#
#     Returns
#     -------
#     bool
#         True if valid sequence, False otherwise
#     str
#         Invalid characters and their indices in sequence
#     """
#     invalid = [
#         (i, aa) for i, aa in enumerate(seq) if not (
#             aa in AA_TO_INDEX or
#             (allow_mask and aa == MASK) or
#             (allow_gap and aa == GAP)
#         ) or (
#             not allow_ambiguous and aa in AA_TO_INDEX and INDEX_TO_AA[AA_TO_INDEX[aa]] != aa
#         )
#     ]
#
#     return len(invalid) == 0, invalid


def read_fasta(f: TextIO):
    """
    Generator function to read a FASTA-format file
    (includes aligned FASTA, A2M, A3M formats)

    Parameters
    ----------
    f : file-like object
        FASTA alignment file

    Returns
    -------
    generator of (str, str) tuples
        Returns tuples of (sequence ID, sequence)
    """
    current_sequence = ""
    current_id = None

    for line in f:
        # Start reading new entry. If we already have
        # seen an entry before, return it first.
        if line.startswith(">"):
            if current_id is not None:
                yield current_id, current_sequence

            current_id = line.rstrip()[1:]
            current_sequence = ""

        elif not line.startswith(";"):
            current_sequence += line.rstrip()

    # Also do not forget last entry in file
    yield current_id, current_sequence
