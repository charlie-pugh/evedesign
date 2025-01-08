from protdesign.constants import AA_TO_INDEX, INDEX_TO_AA, MASK, GAP
from typing import Tuple, List, Sequence, Any, TypeVar

T = TypeVar("T")


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


def ensure_sequence(x: T | Sequence[T]) -> Sequence[T]:
    if isinstance(x, Sequence):
        return x
    else:
        return [x]
