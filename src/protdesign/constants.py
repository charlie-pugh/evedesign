# Mapping according to HHblits convention
# https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L975

MASK = "*"
GAP = "-"

AA_TO_INDEX = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
}

INDEX_TO_AA = {
    idx: symbol for symbol, idx in AA_TO_INDEX.items() if symbol not in {"U", "B", "Z", "J", "O"}
}

VALID_AA_TO_INDEX = {
    symbol: idx for symbol, idx in  AA_TO_INDEX.items() if symbol not in {"U", "B", "Z", "J", "O", "X"}
}

VALID_AA = set(VALID_AA_TO_INDEX)
VALID_AA_SORTED = sorted(VALID_AA)

VALID_AA_OR_GAP = VALID_AA | {GAP}
VALID_AA_OR_GAP_SORTED = sorted(VALID_AA) + [GAP]