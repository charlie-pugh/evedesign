"""
Tools for clustering designs on sequence/structure/embedding level
"""

def cluster_sequences_mmseqs(
    sequences: list[str],
    target_num_clusters: int,
    priorities: list[float] | None = None,
    mmseqs_path: str = "mmseqs"
) -> tuple[list[int], list[int]]:
    """
    Reduce sequences down to a specified number of clusters with MMseqs

    Parameters
    ----------
    sequences
        Corresponds to what is supplied to shell script as input.fasta
    target_num_clusters
        Target number of clusters
    priorities
        Corresponds to priority.tsv; in same order as sequence list
    mmseqs_path
        Path to mmseqs binary (optional, defaults to assuming mmseqs is on $PATH)

    Returns
    -------
    Tuple containing
    (i) indices of picked sequences (list[int]) in input sequence list; each list element represents a cluster
    (ii) corresponding cluster indices (list[int]) for all input sequences, same length as input sequence list
    """
    raise NotImplementedError()