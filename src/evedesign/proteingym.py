from typing import Sequence

import polars as pl
from biotite.structure import AtomArray

from evedesign import sequence
from evedesign.dataset import LabeledInstanceDataset, LabeledInstanceTrainTestDataset
from evedesign.structure import Structure
from evedesign.system import EntityInstance, Protein, System, SystemInstance

from proteingym.base import Dataset, Subsets
from proteingym.base.sequence import SequenceType


def wildtype_or_none(dataset: Dataset) -> str | None:
    """Extract the wildtype sequence from a dataset to set rep.

    Parameters
    ----------
    dataset : Dataset
        ProteinGym dataset to extract the wildtype sequence from.

    Returns
    -------
    str or None
        The wildtype sequence if available, otherwise None.
    """
    wt = [
        seq for seq in dataset.sequences if seq.type == SequenceType.WILD_TYPE
    ]

    if len(wt) > 0:
        return str(wt[0].value)
    else:
        return None


def msa_to_sequences(dataset: Dataset) -> sequence.Sequences | None:
    """Convert the first MSA in a dataset into an evedesign Sequences object.

    Parameters
    ----------
    dataset : Dataset
        ProteinGym dataset to extract the MSA from.

    Returns
    -------
    sequence.Sequences or None
        The first MSA as an evedesign Sequences object, or None if the
        dataset has no MSAs.

    Raises
    ------
    NotImplementedError
        If the MSA region start is different from 1.
    """
    if len(dataset.msas) == 0:
        return None

    # this case is non-trivial scientifically yet highly relevant; discuss first before implementing...
    #if dataset.msas[0].sequence_start is not None and dataset.msas[0].sequence_start != 1:
    #    raise NotImplementedError(
    #        "MSA region start different from 1"
    #    )

    # take first MSA by default for now
    first_msa = dataset.msas[0].value

    # TODO: length of weights and sequences not equal, presumaby invalid sequence
    #  filtered out - but no straightforward way to match them back together?
    if len(dataset.msa_weights) == 0:
        weights = None
    else:
        weights = dataset.msa_weights[0].value
        assert len(weights) == len(first_msa), "MSA and weights length does not match"

    return sequence.Sequences(
        seqs=[sequence.Sequence(seq=str(seq), id=None) for seq in first_msa],
        aligned=True,
        type="protein",
        weights=weights,
        format="a3m", # everything will be a3m
    )


def update_structure(atom_array: AtomArray) -> Structure:
    """Wrap a ProteinGym structure into the evedesign Structure model.

    Assumes monomers.

    Parameters
    ----------
    atom_array : AtomArray
        ProteinGym structure as a biotite AtomArray.

    Returns
    -------
    Structure
        The wrapped evedesign Structure.
    """
    s = Structure(atom_array)
    assert len(s.chains()) == 1
    return s


def seqs_to_instances(sequences: Sequence[str]) -> list[SystemInstance]:
    """Transform standard string-based sequences into evedesign instances.

    Parameters
    ----------
    sequences : Sequence[str]
        String-based sequences to transform.

    Returns
    -------
    list[SystemInstance]
        The sequences wrapped as evedesign instances.
    """
    return [
        SystemInstance([
            EntityInstance(rep=seq)
        ])
        for seq in sequences
    ]


def system_from_dataset(dataset: Dataset) -> System:
    """Build an evedesign System from a ProteinGym Dataset.

    We default to a single-component protein system as there are no other cases
    in ProteinGym (yet...) and assume structure numbering matches the rep
    sequence and first_index by convention.

    Parameters
    ----------
    dataset : Dataset
        ProteinGym dataset to build the System from.

    Returns
    -------
    System
        An evedesign System with a single Protein.
    """
    return System([
        Protein(

            first_index=1, # this is the case for 90% of the database, would
            # be good to have as an inferred param eventually - we aren't using
            # mutation string labels for anything, so this won't be an issue

            # extract WT sequence, otherwise set to None - but we shouldn't ever
            # have that case with ProteinGym? - agreed
            rep=wildtype_or_none(dataset),

            # extract first MSA; we need to discuss the case where MSA target
            # region != full sequence
            sequences=msa_to_sequences(dataset),

            # Wrap each dataset structure (a biotite AtomArray) directly into the
            # evedesign Structure model
            structures={
                struc.name: update_structure(struc.value)
                for struc in dataset.structures
            },
        )
    ])


def labeled_dataset_from_df(df: pl.DataFrame, target: str) -> LabeledInstanceDataset:
    """Build a LabeledInstanceDataset from a ProteinGym dataframe slice.

    TODO: need to handle insertion/deletion datasets properly; sequences are
    currently treated as full fixed-length substitution sequences.

    Comment:
    these instances have lowercase/gaps, so I guess it will be on each
    individual model to featurize accordingly.

    Parameters
    ----------
    df : pl.DataFrame
        ProteinGym dataframe slice containing a sequence column and the
        target column.
    target : str
        Name of the target column to extract labels from.

    Returns
    -------
    LabeledInstanceDataset
        Dataset mapping each sequence instance to its target value.
    """
    sequences = df["sequence"].to_list()
    # missing target values come back as None
    values = df[target].to_list()

    return LabeledInstanceDataset(
        instances=seqs_to_instances(sequences),
        labels={target: values},
    )


def dataset_to_evedesign(
    subsets: Subsets,
    split: str,
    target: str,
    test_fold: int,
) -> tuple[System, LabeledInstanceTrainTestDataset]:
    """Map a ProteinGym dataset (subset) to evedesign representations.

    Follows the train/test conventions of the benchmark training entrypoint.

    Parameters
    ----------
    subsets : Subsets
        Loaded ProteinGym Subsets object (ex. Subsets.from_path(path)).
    split : str
        Name of the split column in the dms csv.
    target : str
        Name of the assay target to extract, ex. 'DMS Score'.
    test_fold : int
        Index of the slice to use as the test set. The remaining folds
        form the training set. If None, the whole dataset is used as the
        training set and the test set is None.

    Returns
    -------
    system : System
        evedesign System with a single Protein. We default to a
        single-component protein system as there are no other cases in ProteinGym
        (yet), and assume structure numbering matches the rep sequence and
        first_index by convention.
    data : LabeledInstanceTrainTestDataset
        Dataset mapping each assay sequence (as a SystemInstance) to its target
        value, split into train/test sets according to test_fold.

    Raises
    ------
    ValueError
        If target is not present in the dataset assay targets, or if
        test_fold is out of range for the given split.
    """

    subset_split = subsets[split]
    dataset = subset_split.dataset

    system = system_from_dataset(dataset)

    valid_targets = [t.name for t in dataset.assay_targets]
    if target not in valid_targets:
        raise ValueError(
            f"Target '{target}' is not present in dataset assay targets, "
            f"valid options are: {', '.join(valid_targets)}"
        )

    n_folds = len(subset_split.slices)
    if not 0 <= test_fold < n_folds:
        raise ValueError(
            f"test_fold {test_fold} is out of range for split '{split}' "
            f"with {n_folds} folds"
        )

    # test set is the requested fold; training set is everything else
    test_df = dataset[subset_split.slices[test_fold]].to_df()
    test_set = labeled_dataset_from_df(test_df, target)

    train_dfs = [
        dataset[subset_split.slices[i]].to_df()
        for i in range(n_folds)
        if i != test_fold
    ]

    train_df = pl.concat(train_dfs) if train_dfs else test_df
    training_set = labeled_dataset_from_df(train_df, target)

    data = LabeledInstanceTrainTestDataset(
        training_set=training_set,
        test_set=test_set,
    )

    return system, data
