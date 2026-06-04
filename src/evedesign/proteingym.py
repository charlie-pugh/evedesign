from typing import Sequence

from biotite.structure import AtomArray

from evedesign import sequence
from evedesign.dataset import LabeledInstanceDataset
from evedesign.structure import Structure
from evedesign.system import EntityInstance, Protein, System, SystemInstance

from proteingym.base import Dataset, Subsets
from proteingym.base.sequence import SequenceType


def wildtype_or_none(dataset: Dataset) -> str | None:
    """
    Extract wildtype sequence from dataset if available to set Entity rep, which
    needs to be set to None otherwise

    Note the reference_sequence computed field on Dataset fails with StopIteration error
    """
    wt = [
        seq for seq in dataset.sequences if seq.type == SequenceType.WILD_TYPE
    ]

    if len(wt) > 0:
        return str(wt[0].value)
    else:
        return None


def msa_to_sequences(dataset: Dataset) -> sequence.Sequences | None:
    """
    Convert the first MSA in the dataset into an evedesign ``Sequences`` object,
    or return None if the dataset has no MSAs.
    """
    if len(dataset.msas) == 0:
        return None

    # this case is non-trivial scientifically yet highly relevant; discuss first before implementing...
    if dataset.msas[0].sequence_start is not None and dataset.msas[0].sequence_start != 1:
        raise NotImplementedError(
            "MSA region start different from 1"
        )

    # take first MSA by default for now
    first_msa = dataset.msas[0].value

    # TODO: length of weights and sequences not equal, presumaby invalid sequence
    #  filtered out - but no straightforward way to match them back together?
    # if len(dataset.msa_weights) == 0:
    #     weights = None
    # else:
    #    weights = dataset.msa_weights[0].value
    #
    # assert len(weights) == len(first_msa), "MSA and weights length does not match"

    return sequence.Sequences(
        seqs=[sequence.Sequence(seq=str(seq), id=None) for seq in first_msa],
        aligned=True,
        type="protein",
        # weights=weights,
        format="a3m", # everything will be a3m
    )


def update_structure(atom_array: AtomArray) -> Structure:
    """
    Wrap a ProteinGym structure (a biotite ``AtomArray``) into the evedesign
    ``Structure`` model -- assumes monomers
    """
    s = Structure(atom_array)
    assert len(s.chains()) == 1
    return s.get_chain(s.chains()[0])


def seqs_to_instances(sequences: Sequence[str]) -> list[SystemInstance]:
    """
    Transform standard string-based sequences into evedesign instances
    """
    return [
        SystemInstance([
            EntityInstance(rep=seq)
        ])
        for seq in sequences
    ]


def dataset_to_evedesign(
    subsets: Subsets,
    split: str,
    target: str,
) -> tuple[System, LabeledInstanceDataset]:
    """
    Map a ProteinGym dataset (subset) to evedesign representations.

    Params
    ----------
    subsets:
        Loaded ProteinGym Subsets object (ex. Subsets.from_path(path))
    split:
        Name of the split whose dataset should be converted ex. 'train'
    target:
        Name of the assay target to extract ex. 'DMS Score'

    Gives
    -------
    system:
        evedesign System with a single Protein. We default to a
        single-component protein system as there are no other cases in ProteinGym
        (yet), and assume structure numbering matches the rep sequence and
        first_index by convention
    data:
        LabeledInstanceDataset mapping each assay sequence (as a
        SystemInstance) to its target value
    """

    dataset = subsets[split].dataset

    # Default to single-component protein system as there are
    # no other cases in PG (yet...) We assume structure numbering
    # matches rep sequence and first_index by convention.

    system = System([
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

    # Build labeled instances ("X" and "y") from the dataset's assay records.
    valid_targets = [t.name for t in dataset.assay_targets]
    if target not in valid_targets:
        raise ValueError(
            f"Target '{target}' is not present in dataset assay targets, "
            f"valid options are: {', '.join(valid_targets)}"
        )

    df = dataset.to_df()
    sequences = df["sequence"].to_list()
    # missing target values come back as None (not NaN)
    values = df[target].to_list()

    data = LabeledInstanceDataset(
        # TODO: need to handle insertion/deletion datasets properly; sequences
        # are currently treated as full fixed-length substitution sequences
        # comment: these instances have lowercase/gaps, so I guess it will be
        # on each individual model to featurize accordingly
        instances=seqs_to_instances(sequences),
        labels={target: values},
    )

    return system, data
