"""
Wrapper classes around the EVcouplings/EVmutation Potts model

This wrapper deliberately keeps only the two stages of the evcouplings pipeline that
are relevant for modelling a single, already-aligned protein family inside evedesign:

MSA construction is out of scope: the provided System is expected to already carry a MSA

Two concrete engines are provided as subclasses of the abstract EVcouplings base:

EVcouplingsMeanField runs mean-field DCA and has no external dependency

EVcouplingsPLM runs the pseudo-likelihood solver via the plmc binary, a program 
that is not on PyPI (see https://github.com/debbiemarkslab/plmc). Use
EVcouplingsMeanField to avoid the external dependency
"""
import tempfile
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Self, Sequence

import numpy as np

from evedesign.model import BaseModel, Scorer
from evedesign.system import System, SystemInstance
from evedesign.sequence import REMOVE_INSERTIONS_TRANSLATION
from evedesign.constants import GAP, MASK
from evedesign.types import StatusCallback
from evedesign.utils import status_done, status_start

try:
    from evcouplings.align.alignment import Alignment, ALPHABET_PROTEIN
    from evcouplings.couplings.mean_field import MeanFieldDCA
    from evcouplings.couplings.model import CouplingsModel
    from evcouplings.couplings.tools import run_plmc
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class EVcouplings(BaseModel, Scorer):
    """
    Abstract base wrapper around EVcouplings/EVmutation

    Holds all logic shared between the inference engines. Subclasses implement 
    _fit() for a specific engine. Instantiate EVcouplingsPLM or EVcouplingsMeanField 
    directly
    """
    available = IMPORT_AVAILABLE
    name: str = "EVcouplings"
    citations: list[str] = [
        # EVmutation
        "10.1038/nbt.3769",
        # EVcouplings
        "10.1093/bioinformatics/bty862",
        # plmc ? idk if there is a paper
    ]

    # core properties
    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = True
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = ["sequences"]
    optional_entity_attributes: list[str] | None = None

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        theta: float = 0.8,
    ):
        """
        Initialise the shared EVcouplings state.

        Parameters
        ----------
        max_gap_fraction
            Alignment columns whose (unweighted) gap frequency strictly exceeds this
            threshold are excluded from the fitted model and from positions()
        theta
            Sequence reweighting identity threshold; sequences with pairwise identity
            >= theta are clustered and down-weighted. Used by both engines.
        """
        if not self.available:
            raise ImportError(
                "evcouplings package could not be imported. Install w/"
                "pip install evedesign[evcouplings]"
            )

        if not 0.0 < max_gap_fraction <= 1.0:
            raise ValueError("max_gap_fraction must be in (0, 1]")

        self.max_gap_fraction = max_gap_fraction
        self.theta = theta

        self._system: System | None = None

        # parsed Potts model produced by build(); pickles directly with the wrapper
        self.model: CouplingsModel | None = None
        # residue indices of the positions w/ enough coverage
        self._index_list: np.ndarray | None = None


    @property
    def system(self) -> System | None:
        return self._system

    @property
    def ready(self) -> bool:
        return self._system is not None and self.model is not None

    @classmethod
    def can_model(cls, system: System, data: Any = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support a data parameter (must be None)"

        # will eventually include evcomplex, would have been kind of a lot of work
        # and not sure if evedesign can handle paired MSA+instances without modification
        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle a single-component protein system"

        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have a defined rep sequence"

        if target.sequences is None or len(target.sequences.seqs) == 0:
            return False, "Must provide an MSA (entity.sequences) for model inference"

        if not target.sequences.aligned:
            return False, "Provided sequences must be aligned"

        return True, ""


    @staticmethod
    def _match_states(seq: str) -> str:
        """Strip insertion states from an aligned sequence, leaving only match-state columns"""
        return seq.translate(REMOVE_INSERTIONS_TRANSLATION)

    def _build_alignment(self, target) -> tuple["Alignment", np.ndarray]:
        """
        Build an evcouplings Alignment from the system MSA and lower-case
        the high-gap columns so they are excluded from the fit

        Returns the (possibly modified) Alignment and the boolean mask of excluded columns
        """
        seqs = target.sequences.seqs
        target_rep = "".join(target.rep)
        length = len(target_rep)

        match_seqs = [self._match_states(s.seq) for s in seqs]

        bad = [i for i, m in enumerate(match_seqs) if len(m) != length]
        if bad:
            raise ValueError(
                f"MSA match-state length does not match target length ({length}) "
                f"for {len(bad)} sequence(s), e.g. sequence index {bad[0]}"
            )

        if match_seqs[0] != target_rep:
            raise ValueError(
                "First MSA sequence (match states) must equal the target/focus sequence. "
                "EVcouplings requires the target sequence as the first alignment record"
            )

        first_index = target.first_index
        focus_id = str(seqs[0].id_).split()[0]
        ids = (
            [f"{focus_id}/{first_index}-{first_index + length - 1}"]
            + [str(s.id_) for s in seqs[1:]]
        )

        matrix = np.array([list(m) for m in match_seqs])
        alignment = Alignment(matrix, sequence_ids=ids, alphabet=ALPHABET_PROTEIN)

        # unweighted per-column gap frequency, exclude columns above the threshold
        gap_freq = alignment.count(GAP, axis="pos", normalize=True)
        excluded = gap_freq > self.max_gap_fraction

        if excluded.all():
            raise ValueError(
                "All positions exceed max_gap_fraction, what are you aligning...?"
            )

        if excluded.any():
            # lower-casing turns these into fake insert columns, which get stripped later
            alignment = alignment.lowercase_columns(np.where(excluded)[0])

        return alignment, excluded


    @abstractmethod
    def _fit(
        self,
        alignment: "Alignment",
        focus_id: str,
        num_model_positions: int,
    ) -> "CouplingsModel":
        """
        Fit the engine-specific Potts model and return the parsed CouplingsModel.
        """
        raise NotImplementedError


    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,
    ) -> Self:
        self.can_model_or_raise(system, data)

        status_start(status_callback, "Fitting EVcouplings model")

        self._system = system
        target = system[0]

        # reset any previous fit
        self.model = None
        self._index_list = None

        alignment, _ = self._build_alignment(target)

        # number of modelled positions = match columns
        focus_id = str(target.sequences.seqs[0].id_).split()[0]
        num_model_positions = int(
            np.array([c.isupper() and c != GAP for c in alignment[0]]).sum()
        )

        self.model = self._fit(alignment, focus_id, num_model_positions)
        self._index_list = np.asarray(self.model.index_list, dtype=int)

        status_done(status_callback, "EVcouplings model finished fitting")

        return self


    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        """
        Return the modelled positions (in entity 0). Positions excluded due to high gap
        content are not part of the fitted model and are therefore not returned
        """
        self.ready_or_raise()
        return [(0, int(pos)) for pos in self._index_list]


    # elected to score full sequence to avoid dealing with indexing
    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,
    ) -> np.ndarray:
        """
        Score full sequences by their statistical energy

        Only the modelled (non-excluded) positions contribute, excluded positions are
        ignored
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if len(instances) == 0:
            return np.empty((0,), dtype=float)

        status_start(status_callback, "Scoring sequences")

        first_index = self.system[0].first_index
        # map modelled residue numbers to 0-based positions in the (fixed-length) rep
        col_pos = self._index_list - first_index

        subseqs = []
        for instance in instances:
            rep = instance[0].rep
            subseq = "".join(str(c) for c in np.asarray(rep)[col_pos])
            if MASK in subseq:
                raise ValueError(
                    "Cannot score sequence containing mask symbol at a modelled position"
                )
            subseqs.append(subseq)

        # column 0 is the total Hamiltonian (J_ij + h_i sub-sums in columns 1, 2)
        hamiltonians = self.model.hamiltonians(subseqs)[:, 0]

        status_done(status_callback, "Scoring complete")

        return np.asarray(hamiltonians, dtype=float)


class EVcouplingsMeanField(EVcouplings):
    """
    EVcouplings model fitted with mean-field DCA
    """
    name: str = "EVcouplingsMeanField"

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        theta: float = 0.8,
        pseudo_count: float = 0.5,
    ):
        """
        Instantiate a mean-field DCA EVcouplings model.

        Parameters
        ----------
        max_gap_fraction
            Alignment columns whose (unweighted) gap frequency strictly exceeds this
            threshold are excluded from the fitted model and from positions()
        theta
            Sequence reweighting identity threshold; sequences with pairwise identity
            >= theta are clustered and down-weighted
        pseudo_count
            Pseudo-count for frequency regularization
        """
        super().__init__(max_gap_fraction=max_gap_fraction, theta=theta)
        self.pseudo_count = pseudo_count

    def _fit(
        self,
        alignment: "Alignment",
        focus_id: str,
        num_model_positions: int,
    ) -> "CouplingsModel":
        return MeanFieldDCA(alignment).fit(
            theta=self.theta, pseudo_count=self.pseudo_count
        )


class EVcouplingsPLM(EVcouplings):
    """
    EVcouplings model fitted with the plmc pseudo-likelihood solver

    Requires the external plmc binary (https://github.com/debbiemarkslab/plmc)
    """
    name: str = "EVcouplingsPLM"
    # plmc can be compiled w/ multi-core fitting
    supports_cpu_parallel: bool = True

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        theta: float = 0.8,
        lambda_h: float = 0.01,
        lambda_J: float = 0.01,
        lambda_J_times_Lq: bool = True,
        lambda_group: float | None = None,
        scale_clusters: float | None = None,
        iterations: int | None = None,
        ignore_gaps: bool = False,
        plmc_binary: str | PathLike = "plmc",
        cpu: int | Literal["max"] | None = None,
    ):
        """
        Instantiate a plmc (pseudo-likelihood) EVcouplings model.

        Parameters
        ----------
        max_gap_fraction
            Alignment columns whose (unweighted) gap frequency strictly exceeds this
            threshold are excluded from the fitted model and from positions()
        theta
            Sequence reweighting identity threshold; sequences with pairwise identity
            >= theta are clustered and down-weighted
        lambda_h
            L2 regularisation strength on fields h_i
        lambda_J
            L2 regularisation strength on couplings J_ij. If lambda_J_times_Lq is True,
            this base value is scaled by (num_symbols - 1)*(L - 1) (as in standard
            evcouplings)
        lambda_J_times_Lq
            Scale lambda_J by the number of states and modelled positions
        lambda_group
            Group L1 regularisation strength on couplings (None = plmc default)
        scale_clusters
            Scale weights of sequence clusters by this value (None = plmc default)
        iterations
            Maximum optimization iterations (None = plmc default)
        ignore_gaps
            If True, exclude gaps from parameter inference. Note that this also implies
            gaps cannot be scored--the default (False) keeps gap as a model symbol
        plmc_binary
            Path to / name of the plmc binary
        cpu
            Number of cores for plmc (requires OpenMP-compiled plmc) - or "max"
        """
        super().__init__(max_gap_fraction=max_gap_fraction, theta=theta)
        # most of these params are just defaults inherited from EVcouplings, leaving here
        # in case users want to mess with them for some reason
        self.lambda_h = lambda_h
        self.lambda_J = lambda_J
        self.lambda_J_times_Lq = lambda_J_times_Lq
        self.lambda_group = lambda_group
        self.scale_clusters = scale_clusters
        self.iterations = iterations
        self.ignore_gaps = ignore_gaps
        self.plmc_binary = plmc_binary
        self.cpu = cpu

    def _fit(
        self,
        alignment: "Alignment",
        focus_id: str,
        num_model_positions: int,
    ) -> "CouplingsModel":
        # scale lambda_J as in the standard couplings protocol
        lambda_J = self.lambda_J
        if self.lambda_J_times_Lq:
            num_symbols = len(ALPHABET_PROTEIN) - (1 if self.ignore_gaps else 0)
            lambda_J = lambda_J * (num_symbols - 1) * (num_model_positions - 1)

        # writing temp files will be unavoidable if we want to limit
        # the amount of code we copy from couplings
        # although writing the alignment to file is super annoying
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            aln_file = tmp / "alignment.fasta"
            couplings_file = tmp / "couplings.txt"
            param_file = tmp / "model.params"

            with open(aln_file, "w") as f:
                alignment.write(f, format="fasta")

            run_plmc(
                str(aln_file),
                str(couplings_file),
                param_file=str(param_file),
                focus_seq=focus_id,
                # None -> plmc default protein alphabet (gap included)
                alphabet=None,
                theta=self.theta,
                scale=self.scale_clusters,
                ignore_gaps=self.ignore_gaps,
                iterations=self.iterations,
                lambda_h=self.lambda_h,
                lambda_J=lambda_J,
                lambda_g=self.lambda_group,
                cpu=self.cpu,
                binary=str(self.plmc_binary),
            )

            return CouplingsModel(str(param_file), file_format="plmc_v2")
