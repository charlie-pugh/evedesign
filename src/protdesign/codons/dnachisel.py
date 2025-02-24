"""
Codon optimization with the DNA Chisel package
"""
import multiprocess as mp
from typing import Sequence, Literal

from dnachisel import AvoidPattern

try:
    import dnachisel as dc  # noqa
    from dnachisel.builtin_specifications.codon_optimization.BaseCodonOptimizationClass import (
        BaseCodonOptimizationClass  # noqa
    )
    from Bio.Restriction.Restriction_Dictionary import rest_dict  # noqa
    from Bio.Data.CodonTable import unambiguous_dna_by_name  # noqa
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False

from protdesign.entity import System, SystemInstance

OPTIMIZATION_METHODS = [
    "use_best_codon",
    "match_codon_usage"
]

CodonUsageTable = dict[str, [dict[str, float]]]

class DNAChiselCodonOptimizer:
    available = IMPORT_AVAILABLE

    def __init__(
        self,
        method: Literal["use_best_codon", "match_codon_usage"],
        codon_usage_table: str | CodonUsageTable,
        avoid_sites: list[str] | None,
        gc_min: float | None = 0.4,
        gc_max: float | None = 0.6,
        gc_window: int | None = None,
        genetic_code: str = "Standard",
        extra_constraints: Sequence[dc.Specification] | None = (
            dc.AvoidHairpins(),
            dc.AvoidPattern(dc.HomopolymerPattern("A", 5)),
            dc.AvoidPattern(dc.HomopolymerPattern("C", 5)),
            dc.AvoidPattern(dc.HomopolymerPattern("G", 5)),
            dc.AvoidPattern(dc.HomopolymerPattern("T", 5)),
            dc.AvoidPattern(dc.RepeatedKmerPattern(2, 5)),
        ),
        cpu: int = 1,
    ):
        """
        Create new codon optimizer based on DNA Chisel

        Parameters
        ----------
        method
            Optimize codon usage by maximizing codon adaption index (use_best_codon), or by matching
             match codon frequencies in target species (match_codon_usage)
        codon_usage_table
            Codon usage table for optimization. Can be any species valid for the python_codon_tables package
             (str; e.g. h_sapiens or e_coli), a taxonomy identifier (str of numeric code, will be downloaded
            from web), or an explicit codon usage table dictionary (CodonUsageTable)
        avoid_sites
            List of restriction enzyme sites to avoid during optimization (e.g. "BsaI"). For all valid options,
            see Bio.Restriction.Restriction_Dictionary.rest_dict
        gc_min
            Minimum GC content to enforce in optimized nucleotide sequence
        gc_max
            Maximum GC content to enforce in optimized nucleotide sequence
        gc_window
            If specified, compute GC content in a local window; otherwise will compute over entire nucleotide sequence
        genetic_code
            Genetic code to ensure nucleotide sequence translates into specified amino acid sequences
            (note this is redundant to codon_usage_table but internally needed by dnachisel)
        extra_constraints
            Extra dnachisel specifications to use during optimization
        cpu
            If cpu > 1, parallelize optimization over different instances with specified number of processes.
        """
        if not self.available:
            raise ValueError(
                "dnachisel or biopython package could not be imported. Are they already installed?"
            )

        self.genetic_code = genetic_code

        if method not in OPTIMIZATION_METHODS:
            raise ValueError(
                f"Invalid optimization method, valid options are {OPTIMIZATION_METHODS} "
            )

        self.method = method

        # verify we have a valid genetic code specified
        if genetic_code not in dc.biotools.CODON_TABLE_NAMES:
            raise ValueError(
                f"Invalid codon table, valid options are {dc.biotools.CODON_TABLE_NAMES}"
            )

        # retrieve explicit codon table as dictionary right away so we can verify against genetic code
        if isinstance(codon_usage_table, str):
            self.codon_table = BaseCodonOptimizationClass.get_codons_table(
                species=codon_usage_table, codon_usage_table=None
            )
        elif isinstance(codon_usage_table, dict):
            self.codon_table = codon_usage_table
        else:
            raise ValueError("Invalid codon_table argument")

        # verify that genetic code matches codon table
        for codon, aa in unambiguous_dna_by_name[self.genetic_code].forward_table.items():
            if codon not in self.codon_table[aa]:
                raise ValueError(
                    f"Mismatch between codon_usage_table and genetic_code:" +
                    f"aa: {aa} codon: {codon} options: {self.codon_table[aa]}"
                )

        # extra specifications to be added to optimization problem
        if extra_constraints is not None:
            self.specifications = list(extra_constraints)
        else:
            self.specifications = []

        if (gc_min is None and gc_max is not None) or (gc_min is not None and gc_max is None):
            raise ValueError(
                "gc_min and gc_max need to be both specified or None"
            )

        if gc_min is not None and gc_max is not None:
            if not 0 <= gc_min < gc_max <= 1:
                raise ValueError(
                    "GC content specification must be 0 <= gc_min < gc_max <= 1"
                )

            self.specifications.append(
                dc.EnforceGCContent(mini=gc_min, maxi=gc_max, window=gc_window)
            )

        if avoid_sites is not None and len(avoid_sites) > 0:
            for site in avoid_sites:
                if site not in rest_dict:
                    raise ValueError(
                        f"Restriction site {site} not available through biopython rest_dict"
                    )

                # explicitly specify we match to both strands for clarity even though defaults to "both" internally
                self.specifications.append(
                    dc.AvoidPattern(dc.EnzymeSitePattern(site), strand="both")
                )

        # Number of CPUs to use for parallelization
        self.cpu = cpu

    def optimize(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int,
        upstream_dna: str,
        downstream_dna: str,
        reference: SystemInstance | None = None,
        reference_dna: str | None = None,
    ) -> None:  # TODO: add proper return type
        # verify that vali


        # TODO: create sub-method that just operates on sequences (not instances etc.)
        # TODO: wrap all generic unpacking code in abstract class?
        # TODO: make use of mutations_space...
        # TODO: deduplicate input sequences (do not perform unnecessary optimizations)
        # TODO: parallelization? method-specific?
        # TODO: need to handle fixed base sequence (with inserts/gap)
        # TODO: need to make sure we have a stop codon
        # TODO: check provided sequences are valid according to alphabet
        # TODO: after optimization verify forward and reverse complement for absence of patterns or raise error
        # TODO: genetic_table and translation to EnforceTranslation
        """
        # create simple reverse translation
        insert_dna = dc.reverse_translate(insert_seq_uc_nogaps)

        problem = dc.DnaOptimizationProblem(
            sequence=raw_sequence,
            constraints=[dc.EnforceTranslation(), dc.AvoidHairpins()],
            objectives=[dc.CodonOptimize(
                species=species,
                method=method,
                location=opt_region
            )]
        )

        problem.resolve_constraints()
        print(problem.objectives_text_summary())
        raw_score = problem.objective_scores_sum()

        problem.optimize()

        print(problem.objectives_text_summary())
        opt_score = problem.objective_scores_sum()

        # raw = problem.sequence_before[opt_region[0]:opt_region[1]]
        # optimized = problem.sequence[opt_region[0]:opt_region[1]]

        # verify translations against original sequence
        assert Seq(raw_sequence).translate() == Seq(problem.sequence).translate()

        # make sure something changed
        assert Seq(raw_sequence) != Seq(problem.sequence)
        """
        raise NotImplementedError()
