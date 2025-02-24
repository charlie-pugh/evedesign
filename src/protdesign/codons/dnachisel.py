"""
Codon optimization with the DNA Chisel package

TODO: if implementing any other codon optimizers in the future, rethink interfaces and extract
 shared functionality into abstract base class
"""
import multiprocess as mp
from typing import Sequence, Literal, Any
import pandas as pd

try:
    import dnachisel as dc  # noqa
    from dnachisel.builtin_specifications.codon_optimization.BaseCodonOptimizationClass import (
        BaseCodonOptimizationClass  # noqa
    )
    from Bio.Seq import Seq
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
        extra_constraints: Sequence[dc.Specification] | None = (
            dc.AvoidHairpins(),
            dc.AvoidPattern(dc.HomopolymerPattern("A", 5)),
            dc.AvoidPattern(dc.HomopolymerPattern("C", 5)),
            dc.AvoidPattern(dc.HomopolymerPattern("G", 5)),
            dc.AvoidPattern(dc.HomopolymerPattern("T", 5)),
            dc.AvoidPattern(dc.RepeatedKmerPattern(2, 5)),
        ),
        genetic_code: str = "Standard",
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

                # explicitly specify we match to both strands for clarity even though defaults
                # to "both" internally
                self.specifications.append(
                    dc.AvoidPattern(dc.EnzymeSitePattern(site), strand="both")
                )

        # Number of CPUs to use for parallelization
        self.cpu = cpu

    def _optimize_seq(
        self,
        seq: str,
        upstream_dna: str,
        downstream_dna: str,
        reference_seq: str | None = None,
        reference_dna: str | None = None,
    ) -> tuple[str, float]:
        """
        Codon-optimize a single sequence

        Parameters
        ----------
        seq
            Protein sequence for which to create codon-optimized coding DNA sequence
        upstream_dna
            Upstream nucleotides before coding sequence (e.g. assembly/cloning overhangs)
        downstream_dna
            Downstream nucleotides after coding sequence (e.g. assembly/cloning overhangs)
        reference_seq

        reference_dna

        Returns
        -------
        Tuple with
         (i) optimized DNA sequence for seq (*excluding* upstream and downstream DNA)
         (ii) final optimization score
        """
        # TODO: need to handle fixed base sequence (with inserts/gap)
        # TODO: genetic_table and translation to EnforceTranslation
        # TODO: check reference seq and dna are both specified

        seq_norm = seq  # TODO: need to normalize seq
        upstream_dna = upstream_dna.upper()
        downstream_dna = downstream_dna.upper()

        # if no reference given, simply initialize the sequence
        if reference_dna is None:
            seq_dna = dc.reverse_translate(seq)   # TODO: use normalized sequence?
        else:
            # TODO: set up optimization relative to reference, need to fill in insertions
            #  and mark positions to fix
            seq_dna = None

        # full sequence context for optimization problem
        full_dna = upstream_dna + seq_dna + downstream_dna

        # region in full_dna to optimize (corresponds to seq_dna, i.e. keep upstream/downstream sequence fixed)
        seq_dna_start = len(upstream_dna)
        seq_dna_end = len(upstream_dna) + len(seq_dna)
        seq_dna_loc = (seq_dna_start, seq_dna_end)

        # enforce correct translation of sequence and do not change upstream/downstream sequences
        seq_constraints = [
            dc.EnforceTranslation(
                location=seq_dna_loc, genetic_table=self.genetic_code, translation=seq_norm
            ),
            dc.AvoidChanges(
                location=(0, seq_dna_start),
            ),
            dc.AvoidChanges(
                location=(seq_dna_end, len(full_dna)),
            )
        ]

        problem = dc.DnaOptimizationProblem(
            sequence=full_dna,
            constraints=self.specifications + seq_constraints,
            objectives=[dc.CodonOptimize(
                codon_usage_table=self.codon_table,
                method=self.method,
                location=seq_dna_loc,
            )],
            logger=None
        )

        # raw_score = problem.objective_scores_sum()
        problem.resolve_constraints()
        problem.optimize()
        opt_score = problem.objective_scores_sum()

        # extract full optimized sequence with upstream/downstream DNA
        dna_opt = problem.sequence
        assert len(dna_opt) == len(upstream_dna) + len(seq_dna) + len(downstream_dna)
        assert dna_opt[:seq_dna_start] == upstream_dna, "Upstream DNA sequence does not match input"
        assert dna_opt[seq_dna_end:] == downstream_dna, "Downstream DNA sequence does not match input"

        # extract optimized protein-coding DNA sequence and verify it translates correctly
        dna_seq_opt = dna_opt[seq_dna_start:seq_dna_end]
        dna_seq_transl = Seq(dna_seq_opt).translate(table=self.genetic_code)
        assert dna_seq_transl == seq_norm, "Translation of optimized sequence does not match input"

        # print(problem.mutation_space.string_representation())
        # print(problem.objectives_text_summary())

        return dna_seq_opt, opt_score

    def optimize(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        entity: int,
        upstream_dna: str,
        downstream_dna: str,
        reference: SystemInstance | None = None,
        reference_dna: str | None = None,
    ) -> Any:  # TODO: add proper return type
        # TODO: documentation
        # TODO: extract all generic unpacking code in abstract class?
        # verify that valid entity is selected
        if not 0 <= entity <= len(system):
            raise ValueError("Invalid entity index")

        if system[entity].type_ != "protein":
            raise ValueError("Can only optimize protein entities")

        # validate provided instances
        [
            system.valid_instance(
                instance, validate_reps=True, fixed_length=False, allow_deletions=True, raise_invalid=True,
            ) for instance in instances
        ]

        # check if we optimize a given reference sequence
        if reference is not None:
            # validate reference first
            system.valid_instance(
                reference, validate_reps=True, fixed_length=False, allow_deletions=True, raise_invalid=True,
            )

            # create normalized and raw versions of instance sequence
            # (the latter to keep potential alignment information)
            reference_seq_norm = "".join(reference[entity].normalized_rep())
            reference_seq = "".join(reference[entity].rep)

            # if we don't have a reference sequence, optimize it
            if reference_dna is None:
                # optimize reference sequence first (as this is reference, do this without being constrained
                # by any other sequence)
                reference_dna = self._optimize_seq(
                    seq=reference_seq_norm, upstream_dna=upstream_dna, downstream_dna=downstream_dna
                )
            else:
                # verify that reference_dna has valid length and translation matches (with specified genetic code)
                if len(reference_dna) != len(reference_seq_norm) * 3:
                    raise ValueError(
                        "reference_dna length must be length of instance sequence * 3"
                    )

                if Seq(reference_dna).translate(table=self.genetic_code) != reference_seq_norm:
                    raise ValueError(
                        "reference_dna does not translate into reference instance sequence"
                    )
        else:
            reference_seq = None
            reference_seq_norm = None

        print("REF DNA", reference_dna)

        # print("REFERENCE", reference_seq)  # TODO: remove
        # print("REFERENCE NORM", reference_seq_norm)  # TODO: remove
        # print("REFERENCE DNA", reference_dna)   # TODO: remove

        return  # TODO: remove

        # extract and deduplicate protein sequences (do not perform unnecessary optimizations);
        # do not normalize to keep potential alignment information
        unique_seqs = pd.Series(
            "".join(inst[entity].rep) for inst in instances
        ).drop_duplicates().tolist()
        print("unique seqs", unique_seqs)  # TODO: remove

        # TODO: parallelization? method-specific?
        # TODO: after optimization verify forward and reverse complement for absence of patterns or raise error
        return
