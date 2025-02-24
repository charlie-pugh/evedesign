"""
Generic interfaces for building DNA sequences from designed proteins
"""
from abc import ABC, abstractmethod
from typing import Sequence
from protdesign.entity import System, SystemInstance

class CodonOptimizer(ABC):
    """
    TODO: species
    TODO: method (method-specific?)
    TODO: restriction sites to avoid
    TODO: constraints (hairpins etc.)
    TODO: parallelization? method-specific?
    TODO: overhangs at both ends
    TODO: add proper return type
    """

    @abstractmethod
    def optimize(
        self,
        system: System,
        instances: Sequence[SystemInstance],
        dna_overhang_n: str, # TODO: this needs to be entity-specific
        dna_overhang_c: str,  # TODO: this needs to be entity-specific
        dna_template: str | None = None,
    ):
        # TODO: documentation
        pass