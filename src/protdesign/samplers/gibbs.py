"""
Sequence generation with Gibbs sampling.

Implementation assumes fixed length of sequences (no inserts, deletions can be sampled if part of alphabet).
"""
from typing import List, Sequence
from protdesign.model import Generator, Scorer
from protdesign.entity import Instance
from protdesign.types import StatusCallback


class GibbsSampler(Generator):
    """
    Gibbs sampling from linear combination of Scorers
    # TODO: also implement Scorer interface to allow sequence scoring under joint energy function
    # TODO: separate single mutant scorers P(x_i | x_\i) into separate interface?
    """
    def __init__(
        self,
        scorers: List[Scorer],
        weights: List[float] | None = None,
    ):
        # TODO: assume all weights to be 1.0 if weights is None
        # TODO: add temperature and temperature schedule
        # TODO: add init strategy
        # TODO: add position sampling strategy
        # TODO: number of chains
        # TODO: number of steps
        pass

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: Sequence[Sequence[int]] | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None
    ) -> List[Instance]:
        # TODO: implement generation parameters
        # TODO: implement actual generation
        return []