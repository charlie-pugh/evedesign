from typing import Literal, Callable, Any, Dict
import numpy as np

BioPolymers = {"protein", "dna", "rna"}
BioPolymer = Literal["protein", "dna", "rna"]
EntityType = BioPolymer
DeviceType = Literal["cpu", "cuda", "mps"]
BatchSize = int | Literal["auto"] | None
Metadata = Dict[str, Any]

# status, progress (optional), message (optional)
Status = Literal["running", "done", "failed"]
StatusCallback = Callable[[Status, float | None, str | None], Any]
RepSequence = np.ndarray[tuple[int], np.dtype["U1"]]