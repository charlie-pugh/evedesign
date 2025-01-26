from typing import Literal, Callable, Any, Dict

BioPolymer = Literal["protein", "dna", "rna"]
EntityType = BioPolymer
DeviceType = Literal["cpu", "cuda", "mps"]
BatchSize = int | Literal["auto"] | None
Metadata = Dict[str, int | str | bool]

# status, progress (optional), message (optional)
Status = Literal["running", "done", "failed"]
StatusCallback = Callable[[Status, float | None, str | None], Any]
