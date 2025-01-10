from typing import Any, Callable, Literal, Sequence, TypeVar

T = TypeVar("T")
DeviceType = Literal["cpu", "gpu"]
Status = Literal["running", "done", "failed"]
# status, progress (optional), message (optional)
StatusCallback = Callable[[Status, float | None, str | None], Any]


def ensure_sequence(x: T | Sequence[T]) -> Sequence[T]:
    if isinstance(x, Sequence):
        return x
    else:
        return [x]
