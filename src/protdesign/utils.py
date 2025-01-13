from contextlib import contextmanager
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def ensure_sequence(x: T | Sequence[T]) -> Sequence[T]:
    if isinstance(x, Sequence):
        return x
    else:
        return [x]


@contextmanager
def model_param_context(
    load_func: Callable[[], None],
    delete_func: Callable[[], None],
    keep_model: bool
):
    # TODO: don't reload
    try:
        load_func()
        yield
    finally:
        if not keep_model:
            print("DELETE")
            delete_func()
        else:
            print("KEEP")
