from contextlib import contextmanager
from typing import Callable, Sequence, TypeVar, Mapping
import numpy as np
from protdesign.types import StatusCallback

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
    try:
        load_func()
        yield
    finally:
        if not keep_model:
            delete_func()
        else:
            pass

def status_start(
    status_callback: StatusCallback | None, message: str | None = None
):
    if status_callback is not None:
        status_callback("running", None, message)

def status_done(
    status_callback: StatusCallback | None, message: str | None = None
):
    if status_callback is not None:
        status_callback("done", None, message)

def status_progress(
    status_callback: StatusCallback | None, progress: float
):
    if status_callback is not None:
        status_callback("running", progress, None)

def shorten(text: str, max_len=50):
    return (text[:max_len] + "...") if len(text) > 50 else text

def str_to_np_char_view(x: Sequence[str]):
    """
    Quickly transform a list of strings into a numpy character
    array (much faster than np.array([list(s) for s in x])
    and return a view

    Parameters
    ----------
    x
        List of equal length strings (not checked here)

    Returns
    -------
    2D character array
    """
    x_np = np.array(
        x, dtype=np.str_
    )

    return x_np.view(
        "U1"
    ).reshape(
        (x_np.size, -1)
    )

def map_array(x: np.ndarray, map_: Mapping) -> np.ndarray:
    """
    Efficiently map elements of a numpy array

    Parameters
    ----------
    x
        Array to be mapped
    map_
        Mapping to be applied to individual elements
        (to cover potentially missing values, use a defaultdict)

    Returns
    -------

    """
    return np.vectorize(
        map_.__getitem__
    )(x)
