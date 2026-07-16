"""
Progress-bar helpers.

tqdm is an optional dependency. If it isn't installed, these fall back to
no-op iterators so training still runs (and still prints its per-epoch log
lines) rather than failing on an import -- which matters in deployment
environments where adding a package means clearing a mirror first.
"""

from typing import Iterable, Optional

try:
    from tqdm.auto import tqdm as _tqdm
    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    TQDM_AVAILABLE = False


class _NullBar:
    """Minimal stand-in exposing the parts of the tqdm API used here."""

    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, n=1):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass


def progress_bar(
    iterable: Optional[Iterable] = None,
    enabled: bool = True,
    **kwargs,
):
    """
    tqdm bar when available and enabled, otherwise a no-op passthrough.

    Shows elapsed and estimated-remaining time by default, which is the point
    of using it on runs that take tens of minutes.
    """
    if not enabled or not TQDM_AVAILABLE:
        return _NullBar(iterable, **kwargs)
    return _tqdm(iterable, **kwargs)


def tqdm_write(msg: str):
    """
    Print without corrupting an active tqdm bar.

    A bare print() during a live bar interleaves with the bar's redraw and
    garbles both; tqdm.write() clears the bar, writes, and redraws.
    """
    if TQDM_AVAILABLE:
        _tqdm.write(msg)
    else:
        print(msg)
