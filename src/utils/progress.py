from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


class _NullProgress:
    def __init__(self, iterable: Optional[Iterable[Any]] = None) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(()) if self._iterable is None else iter(self._iterable)

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, ordered_dict: Optional[Mapping[str, Any]] = None, refresh: bool = True, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def make_progress(
    *,
    iterable: Optional[Iterable[Any]] = None,
    total: Optional[int] = None,
    desc: str = "",
    disable: bool = False,
    leave: bool = True,
) -> Any:
    if disable or _tqdm is None:
        return _NullProgress(iterable=iterable)
    return _tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        disable=disable,
        leave=leave,
        dynamic_ncols=True,
    )


def progress_write(message: str, *, disable: bool = False) -> None:
    if disable:
        return
    if _tqdm is not None:
        _tqdm.write(message)
        return
    print(message)


def format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_task_metrics(task_metrics: Mapping[str, Any]) -> str:
    if not task_metrics:
        return "-"
    return ",".join(f"{task}:{format_metric(value)}" for task, value in sorted(task_metrics.items()))
