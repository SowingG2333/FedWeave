from __future__ import annotations

from typing import Any, Dict, Optional


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten(child_prefix, child, out)
        return
    if isinstance(value, (list, tuple)):
        return
    out[prefix] = value


def flatten_metrics(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    _flatten(prefix, payload, flat)
    return flat


class SwanLabRun:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        name: str,
        mode: str,
        config: Dict[str, Any],
    ) -> None:
        self.enabled = bool(enabled)
        self._run: Optional[Any] = None
        if not self.enabled:
            return
        try:
            import swanlab  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is enabled but `swanlab` is not installed. "
                "Install requirements-optional.txt or rerun with `--no_use_swanlab`."
            ) from exc

        init_kwargs = {
            "project": str(project),
            "mode": str(mode),
            "config": config,
        }
        if str(name).strip():
            init_kwargs["experiment_name"] = str(name).strip()
        self._run = swanlab.init(**init_kwargs)

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self._run is None:
            return
        import swanlab  # type: ignore

        if step is None:
            swanlab.log(payload)
        else:
            swanlab.log(payload, step=int(step))

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        import swanlab  # type: ignore

        finish_fn = getattr(swanlab, "finish", None)
        if callable(finish_fn):
            finish_fn()
