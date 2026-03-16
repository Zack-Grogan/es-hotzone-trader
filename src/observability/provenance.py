from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Optional

from src.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_NAME = "es-hotzone-trader"


def collect_run_provenance(
    config: Config,
    *,
    config_path: Optional[str],
    log_path: Path,
    sqlite_path: str,
    data_mode: str,
    health_url: str,
    debug_url: str,
    mcp_url: Optional[str],
) -> dict[str, Any]:
    resolved_config_path = _resolve_config_path(config_path)
    git_metadata = _read_git_metadata(PROJECT_ROOT)
    return {
        "started_at": _iso_now(),
        "process_id": os.getpid(),
        "project_root": str(PROJECT_ROOT),
        "symbols": list(config.symbols),
        "data_mode": data_mode,
        "app_version": _read_package_version(),
        "config_path": str(resolved_config_path) if resolved_config_path is not None else None,
        "config_hash": _hash_config(config, resolved_config_path),
        "log_path": str(log_path),
        "sqlite_path": str(sqlite_path),
        "health_url": health_url,
        "debug_url": debug_url,
        "mcp_url": mcp_url,
        **git_metadata,
    }


def _resolve_config_path(config_path: Optional[str]) -> Optional[Path]:
    if config_path:
        return Path(config_path).expanduser().resolve()
    default_path = PROJECT_ROOT / "config" / "default.yaml"
    return default_path.resolve() if default_path.exists() else None


def _hash_config(config: Config, resolved_config_path: Optional[Path]) -> str:
    if resolved_config_path is not None and resolved_config_path.exists():
        return sha256(resolved_config_path.read_bytes()).hexdigest()
    payload = json.dumps(_normalize_value(asdict(config)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def _read_git_metadata(project_root: Path) -> dict[str, Any]:
    repo_root = _discover_git_root(project_root)
    if repo_root is None:
        return {
            "git_available": False,
            "git_repo_root": None,
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
        }
    branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    status = _run_git(repo_root, "status", "--porcelain")
    return {
        "git_available": branch is not None or commit is not None,
        "git_repo_root": str(repo_root),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status) if status is not None else None,
    }


def _discover_git_root(start: Path) -> Optional[Path]:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _read_package_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.1.0"
    except Exception:
        return "unknown"


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
