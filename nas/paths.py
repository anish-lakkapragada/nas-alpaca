"""Repository paths and launch commands for packaged and archived layouts."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_SUPPORT = ("nas/__init__.py", "nas/paths.py")


def source_path(name, project_root=PROJECT_ROOT):
    """Resolve a recorded source name, including the original flat layout."""
    root = Path(project_root)
    relative = Path(name)
    original = root / relative
    if original.exists() or len(relative.parts) != 1:
        return original
    for folder in ("nas", "nas/cloud", "nas/mlx", "tests"):
        candidate = root / folder / relative
        if candidate.is_file():
            return candidate
    return original


def python_command(name, project_root=PROJECT_ROOT, *, unbuffered=True):
    """Use module execution for packages and scripts for archived checkouts."""
    root = Path(project_root).resolve()
    path = source_path(name, root).resolve()
    prefix = [sys.executable] + (["-u"] if unbuffered else [])
    relative = path.relative_to(root)
    if relative.parts[0] in ("nas", "tests") and path.suffix == ".py":
        return prefix + ["-m", ".".join(relative.with_suffix("").parts)]
    return prefix + [str(path)]
