"""Provenance envelope for results/*.json -- lets the notebook (and anyone
else) assert that every loaded number came from the same panel and state
which script, commit, and moment produced it, rather than trusting a bare
JSON blob on faith.

panel_hash() is deliberately simple: sha256 of the one canonical source
file every explore script reads (panel_delivery/panel_c1_c24.parquet),
not a re-derivation from Panel.from_prices. Every script in this project
applies the identical >=2010-01-01 filter and first-18-column truncation,
so hashing the source file is sufficient to detect "this result came from
a different panel" and far simpler than hashing a constructed array.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PANEL_PATH = ROOT / "panel_delivery" / "panel_c1_c24.parquet"


def panel_hash(path: Path = CANONICAL_PANEL_PATH, digest_len: int = 16) -> str:
    """sha256 of the canonical panel source file, truncated for display.
    Raises FileNotFoundError loudly rather than silently returning a
    placeholder -- a provenance check that can't see the panel is not a
    provenance check."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:digest_len]


def package_version() -> str:
    """git describe if this is a git checkout (most informative -- ties
    the version to an exact commit even between releases), falling back
    to the pyproject.toml version string."""
    import subprocess
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:
        import tomllib
        with open(ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return "unknown"
