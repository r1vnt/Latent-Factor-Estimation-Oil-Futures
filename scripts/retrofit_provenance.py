"""One-time (and re-runnable) retrofit: add a `_provenance` envelope to
every results/*.json, non-destructively -- existing keys untouched, so
every explore script that already reads these files keeps working.

producing_script is found by grep, not by a hand-maintained mapping
table: every explore script in this project writes its output via a
literal `out_path = ROOT / "results" / "<filename>.json"` (or the f-string
equivalent for per-block-out files), so the exact writer is discoverable
mechanically and reliably, rather than guessed from filename conventions
that have drifted over 65 scripts.

git_commit/timestamp_utc come from `git log -1` on the JSON file itself --
the commit that last touched the artefact, which is what "when was this
number produced" actually means for a committed result. A file with no
git history (never committed) gets git_commit=None, flagged explicitly
rather than silently omitted.

Run: python scripts/retrofit_provenance.py [--dry-run]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csr.provenance import package_version, panel_hash  # noqa: E402

RESULTS_DIR = ROOT / "results"
EXPLORE_DIR = ROOT / "explore"

# A results-path construction on a line: ROOT / "results" / "name.json"
# (or the f-string form for per-block-out files).
PATH_PATTERN = re.compile(r'ROOT\s*/\s*"results"\s*/\s*(?:f?)"([^"{}]*)(?:\{[^}]*\})?([^"{}]*)"')


def _is_write_line(line: str) -> bool:
    """A line containing a results-path construction is a WRITE if it
    either (a) is itself immediately followed by .write_text(, i.e. the
    inline `(ROOT / "results" / "x.json").write_text(...)` convention
    used by explore/27-37-ish, or (b) assigns to a variable named
    out_path* (the two-step `out_path = ...` / `out_path.write_text(...)`
    convention used everywhere since). Anything else (json.loads((...)
    .read_text())) is a READ, not a writer."""
    stripped = line.strip()
    if ".write_text(" in line:
        return True
    if re.match(r'out_path\w*\s*=', stripped):
        return True
    return False


def _find_producing_script(json_filename: str) -> str | None:
    """Grep every explore/*.py for a WRITE of this exact filename. Exact
    literal matches first; f-string PREFIX/SUFFIX matches (for per-block-
    out files like c1c18_study_a3_{BLOCK}.json) as a fallback."""
    exact_hits = []
    prefix_hits = []
    for script in sorted(EXPLORE_DIR.glob("*.py")):
        for line in script.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = PATH_PATTERN.search(line)
            if not m or not _is_write_line(line):
                continue
            prefix, suffix = m.group(1), m.group(2)
            if prefix + suffix == json_filename:
                exact_hits.append(script.name)
            elif json_filename.startswith(prefix) and json_filename.endswith(suffix) and (prefix or suffix):
                prefix_hits.append(script.name)
    if exact_hits:
        return f"explore/{exact_hits[0]}" + ("" if len(exact_hits) == 1 else f"  [ALSO: {exact_hits[1:]}]")
    if prefix_hits:
        return f"explore/{prefix_hits[0]}" + ("" if len(prefix_hits) == 1 else f"  [ALSO: {prefix_hits[1:]}]")
    return None


def _git_info(path: Path) -> tuple[str | None, str | None]:
    res = subprocess.run(
        ["git", "log", "-1", "--format=%H|%aI", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True,
    )
    line = res.stdout.strip()
    if not line:
        return None, None
    commit, ts = line.split("|", 1)
    return commit, ts


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    ph = panel_hash()
    pv = package_version()
    print(f"panel_hash={ph}  package_version={pv}  dry_run={dry_run}")

    n_ok, n_no_script, n_no_git = 0, 0, 0
    unmatched = []
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(jf.read_text())
        script = _find_producing_script(jf.name)
        commit, ts = _git_info(jf)
        if script is None:
            n_no_script += 1
            unmatched.append(jf.name)
        if commit is None:
            n_no_git += 1

        data["_provenance"] = {
            "result_id": jf.stem,
            "producing_script": script,
            "git_commit": commit,
            "timestamp_utc": ts,
            "panel_hash": ph,
            "package_version": pv,
        }
        n_ok += 1
        if not dry_run:
            jf.write_text(json.dumps(data, indent=2, default=str))

    print(f"\n{n_ok} files processed, {n_no_script} with no producing script found, "
         f"{n_no_git} with no git history (uncommitted).")
    if unmatched:
        print("no producing script found for:")
        for name in unmatched:
            print(f"  {name}")


if __name__ == "__main__":
    main()
