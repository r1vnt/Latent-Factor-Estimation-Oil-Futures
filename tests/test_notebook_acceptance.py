"""Three acceptance checks for notebooks/evidence_index.ipynb and
numbers.tex, per the evidence-index design: the notebook is only useful
if it stays honest as the underlying results evolve.

1. Restart-and-run-all completes under 90s with zero exceptions.
2. Every number.tex macro traces to a JSON whose panel_hash matches the
   canonical one this session.
3. No results/*.json artefact is older (by its own last-touching commit)
   than the last commit to the explore/ script that produced it -- the
   notebook-figure equivalent of a stale-figure check.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from csr.provenance import panel_hash  # noqa: E402


@pytest.mark.slow
def test_notebook_restart_and_run_all_under_90s_no_exceptions():
    nb_path = ROOT / "notebooks" / "evidence_index.ipynb"
    t0 = time.monotonic()
    res = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--output", "_acceptance_check.ipynb",
         "--ExecutePreprocessor.kernel_name=csr-nb", "--ExecutePreprocessor.timeout=120",
         str(nb_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    elapsed = time.monotonic() - t0
    out_path = nb_path.parent / "_acceptance_check.ipynb"
    try:
        assert res.returncode == 0, f"nbconvert failed:\n{res.stderr[-4000:]}"
        assert elapsed < 90, f"restart-and-run-all took {elapsed:.1f}s, budget is 90s"

        import nbformat
        nb = nbformat.read(out_path, as_version=4)
        errors = [
            (i, o.get("ename"), o.get("evalue"))
            for i, c in enumerate(nb.cells) if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error"
        ]
        assert not errors, f"cell execution errors: {errors}"
    finally:
        if out_path.exists():
            out_path.unlink()


def test_numbers_tex_traces_to_canonical_panel_hash():
    import emit_numbers_tex

    canonical = panel_hash()
    macros = emit_numbers_tex.build_macros()
    assert len(macros) >= 20, "fewer headline macros than expected -- emitter may have regressed"

    # Every _load() inside build_macros() already asserts existence; here
    # we additionally re-open each cited source and check its panel_hash.
    cited_sources = set()
    for _, _, source_comment in macros:
        # source comments are "filename.json: field.path" or similar
        fname = source_comment.split(":")[0].strip().split()[0]
        if fname.endswith(".json"):
            cited_sources.add(fname)

    mismatches = []
    for fname in cited_sources:
        path = ROOT / "results" / fname
        if not path.exists():
            mismatches.append((fname, "file not found"))
            continue
        data = json.loads(path.read_text())
        h = data.get("_provenance", {}).get("panel_hash")
        if h != canonical:
            mismatches.append((fname, f"panel_hash {h} != canonical {canonical}"))
    assert not mismatches, f"numbers.tex sources with a stale/missing panel hash: {mismatches}"


def test_no_result_artifact_predates_its_producing_scripts_last_commit():
    """A `_provenance.verified_noop_diff: true` flag is the only escape
    hatch, and it is set manually, one file at a time, only after
    reading the actual `git diff` between the artifact's commit and HEAD
    and confirming the change cannot affect the computed values (e.g. a
    return-arity fix to an unused return slot). retrofit_provenance.py
    never sets this itself -- an automatic pass that could silence this
    check on its own would defeat the point of having it."""
    def git_commit_date(path: Path) -> str | None:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", str(path.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, text=True,
        )
        return res.stdout.strip() or None

    stale = []
    for jf in sorted((ROOT / "results").glob("*.json")):
        data = json.loads(jf.read_text())
        prov = data.get("_provenance", {})
        script = prov.get("producing_script")
        artifact_ts = prov.get("timestamp_utc")
        if not script or not artifact_ts:
            continue   # unmatched/uncommitted -- not this check's concern
        script = script.split("  [ALSO:")[0].strip()
        script_path = ROOT / script
        if not script_path.exists():
            continue
        script_ts = git_commit_date(script_path)
        if script_ts is None:
            continue
        if script_ts > artifact_ts and not prov.get("verified_noop_diff"):
            stale.append((jf.name, script, artifact_ts, script_ts))
    assert not stale, (
        f"artifacts older than their producing script's last commit "
        f"(figure/number may be stale relative to the code that made it): {stale}"
    )
