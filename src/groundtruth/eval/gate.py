"""CI regression gate: fail the build when retrieval quality drops.

Run: `python -m groundtruth.eval.gate --baseline baseline.json --candidate run.json`

A metric nobody enforces is decoration. This is the script the GitHub Actions
workflow calls after running eval on a PR: it compares the candidate's nDCG@10
against a committed baseline and exits non-zero if the drop exceeds a tolerance.
That converts "we track nDCG" into "you cannot merge a change that quietly makes
retrieval worse" -- the property that makes the whole eval harness load-bearing
rather than ornamental.

Two design points worth defending:

* **The gate is one-sided.** Improvements never fail; only regressions beyond the
  tolerance do. The tolerance (default 2%) absorbs the run-to-run noise of an
  LLM-judged slice without letting a real regression through.

* **It checks the human-verified slice too, when present.** A change can look neutral
  on the noisy bootstrapped labels while regressing on the trusted slice; gating on
  both catches a judge-pleasing change that real users would feel.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast


def _load(path: str) -> dict[str, Any]:
    with open(path) as fh:
        payload: dict[str, Any] = json.load(fh)
    # Accept either a raw metrics dict or the full run_eval output envelope.
    result = payload.get("metrics", payload)
    return cast(dict[str, Any], result)


def _ndcg(metrics: dict[str, Any], slice_name: str) -> float | None:
    section = metrics.get(slice_name) or {}
    val = section.get("ndcg@10")
    return float(val) if isinstance(val, (int, float)) else None


def compare(
    baseline: dict[str, Any], candidate: dict[str, Any], *, tolerance: float
) -> tuple[bool, list[str]]:
    """Return (passed, human-readable lines). Regressions worse than `tolerance`
    (a fraction, e.g. 0.02 for 2%) on any checked slice fail the gate."""
    lines: list[str] = []
    passed = True

    for slice_name in ("overall", "human_verified"):
        base = _ndcg(baseline, slice_name)
        cand = _ndcg(candidate, slice_name)
        if base is None or cand is None:
            lines.append(f"{slice_name:16s} nDCG@10: skipped (missing in one run)")
            continue

        delta = cand - base
        # Guard against a zero/absent baseline producing a divide-by-zero.
        pct = (delta / base * 100) if base else 0.0
        arrow = "▲" if delta >= 0 else "▼"
        verdict = "ok"
        # Absolute floor mirrors the fractional tolerance so a near-zero baseline
        # can't make any drop look catastrophic in percentage terms. The 1e-9
        # epsilon keeps a drop of *exactly* the tolerance on the passing side --
        # otherwise float representation error alone could flip a boundary tie.
        floor = -tolerance * max(base, 1e-9)
        if delta < floor - 1e-9:
            passed = False
            verdict = "REGRESSION"
        lines.append(
            f"{slice_name:16s} nDCG@10: {base:.4f} -> {cand:.4f} ({arrow} {pct:+.2f}%) {verdict}"
        )

    return passed, lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail CI on retrieval-quality regression")
    ap.add_argument("--baseline", required=True, help="committed baseline metrics JSON")
    ap.add_argument("--candidate", required=True, help="this PR's eval output JSON")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="max fractional nDCG@10 drop tolerated before failing (default 2%%)",
    )
    args = ap.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    passed, lines = compare(baseline, candidate, tolerance=args.tolerance)

    print("=" * 60)
    print(f"Retrieval regression gate (tolerance {args.tolerance:.0%})")
    print("-" * 60)
    for line in lines:
        print("  " + line)
    print("=" * 60)

    if not passed:
        print("GATE FAILED: retrieval quality regressed beyond tolerance.", file=sys.stderr)
        sys.exit(1)
    print("GATE PASSED.")


if __name__ == "__main__":
    main()
