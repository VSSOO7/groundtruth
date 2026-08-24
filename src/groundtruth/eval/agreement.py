"""Judge accountability: how far can the cheap labeler be trusted?

Run: `python -m groundtruth.eval.agreement`

`build_labels.py` grades thousands of (query, chunk) pairs with the cheap model.
`eval_labels` keeps machine and human grades side by side for the same pair, which
lets us answer the question most RAG write-ups skip: **does the automated labeler
actually agree with a human?**

Reported here:

* **Cohen's kappa** on the exact 0-3 grades -- chance-corrected, so a labeler that
  scores everything "2" gets no credit for accidental hits.
* **Binary kappa** on relevant (>=2) vs not, which is the decision the retriever
  is really being graded on and is usually the more forgiving, more honest number.
* **Off-by-one rate**, because a 2-vs-3 disagreement barely moves nDCG while a
  0-vs-3 disagreement inverts the ranking.

Rough reading of kappa: <0.20 poor, 0.21-0.40 fair, 0.41-0.60 moderate,
0.61-0.80 substantial, >0.80 near-perfect. Below "moderate" on the binary view,
the bootstrapped labels are too noisy to trust for gating and the human slice
should carry the decision.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from psycopg import Connection
from psycopg.rows import dict_row

from groundtruth.config import get_settings
from groundtruth.eval.metrics import cohens_kappa

log = structlog.get_logger(__name__)


def load_paired_grades(conn: Connection) -> list[dict[str, Any]]:
    """Pairs graded by BOTH the machine and a human -- the only comparable rows."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT m.query_id, m.chunk_id,
                      m.relevance AS machine,
                      h.relevance AS human
                 FROM eval_labels m
                 JOIN eval_labels h
                   ON h.query_id = m.query_id
                  AND h.chunk_id = m.chunk_id
                  AND h.label_source = 'human_verified'
                WHERE m.label_source = 'llm_bootstrap'
                ORDER BY m.query_id, m.chunk_id"""
        )
        return cur.fetchall()


def report(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    machine = [int(p["machine"]) for p in pairs]
    human = [int(p["human"]) for p in pairs]

    # Binary view: "would this chunk count as relevant?" -- the decision that
    # actually drives retrieval quality.
    machine_bin = [1 if g >= 2 else 0 for g in machine]
    human_bin = [1 if g >= 2 else 0 for g in human]

    exact = sum(1 for m, h in zip(machine, human, strict=True) if m == h)
    off_by_one = sum(1 for m, h in zip(machine, human, strict=True) if abs(m - h) == 1)
    severe = sum(1 for m, h in zip(machine, human, strict=True) if abs(m - h) >= 2)
    n = len(pairs)

    return {
        "n_pairs": n,
        "kappa_graded": round(cohens_kappa(machine, human), 4),
        "kappa_binary_relevant": round(cohens_kappa(machine_bin, human_bin), 4),
        "exact_agreement": round(exact / n, 4),
        "off_by_one_rate": round(off_by_one / n, 4),
        "severe_disagreement_rate": round(severe / n, 4),
    }


def interpret(kappa: float) -> str:
    if kappa < 0.20:
        return "poor -- do not gate on bootstrapped labels"
    if kappa < 0.41:
        return "fair -- treat bootstrapped labels as directional only"
    if kappa < 0.61:
        return "moderate -- usable for ranking comparisons, report the human slice too"
    if kappa < 0.81:
        return "substantial"
    return "near-perfect"


def main() -> None:
    settings = get_settings()
    conn = Connection.connect(settings.database_url, autocommit=True)
    pairs = load_paired_grades(conn)
    conn.close()

    if not pairs:
        raise SystemExit(
            "No pairs graded by both the machine and a human.\n"
            "Hand-verify a slice first: insert the same (query_id, chunk_id) rows "
            "with label_source='human_verified'."
        )

    stats = report(pairs)
    stats["verdict_graded"] = interpret(stats["kappa_graded"])
    stats["verdict_binary"] = interpret(stats["kappa_binary_relevant"])

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
