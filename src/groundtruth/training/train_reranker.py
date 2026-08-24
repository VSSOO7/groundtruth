"""Train the XGBoost LambdaMART reranker.

Run: `python -m groundtruth.training.train_reranker --out models/reranker.ubj`

Three things here are the difference between a real learning-to-rank setup and a
tutorial:

1. **Grouped splits.** Ranking data has one group per query. The train/validation
   split is by *query*, never by row -- otherwise chunks from the same query land
   on both sides and the model memorizes the query instead of learning to rank.
   This is the single most common way LTR benchmarks get silently inflated.

2. **`rank:ndcg` with graded labels.** The objective optimizes the same metric the
   eval harness reports, using the same exponential-gain convention (see
   `eval/metrics.py`), so training signal and reported score agree.

3. **Feature order pinned to disk.** A `.features.json` sidecar records the exact
   feature layout; the serving path refuses to load a model whose layout no longer
   matches the code. Train/serve skew in a reranker is invisible -- results still
   come back ranked, just worse -- so it has to be made loud.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import structlog
import xgboost as xgb

from groundtruth.eval.metrics import ndcg_at_k
from groundtruth.retrieval.features import FEATURE_NAMES

log = structlog.get_logger(__name__)


def group_split(
    qids: np.ndarray, *, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices by query id so no query appears in both sets."""
    unique = np.unique(qids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_fraction))
    val_qids = set(unique[:n_val].tolist())

    val_mask = np.array([q in val_qids for q in qids])
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def group_sizes(qids: np.ndarray) -> list[int]:
    """XGBoost ranking needs contiguous per-query group sizes, in row order."""
    sizes: list[int] = []
    current, count = None, 0
    for q in qids:
        if q != current:
            if current is not None:
                sizes.append(count)
            current, count = q, 1
        else:
            count += 1
    if current is not None:
        sizes.append(count)
    return sizes


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the JSONL training set produced by `build_training_set.py`.

    Each line: {"qid": int, "chunk_id": int, "relevance": 0-3, "features": [...]}
    Rows are sorted by qid so group sizes are contiguous.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No training rows in {path}")

    rows.sort(key=lambda r: r["qid"])

    bad = [r for r in rows if len(r["features"]) != len(FEATURE_NAMES)]
    if bad:
        raise SystemExit(
            f"{len(bad)} rows have wrong feature width (expected {len(FEATURE_NAMES)}). "
            "Regenerate the training set -- features.py changed since it was built."
        )

    X = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.asarray([r["relevance"] for r in rows], dtype=np.float32)
    qids = np.asarray([r["qid"] for r in rows])
    chunk_ids = np.asarray([r["chunk_id"] for r in rows])
    return X, y, qids, chunk_ids


def evaluate_ndcg(
    booster: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    qids: np.ndarray,
    chunk_ids: np.ndarray,
    k: int = 10,
) -> float:
    """Mean nDCG@k over held-out queries, scored through our own metric code."""
    scores = booster.predict(xgb.DMatrix(X, feature_names=list(FEATURE_NAMES)))

    by_query: dict[int, list[tuple[float, int, float]]] = defaultdict(list)
    for score, qid, cid, rel in zip(scores, qids, chunk_ids, y, strict=True):
        by_query[int(qid)].append((float(score), int(cid), float(rel)))

    per_query = []
    for rows in by_query.values():
        rows.sort(key=lambda r: r[0], reverse=True)
        ranked_ids = [cid for _, cid, _ in rows]
        labels = {cid: rel for _, cid, rel in rows}
        per_query.append(ndcg_at_k(ranked_ids, labels, k=k))

    return float(np.mean(per_query)) if per_query else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the LambdaMART reranker")
    ap.add_argument("--data", type=Path, default=Path("data/training/reranker.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("models/reranker.ubj"))
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    X, y, qids, chunk_ids = load_dataset(args.data)
    train_idx, val_idx = group_split(qids, val_fraction=args.val_fraction, seed=args.seed)

    log.info(
        "train.dataset",
        rows=len(X),
        queries=len(np.unique(qids)),
        train_rows=len(train_idx),
        val_rows=len(val_idx),
    )

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=list(FEATURE_NAMES))
    dtrain.set_group(group_sizes(qids[train_idx]))
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=list(FEATURE_NAMES))
    dval.set_group(group_sizes(qids[val_idx]))

    params = {
        "objective": "rank:ndcg",
        "eval_metric": ["ndcg@10"],
        "ndcg_exp_gain": True,  # match eval/metrics.py's exponential gain
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 1.0,
        "seed": args.seed,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping,
        verbose_eval=50,
    )

    val_ndcg = evaluate_ndcg(booster, X[val_idx], y[val_idx], qids[val_idx], chunk_ids[val_idx])

    # RRF-order baseline: score by the rrf_score feature alone. This is the
    # number the reranker has to beat to justify existing at all.
    rrf_col = FEATURE_NAMES.index("rrf_score")
    baseline = []
    by_query: dict[int, list[tuple[float, int, float]]] = defaultdict(list)
    for i in val_idx:
        by_query[int(qids[i])].append((float(X[i][rrf_col]), int(chunk_ids[i]), float(y[i])))
    for rows in by_query.values():
        rows.sort(key=lambda r: r[0], reverse=True)
        baseline.append(ndcg_at_k([c for _, c, _ in rows], {c: r for _, c, r in rows}, k=10))
    baseline_ndcg = float(np.mean(baseline)) if baseline else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.out))
    sidecar = args.out.with_suffix(args.out.suffix + ".features.json")
    sidecar.write_text(json.dumps(list(FEATURE_NAMES), indent=2))

    gains = booster.get_score(importance_type="gain")
    top = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)[:8]

    log.info(
        "train.complete",
        model=str(args.out),
        best_iteration=booster.best_iteration,
        val_ndcg_at_10=round(val_ndcg, 4),
        rrf_baseline_ndcg_at_10=round(baseline_ndcg, 4),
        lift=round(val_ndcg - baseline_ndcg, 4),
    )
    print("\nTop features by gain:")
    for name, gain in top:
        print(f"  {name:<24} {gain:10.1f}")
    print(
        f"\nnDCG@10  reranker={val_ndcg:.4f}  rrf_baseline={baseline_ndcg:.4f}  "
        f"lift={val_ndcg - baseline_ndcg:+.4f}"
    )


if __name__ == "__main__":
    main()
