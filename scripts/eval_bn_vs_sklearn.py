"""Phase B evaluation harness: BN vs sklearn on a held-out split.

Purpose: Train the naive Bayes BN (retrained on the 80% train fold) and a
         sklearn GradientBoostingClassifier on identical folds, evaluate both
         on the same 20% holdout, and also score the *deployed* BN as the
         baseline. One run produces the Phase B gate decision.

Key Decisions: Standalone script — imports the training pipeline functions
               from train_strategy_belief.py instead of duplicating them.
               Same 15 discretized features for both models (fair comparison);
               sklearn gets them one-hot encoded.

Limitations: Holdout metrics are in-sample-family estimates on ~1,300 rows;
             deployed-BN baseline is evaluated on the same window (it never
             saw ANY of these rows at training time, so its score is honest).

Usage:
    python scripts/eval_bn_vs_sklearn.py [--seed 42]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.train_strategy_belief as tsb
from scripts.train_strategy_belief import (
    STRATEGY_CATEGORY_VALUES,
    build_training_data,
    discretize_features,
    fetch_matches,
    filter_ground_truth_only,
)

# Single source of truth — the training script owns the feature list
EVIDENCE_COLS = tsb.SKLEARN_EVIDENCE_COLS

CACHE_PATH = Path("data/eval_cache_training_data.pkl.gz")


def build_evidence_matrix(df_disc: pd.DataFrame) -> pd.DataFrame:
    """Extract the 15 evidence columns as a clean string DataFrame."""
    X = df_disc[EVIDENCE_COLS].copy()
    for col in EVIDENCE_COLS:
        X[col] = X[col].astype(str)
    return X


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, per-class recall, and macro/aggressive split."""
    labels = STRATEGY_CATEGORY_VALUES
    acc = float(np.mean(y_true == y_pred))
    report = {"name": name, "accuracy": acc, "n": len(y_true)}
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            report[f"recall_{label}"] = float(np.mean(y_pred[mask] == label))
            report[f"support_{label}"] = int(mask.sum())
    # Macro-vs-aggressive: is the bot at least right about "is this macro?"
    true_macro = y_true == "macro"
    pred_macro = y_pred == "macro"
    report["macro_split_acc"] = float(np.mean(true_macro == pred_macro))
    # False positives: bot says aggressive, replay says macro
    report["fp_aggressive_vs_macro"] = int(np.sum(~pred_macro & true_macro))
    # False negatives: bot says macro, replay says aggressive
    report["fn_macro_vs_aggressive"] = int(np.sum(pred_macro & ~true_macro))
    return report


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = STRATEGY_CATEGORY_VALUES
    cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(y_true, y_pred):
        cm.loc[t, p] += 1
    cm.index.name = "true\\pred"
    return cm


EVENTS_CACHE_PATH = Path("data/events_cache.pkl")


def prefetch_events_parallel(match_ids: list[int], api_url: str, workers: int = 12) -> dict[int, list]:
    """Fetch per-match events in parallel — the endpoint is ~2.5s per request
    server-side, so parallelism is the only way to cover 1300+ matches.

    Progress persists to data/events_cache.pkl — reruns skip already-fetched
    match ids, so interrupted runs resume instead of starting over.

    Returns {match_id: events_list}. Failed fetches map to empty lists
    (same behavior as the serial fetch_match_events fallback).
    """
    import pickle

    cache: dict[int, list] = {}
    if EVENTS_CACHE_PATH.exists():
        with open(EVENTS_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"    prefetch: {len(cache)} events already cached")

    todo = [mid for mid in match_ids if mid not in cache]
    print(f"    prefetch: {len(todo)} to fetch")
    if not todo:
        return cache

    def _get(mid: int) -> tuple[int, list]:
        try:
            resp = requests.get(f"{api_url}/matches/{mid}/events", timeout=30)
            resp.raise_for_status()
            return mid, resp.json()
        except Exception:
            return mid, []

    fetched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for mid, events in pool.map(_get, todo):
            cache[mid] = events
            fetched += 1
            if fetched % 200 == 0:
                with open(EVENTS_CACHE_PATH, "wb") as f:
                    pickle.dump(cache, f)
                print(f"    prefetched {fetched}/{len(todo)} (checkpoint saved)")

    with open(EVENTS_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"    prefetch complete: {len(cache)} total in cache")
    return cache


def main():
    parser = argparse.ArgumentParser(description="Phase B gate: BN vs sklearn holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Re-fetch and re-build the cached training data")
    args = parser.parse_args()

    print("=== Phase B Evaluation: BN vs sklearn (hard cutover gate) ===\n")

    # 1. Fetch + build data using the existing pipeline (cached between runs —
    #    event enrichment is slow, ~2.5s per request server-side)
    if CACHE_PATH.exists() and not args.refresh_cache:
        print(f"Loading cached training data from {CACHE_PATH}")
        df = pd.read_pickle(CACHE_PATH)
    else:
        matches = fetch_matches(limit=args.limit)
        # Only labeled matches matter — filter_ground_truth_only drops the rest,
        # so fetching their events is pure waste (~33% of requests saved)
        labeled = matches[matches["strategy_category"].notna() &
                          (matches["strategy_category"] != "")].copy()
        print(f"{len(labeled)}/{len(matches)} matches have ground-truth labels — "
              f"prefetching events for those only...")
        match_ids = [int(m) for m in labeled["arena_match_id"].tolist()]
        events_cache = prefetch_events_parallel(match_ids, tsb.API_BASE)
        tsb.fetch_match_events = lambda mid, api_url=tsb.API_BASE: events_cache.get(mid, [])
        df = build_training_data(labeled)
        CACHE_PATH.parent.mkdir(exist_ok=True)
        df.to_pickle(CACHE_PATH)
        print(f"Cached training data to {CACHE_PATH}")
    df = filter_ground_truth_only(df)
    if len(df) < 100:
        print(f"FATAL: only {len(df)} ground-truth rows — aborting.")
        return 1

    # 2. Discretize + build matrices
    df_disc = discretize_features(df)
    df_disc = df_disc.rename(columns={"strategy_label": "strategy"})
    df_disc = df_disc[df_disc["strategy"].notna()].copy()

    X_all = build_evidence_matrix(df_disc)
    y_all = df_disc["strategy"].values

    print(f"\nDataset: {len(y_all)} labeled rows")
    print(f"Class distribution:\n{pd.Series(y_all).value_counts()}\n")

    # 3. Stratified 80/20 split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=args.seed
    )
    print(f"Train fold: {len(y_train)} | Holdout: {len(y_test)}\n")

    results = []
    cms = {}

    # 4. Baseline: the DEPLOYED BN (bot/models .npz — never trained on these rows).
    #    Loads the .npz directly — importing bot.belief.bn_inference pulls the
    #    full bot package (ares, cython_extensions) which isn't available here.
    try:
        npz_path = Path("bot/models/strategy_belief_model.npz")
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=False)
            cpd = data["cpd"]
            node_order = [str(v) for v in data["node_order"].tolist()]
            parent_vars = node_order[1:]  # first axis is strategy
            states = {v: [str(s) for s in data[f"states_{v}"].tolist()] for v in node_order}
            strategy_states = states["strategy"]

            def bn_predict(row: dict) -> str:
                indices = []
                for var in parent_vars:
                    val = str(row.get(var, ""))
                    var_states = states.get(var, [])
                    if val in var_states:
                        indices.append(var_states.index(val))
                    elif var_states:
                        indices.append(0)  # same fallback as BNInference
                    else:
                        return "macro"
                probs = cpd[tuple([slice(None)] + indices)]
                best = int(np.argmax(probs))
                return strategy_states[best] if best < len(strategy_states) else "macro"

            preds = np.array([bn_predict(row.to_dict()) for _, row in X_test.iterrows()])
            results.append(evaluate("DEPLOYED BN (v0.12.4)", y_test, preds))
            cms["DEPLOYED BN (v0.12.4)"] = confusion_matrix(y_test, preds)
        else:
            print("[harness] No deployed .npz model found — skipping deployed BN baseline\n")
    except Exception as e:
        print(f"[harness] Deployed BN eval failed: {e}\n")

    # 5. Retrained BN on the train fold (pgmpy, dev-only)
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.parameter_estimator import DiscreteMLE

        bn_cols = EVIDENCE_COLS + ["strategy"]
        model = DiscreteBayesianNetwork([
            ("enemy_race", "strategy"), ("duration_bin", "strategy"),
            ("pool_bin", "strategy"), ("rax_bin", "strategy"),
            ("gateway_bin", "strategy"), ("bases_bin", "strategy"),
            ("factory_bin", "strategy"),
            ("rax_near_base", "strategy"), ("gw_near_base", "strategy"),
            ("cannon_near_base", "strategy"), ("bunker_near_base", "strategy"),
            ("rax_timing", "strategy"), ("pool_timing", "strategy"),
            ("gw_timing", "strategy"), ("nat_timing", "strategy"),
        ])
        model.fit(X_train.assign(strategy=y_train)[bn_cols],
                  estimator=DiscreteMLE())

        from pgmpy.inference import VariableElimination
        infer = VariableElimination(model)
        preds = []
        for _, row in X_test.iterrows():
            # predict returns dict[StrategyCategory, float] — take argmax by value
            try:
                probs = infer.query(["strategy"], evidence=row.to_dict())
                # values are in the model's state order; get state names
                states = probs.state_names["strategy"] if hasattr(probs, "state_names") else model.states["strategy"]
                idx = int(np.argmax(probs.values))
                preds.append(states[idx])
            except Exception:
                preds.append("macro")  # unobserved evidence combo — safe default
        preds = np.array([p if p in STRATEGY_CATEGORY_VALUES else "macro" for p in preds])
        results.append(evaluate("RETRAINED BN (train fold)", y_test, preds))
        cms["RETRAINED BN (train fold)"] = confusion_matrix(y_test, preds)
    except ImportError:
        print("[harness] pgmpy not installed — skipping retrained BN baseline\n")
    except Exception as e:
        print(f"[harness] Retrained BN eval failed: {e}\n")

    # 6. sklearn GradientBoostingClassifier on the train fold
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    skl = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("clf", GradientBoostingClassifier(random_state=args.seed)),
    ])
    skl.fit(X_train, y_train)
    preds = skl.predict(X_test)
    results.append(evaluate("SKLEARN GBC (train fold)", y_test, preds))
    cms["SKLEARN GBC (train fold)"] = confusion_matrix(y_test, preds)

    # 7. Report
    print("\n" + "=" * 70)
    print("HOLDOUT RESULTS (20% stratified split, seed", args.seed, ")")
    print("=" * 70)
    for r in results:
        print(f"\n  {r['name']}  (n={r['n']})")
        print(f"    accuracy:        {r['accuracy']*100:.1f}%")
        print(f"    macro split acc: {r['macro_split_acc']*100:.1f}%")
        print(f"    FP aggressive->macro: {r['fp_aggressive_vs_macro']} | "
              f"FN macro->aggressive: {r['fn_macro_vs_aggressive']}")
        for label in STRATEGY_CATEGORY_VALUES:
            if f"recall_{label}" in r:
                print(f"    recall {label:14s}: {r[f'recall_{label}']*100:5.1f}%  (n={r[f'support_{label}']})")
    print("\n" + "=" * 70)
    print("CONFUSION MATRICES (rows = true, cols = predicted)")
    print("=" * 70)
    for name, cm in cms.items():
        print(f"\n  --- {name} ---")
        print(cm.to_string())

    # 8. Gate decision
    if len(results) >= 2:
        skl_r = next((r for r in results if "SKLEARN" in r["name"]), None)
        bn_r = next((r for r in results if "BN" in r["name"]), None)
        if skl_r and bn_r:
            delta = (skl_r["accuracy"] - bn_r["accuracy"]) * 100
            print("\n" + "=" * 70)
            print("GATE: sklearn holdout >= 55% AND >= +8 pts over best BN")
            print(f"  sklearn: {skl_r['accuracy']*100:.1f}% | best BN: {bn_r['accuracy']*100:.1f}% | delta: {delta:+.1f} pts")
            if skl_r["accuracy"] >= 0.55 and delta >= 8:
                print("  >> GATE PASSED - proceed with hard cutover")
            elif skl_r["accuracy"] >= 0.55:
                print("  >> GATE FAILED (sklearn ok but margin < +8) - keep BN, retrain full window")
            else:
                print("  >> GATE FAILED (sklearn < 55%) - keep BN, retrain full window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())