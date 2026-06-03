"""Train pgmpy DiscreteBayesianNetwork for Strategy Belief classification.

Purpose: Train a Bayesian Network from telemetry API data to classify opponent
         strategy into cheese/all_in/timing_attack/macro. Falls back to rule-based
         guards when the model file is missing.

Key Decisions: Queries /api/features/match-level-full for enriched data.
               Uses strategy_category column (when populated) as ground truth.
               Falls back to cheese_type → Level-1 mapping for Zerg games.
               Derives Terran/Protoss labels from rush_detect + ARES mediator
               booleans when strategy_category is empty.

Limitations: Requires 50+ games per race for reliable training.
             Zerg labels come from cheese_type; Terran/Protoss labels are
             inferred from ARES mediator flags and game-level heuristics.

Usage:
    python scripts/train_strategy_belief.py [--api-url URL] [--output PATH]

Output:
    bot/models/strategy_belief_model.pkl
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import joblib

API_BASE = "https://telemetry.pownz.com/api"

FEATURE_COLS = [
    "game_time",
    "pool_start",
    "nat_start",
    "last_nat_scout_time",
    "nat_present_on_last_scout",
    "gas_time",
    "queen_time",
    "ling_seen",
    "ling_contact",
    "speed_start",
    "ling_has_speed",
    "gas_workers",
    "enemy_race_int",
]

STRATEGY_CATEGORY_VALUES = ["cheese", "all_in", "timing_attack", "macro"]

# cheese_type → strategy_category mapping (Zerg-ground-truth labels)
CHEESE_TYPE_TO_CATEGORY = {
    "12_pool": "cheese",
    "speedling": "cheese",
    "cannon_rush": "cheese",
    "none": "macro",
}

RACE_MAP = {"Terran": 0, "Zerg": 1, "Protoss": 2, "Random": 3}

MODEL_FILE = Path("bot/models/strategy_belief_model.pkl")


def fetch_matches(limit: int = 500, api_url: str = API_BASE) -> pd.DataFrame:
    """Fetch match-level features from the telemetry API (full endpoint)."""
    url = f"{api_url}/features/match-level-full?limit={limit}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    print(f"Fetched {len(df)} matches from /api/features/match-level-full")
    return df


def fetch_rush_events(match_id: int, api_url: str = API_BASE) -> list[dict]:
    """Fetch rush_detect events for a specific match."""
    url = f"{api_url}/matches/{match_id}/events?subsystem=rush_detect"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def derive_strategy_label(row: dict) -> str:
    """Derive strategy_category from available labels.

    Priority:
      1. API strategy_category column (when populated by Strategy Belief)
      2. cheese_type → Level-1 mapping (Zerg ground truth)
      3. Heuristic from ARES booleans + economy metrics
      4. Default: macro
    """
    # Priority 1: API-provided strategy_category
    cat = row.get("strategy_category", "")
    if cat and cat.strip():
        return cat

    # Priority 2: cheese_type → Level-1 (Zerg games with ground truth)
    cheese_type = row.get("cheese_type", "none") or "none"
    if cheese_type in CHEESE_TYPE_TO_CATEGORY:
        return CHEESE_TYPE_TO_CATEGORY[cheese_type]

    # Priority 3: Heuristic from game-level metrics
    enemy_race = row.get("enemy_race", "Unknown")

    # Zerg with rush_detected = early cheese
    if row.get("rush_detected") == 1.0:
        return "cheese"

    # Short games with high rush confidence → cheese
    avg_rush_conf = row.get("max_rush_confidence")
    if avg_rush_conf is not None and avg_rush_conf > 0.7:
        return "cheese"

    # Under attack early with no expansion → all_in
    under_attack = row.get("under_attack_count", 0)
    attack_count = row.get("attack_count", 0)
    avg_can_win_early = row.get("avg_can_win_early", 1.0)

    # Games where bot was under attack early and couldn't win → likely cheese/all_in
    if under_attack > 0 and avg_can_win_early is not None and avg_can_win_early < 0.4:
        if row.get("duration_min", 999) < 6.0:
            return "cheese"
        elif row.get("duration_min", 999) < 10.0:
            return "all_in"

    # Games with many attacks and won engagements → timing_attack
    if attack_count and attack_count >= 2 and avg_can_win_early is not None and avg_can_win_early > 0.5:
        return "timing_attack"

    # Default: macro
    return "macro"


def build_training_data(matches: pd.DataFrame, api_url: str = API_BASE) -> pd.DataFrame:
    """Build training data from match-level-full features + rush_detect events."""
    rows = []
    # Only query per-match rush_detect for Zerg/Random games (others lack timings)
    zerg_match_ids = set()

    for _, match in matches.iterrows():
        row = {
            "match_id": match.get("arena_match_id", 0),
            "enemy_race": match.get("enemy_race", "Unknown"),
            "enemy_race_int": RACE_MAP.get(match.get("enemy_race", "Unknown"), 3),
            "game_time": match.get("duration_min", 0) * 60.0,
            "cheese_type": match.get("cheese_type", "none") or "none",
            "strategy_category_api": match.get("strategy_category", ""),
            "build_label_api": match.get("build_label", ""),
        }

        # Derive ground truth label
        row["strategy_label"] = derive_strategy_label(row)

        # Game-level aggregation features (new in match-level-full)
        row["avg_army_value"] = match.get("avg_army_value", -1)
        row["avg_workers"] = match.get("avg_workers", -1)
        row["engagement_count"] = match.get("engagement_count", 0)
        row["engagements_won"] = match.get("engagements_won", 0)
        row["engagements_lost"] = match.get("engagements_lost", 0)
        row["avg_can_win"] = match.get("avg_can_win", -1)
        row["avg_can_win_early"] = match.get("avg_can_win_early", -1)
        row["under_attack_count"] = match.get("under_attack_count", 0)
        row["attack_count"] = match.get("attack_count", 0)

        # Match-level rush features
        row["rush_detected"] = 1.0 if match.get("rush_detected") else (
            0.0 if match.get("rush_detected") is False else -1
        )
        row["max_rush_confidence"] = match.get("max_rush_confidence", -1)
        row["avg_12pool_prob"] = match.get("avg_12pool_prob", -1)
        row["avg_speedling_prob"] = match.get("avg_speedling_prob", -1)

        # Collect Zerg match IDs for per-match timing queries
        if match.get("enemy_race") in ("Zerg", "Random") and match.get("rush_detected") is not None:
            zerg_match_ids.add(match.get("arena_match_id", 0))

        # Default missing timing features
        _fill_missing_timing(row)
        row["nat_start"] = -1
        row["last_nat_scout_time"] = -1
        row["nat_present_on_last_scout"] = -1

        rows.append(row)

    # Batch-fetch rush_detect events for Zerg/Random games (enriches timing features)
    fetched = 0
    for match_id in zerg_match_ids:
        if fetched >= 60:  # Rate limit: ~60 API calls
            break
        events = fetch_rush_events(match_id, api_url)
        if events:
            latest = events[-1]
            # Find the matching row
            for row in rows:
                if row["match_id"] == match_id:
                    row["pool_start"] = latest.get("pool_start", -1)
                    row["speed_start"] = latest.get("speed_start", -1)
                    row["queen_time"] = latest.get("queen_time", -1)
                    row["gas_time"] = latest.get("gas_time", -1)
                    row["ling_seen"] = latest.get("ling_seen", -1)
                    row["ling_contact"] = latest.get("ling_contact", -1)
                    row["gas_workers"] = latest.get("gas_workers", 0)
                    row["ling_has_speed"] = 1 if latest.get("ling_has_speed") else 0
                    break
        fetched += 1

    return pd.DataFrame(rows)


def _fill_missing_timing(row: dict) -> None:
    """Fill timing fields with -1 (missing)."""
    for key in [
        "pool_start", "speed_start", "queen_time", "gas_time",
        "ling_seen", "ling_contact", "gas_workers", "ling_has_speed",
    ]:
        row.setdefault(key, -1 if key != "gas_workers" else 0)
        if key == "ling_has_speed":
            row[key] = row.get(key, 0)


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize continuous features into categorical bins for pgmpy BN.

    pgmpy DiscreteBayesianNetwork requires all variables to be categorical.
    Bin continuous timing features into meaningful SC2 game-phase buckets.
    """
    df = df.copy()

    # Pool timing bins: very_early (12-pool), early (speedling), mid, late, unknown
    df["pool_bin"] = pd.cut(
        df["pool_start"].replace(-1, np.nan),
        bins=[0, 42, 52, 80, float("inf")],
        labels=["very_early", "early", "mid", "late"],
    ).fillna("unknown").astype(str)

    # Natural expansion bins
    df["nat_bin"] = pd.cut(
        df["nat_start"].replace(-1, np.nan),
        bins=[0, 80, 120, 200, float("inf")],
        labels=["very_early", "on_time", "late", "very_late"],
    ).fillna("unknown").astype(str)

    # Ling seen timing bins
    df["ling_seen_bin"] = pd.cut(
        df["ling_seen"].replace(-1, np.nan),
        bins=[0, 105, 150, 240, float("inf")],
        labels=["very_early", "early", "mid", "late"],
    ).fillna("unknown").astype(str)

    # Rush confidence bins
    df["rush_conf_bin"] = pd.cut(
        df["max_rush_confidence"].replace(-1, np.nan),
        bins=[0, 0.3, 0.7, 1.0],
        labels=["low", "medium", "high"],
    ).fillna("unknown").astype(str)

    # Game duration bins (helps distinguish cheese/all_in from macro)
    df["duration_bin"] = pd.cut(
        df["game_time"].replace(-1, np.nan),
        bins=[0, 360, 720, 1200, float("inf")],
        labels=["short", "medium", "long", "very_long"],
    ).fillna("unknown").astype(str)

    # Engagement ratio bin (won / total, helps distinguish timing_attack from macro)
    total_engagements = df["engagement_count"].replace(0, 1)
    df["win_rate_bin"] = pd.cut(
        (df["engagements_won"].fillna(0) / total_engagements).replace(-1, np.nan),
        bins=[0, 0.3, 0.6, 1.0],
        labels=["losing", "even", "winning"],
    ).fillna("unknown").astype(str)

    return df


def train_bn(df: pd.DataFrame) -> dict:
    """Train pgmpy DiscreteBayesianNetwork on discretized features.

    Returns a dict with model info. The rule-based system in StrategyBelief
    is the primary engine; the BN is supplemental.
    """
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.estimators import BayesianEstimator
    except ImportError:
        print("pgmpy not installed. Run: poetry add pgmpy")
        return {}

    # Prepare discretized data
    df_disc = discretize_features(df)

    # Filter to rows with strategy labels
    df_disc = df_disc[df_disc["strategy_label"].notna()]

    if len(df_disc) < 20:
        print(f"WARNING: Only {len(df_disc)} labeled samples. BN will be unreliable.")
        print("Recommend collecting 50+ games per race before training.")
        if len(df_disc) < 10:
            print("Falling back: model file will NOT be saved.")
            return {}

    print(f"\nTraining data: {len(df_disc)} rows")
    print(f"Label distribution:\n{df_disc['strategy_label'].value_counts()}")
    print(f"Race distribution:\n{df_disc['enemy_race'].value_counts()}")

    # Define network structure (domain knowledge)
    model = DiscreteBayesianNetwork([
        ("enemy_race", "strategy"),
        ("duration_bin", "strategy"),
        ("pool_bin", "strategy"),
        ("rush_conf_bin", "strategy"),
        ("strategy", "win_rate_bin"),
    ])

    # Fit parameters with Bayesian estimator (pseudo-counts for sparse data)
    train_cols = ["enemy_race", "duration_bin", "pool_bin", "rush_conf_bin",
                  "strategy", "win_rate_bin"]
    try:
        model.fit(df_disc[train_cols],
                  estimator=BayesianEstimator, prior_type="BDeu",
                  equivalent_sample_size=5)
    except Exception as e:
        print(f"BN fitting error: {e}")
        print("Falling back: model file will NOT be saved.")
        return {}

    # Validate model
    try:
        from pgmpy.inference import VariableElimination
        infer = VariableElimination(model)
        result = infer.query(["strategy"], evidence={"enemy_race": "Zerg"})
        print("\n=== BN Validation: P(strategy | enemy_race=Zerg) ===")
        print(result)
    except Exception as e:
        print(f"BN inference test failed: {e}")
        return {}

    return {
        "model": model,
        "categories": STRATEGY_CATEGORY_VALUES,
        "feature_cols": FEATURE_COLS,
        "discretization": "bins_defined_in_code",
        "n_samples": len(df_disc),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Strategy Belief BN model")
    parser.add_argument("--api-url", default=API_BASE, help="Telemetry API base URL")
    parser.add_argument("--output", default=str(MODEL_FILE), help="Output model path")
    parser.add_argument("--limit", type=int, default=500, help="Max matches to fetch")
    args = parser.parse_args()

    print("=== Strategy Belief Model Training ===\n")

    # Fetch data
    matches = fetch_matches(limit=args.limit, api_url=args.api_url)

    # Build training data
    print("\nBuilding training data...")
    df = build_training_data(matches, api_url=args.api_url)

    print(f"\nLabel distribution:")
    print(df["strategy_label"].value_counts())
    print(f"\nRace distribution:")
    print(df["enemy_race"].value_counts())

    # Check how many labels came from each source
    api_labeled = sum(1 for _, r in df.iterrows() if r.get("strategy_category_api", "").strip())
    cheese_labeled = sum(1 for _, r in df.iterrows() if r.get("cheese_type", "none") != "none")
    heuristic_labeled = len(df) - api_labeled - cheese_labeled
    print(f"\nLabel sources: API={api_labeled}, cheese_type={cheese_labeled}, heuristic={heuristic_labeled}")

    # Train BN
    print("\nTraining Bayesian Network...")
    result = train_bn(df)

    if result and "model" in result:
        MODEL_FILE.parent.mkdir(exist_ok=True)
        joblib.dump(result, args.output)
        print(f"\n=== Model saved to {args.output} ===")
        print(f"Categories: {result['categories']}")
        print(f"Training samples: {result['n_samples']}")
    else:
        print("\n=== BN model NOT saved — rule-based guards will be used ===")
        print("Collect more data (50+ games per race) and re-run this script.")


if __name__ == "__main__":
    main()