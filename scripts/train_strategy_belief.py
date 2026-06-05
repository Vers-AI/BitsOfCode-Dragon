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
    """Fetch match records from the telemetry API.

    Tries /api/features/match-level-full first (enriched endpoint).
    Falls back to /api/matches (basic endpoint) if the full one fails.
    """
    # Try enriched endpoint first
    url_full = f"{api_url}/features/match-level-full?limit={limit}"
    try:
        resp = requests.get(url_full, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        print(f"Fetched {len(df)} matches from /api/features/match-level-full")
        return df
    except requests.HTTPError:
        print(f"/api/features/match-level-full unavailable, falling back to /api/matches")

    # Fallback: basic matches endpoint
    url = f"{api_url}/matches?limit={limit}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    print(f"Fetched {len(df)} matches from /api/matches")
    return df


def fetch_match_events(match_id: int, api_url: str = API_BASE) -> list[dict]:
    """Fetch all events for a specific match.

    The first event in the response typically contains rush_detect timing
    features (pool_start, gas_time, ling_seen, etc.) even for non-Zerg games.
    """
    url = f"{api_url}/matches/{match_id}/events"
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
    game_time = row.get("game_time", 0)  # seconds

    # used_cheese_response = bot detected cheese → strong signal
    if row.get("used_cheese_response"):
        # Distinguish cheese from all_in by game length
        if game_time < 360:  # < 6 min
            return "cheese"
        elif game_time < 600:  # < 10 min
            return "all_in"

    # rush_time_seconds present → cheese detected by rush detector
    rush_time = row.get("rush_time_seconds", -1)
    if rush_time is not None and rush_time > 0:
        return "cheese"

    # Zerg with rush_detected = early cheese
    if row.get("rush_detected") == 1.0:
        return "cheese"

    # Short games with high rush confidence → cheese
    avg_rush_conf = row.get("max_rush_confidence")
    if avg_rush_conf is not None and avg_rush_conf > 0.7:
        return "cheese"

    # Under attack early with no expansion → all_in
    under_attack = row.get("under_attack_count", 0)

    # Games where bot was under attack early → likely cheese/all_in
    if under_attack > 0:
        if game_time < 360:  # < 6 min
            return "cheese"
        elif game_time < 600:  # < 10 min
            return "all_in"

    # Short game losses (opponent won fast) → likely cheese or all_in
    result = row.get("result", "")
    if result == "loss" and game_time < 360:
        return "cheese"
    if result == "loss" and game_time < 600:
        return "all_in"

    # Default: macro
    return "macro"


def _extract_timing_from_events(events: list[dict]) -> dict:
    """Extract timing features from per-match events.

    The first event in the response often contains rush_detect timing fields
    (pool_start, gas_time, ling_seen, etc.) even for non-Zerg games.
    Later events may have updated values — take the latest non-null value.
    """
    timing = {}
    timing_fields = [
        "pool_start", "speed_start", "queen_time", "gas_time",
        "ling_seen", "ling_contact", "gas_workers", "ling_has_speed",
        "last_nat_scout_time", "nat_present_on_last_scout",
    ]
    # Also extract rush detection fields
    rush_fields = ["score_12p", "score_speed", "auto_true_fired", "cheese_label"]

    for event in events:
        for field in timing_fields + rush_fields:
            val = event.get(field)
            if val is not None and val != "" and val != -1.0:
                timing[field] = val

    # Derive nat_start from events that have it
    for event in events:
        nat = event.get("nat_start")
        if nat is not None and nat != "" and nat > 0:
            timing["nat_start"] = nat

    return timing


def build_training_data(matches: pd.DataFrame, api_url: str = API_BASE) -> pd.DataFrame:
    """Build training data from match records + per-match events.

    Uses /api/matches for basic match info, then enriches each match with
    timing features from /api/matches/{id}/events.
    """
    rows = []
    fetched_events = 0
    max_event_fetches = min(len(matches), 200)  # Rate limit

    for idx, (_, match) in enumerate(matches.iterrows()):
        match_id = match.get("arena_match_id", 0)
        # game_steps → seconds: SC2 runs at 16 steps/sec on fast speed
        game_steps = match.get("game_steps", 0)
        game_time = game_steps / 16.0 if game_steps else 0.0

        row = {
            "match_id": match_id,
            "enemy_race": match.get("enemy_race", "Unknown"),
            "enemy_race_int": RACE_MAP.get(match.get("enemy_race", "Unknown"), 3),
            "game_time": game_time,
            "cheese_type": match.get("cheese_type", "none") or "none",
            "strategy_category_api": match.get("strategy_category", ""),
            "build_label_api": match.get("build_label", ""),
            # Match-level fields available from /api/matches
            "used_cheese_response": match.get("used_cheese_response", False),
            "rush_time_seconds": match.get("rush_time_seconds", -1),
            "sq": match.get("sq", -1),
        }

        # Derive ground truth label
        row["strategy_label"] = derive_strategy_label(row)

        # Default missing timing features
        _fill_missing_timing(row)
        row["nat_start"] = -1
        row["last_nat_scout_time"] = -1
        row["nat_present_on_last_scout"] = -1

        # Default engagement/rush features (not available from /api/matches)
        row["engagement_count"] = 0
        row["engagements_won"] = 0
        row["engagements_lost"] = 0
        row["avg_can_win"] = -1
        row["avg_can_win_early"] = -1
        row["under_attack_count"] = 0
        row["attack_count"] = 0
        row["rush_detected"] = 1.0 if match.get("rush_detected") else (
            0.0 if match.get("rush_detected") is False else -1
        )
        row["max_rush_confidence"] = -1
        row["avg_12pool_prob"] = -1
        row["avg_speedling_prob"] = -1

        # Enrich with per-match events (timing features)
        if fetched_events < max_event_fetches:
            events = fetch_match_events(match_id, api_url)
            fetched_events += 1
            if events:
                timing = _extract_timing_from_events(events)
                # Merge timing into row (only overwrite defaults)
                for field in ["pool_start", "speed_start", "queen_time", "gas_time",
                              "ling_seen", "ling_contact", "gas_workers",
                              "ling_has_speed", "nat_start", "last_nat_scout_time",
                              "nat_present_on_last_scout"]:
                    if field in timing:
                        row[field] = timing[field]
                # Rush detection scores
                if "score_12p" in timing:
                    row["max_rush_confidence"] = max(
                        float(timing.get("score_12p", -1)),
                        float(timing.get("score_speed", -1)),
                    )
                # Count under_attack transitions from events
                under_count = sum(
                    1 for e in events
                    if e.get("under_attack") is True
                )
                row["under_attack_count"] = under_count

            if fetched_events % 50 == 0:
                print(f"  Fetched events for {fetched_events}/{max_event_fetches} matches...")

        rows.append(row)

    print(f"Enriched {fetched_events} matches with per-match event data")
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
    ).astype(str).replace("NaN", "unknown")

    # Natural expansion bins
    df["nat_bin"] = pd.cut(
        df["nat_start"].replace(-1, np.nan),
        bins=[0, 80, 120, 200, float("inf")],
        labels=["very_early", "on_time", "late", "very_late"],
    ).astype(str).replace("NaN", "unknown")

    # Ling seen timing bins
    df["ling_seen_bin"] = pd.cut(
        df["ling_seen"].replace(-1, np.nan),
        bins=[0, 105, 150, 240, float("inf")],
        labels=["very_early", "early", "mid", "late"],
    ).astype(str).replace("NaN", "unknown")

    # Rush confidence bins
    df["rush_conf_bin"] = pd.cut(
        df["max_rush_confidence"].replace(-1, np.nan),
        bins=[0, 0.3, 0.7, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str).replace("NaN", "unknown")

    # Game duration bins (helps distinguish cheese/all_in from macro)
    df["duration_bin"] = pd.cut(
        df["game_time"].replace(-1, np.nan),
        bins=[0, 360, 720, 1200, float("inf")],
        labels=["short", "medium", "long", "very_long"],
    ).astype(str).replace("NaN", "unknown")

    # Engagement ratio bin (won / total, helps distinguish timing_attack from macro)
    total_engagements = df["engagement_count"].replace(0, 1)
    df["win_rate_bin"] = pd.cut(
        (df["engagements_won"].fillna(0) / total_engagements).replace(-1, np.nan),
        bins=[0, 0.3, 0.6, 1.0],
        labels=["losing", "even", "winning"],
    ).astype(str).replace("NaN", "unknown")

    return df


def train_bn(df: pd.DataFrame) -> dict:
    """Train pgmpy DiscreteBayesianNetwork on discretized features.

    Returns a dict with model info. The rule-based system in StrategyBelief
    is the primary engine; the BN is supplemental.
    """
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.parameter_estimator import DiscreteMLE
    except ImportError:
        print("pgmpy not installed. Run: poetry add pgmpy")
        return {}

    # Prepare discretized data
    df_disc = discretize_features(df)

    # Rename strategy_label → strategy for BN node naming
    df_disc = df_disc.rename(columns={"strategy_label": "strategy"})

    # Filter to rows with strategy labels
    df_disc = df_disc[df_disc["strategy"].notna()]

    if len(df_disc) < 20:
        print(f"WARNING: Only {len(df_disc)} labeled samples. BN will be unreliable.")
        print("Recommend collecting 50+ games per race before training.")
        if len(df_disc) < 10:
            print("Falling back: model file will NOT be saved.")
            return {}

    print(f"\nTraining data: {len(df_disc)} rows")
    print(f"Label distribution:\n{df_disc['strategy'].value_counts()}")
    print(f"Race distribution:\n{df_disc['enemy_race'].value_counts()}")

    # Define network structure (domain knowledge)
    # Only use features with reasonable data coverage:
    #   enemy_race: 100% coverage
    #   duration_bin: 100% coverage (derived from game_steps)
    #   pool_bin: ~15% coverage (Zerg games only, but strong signal)
    # rush_conf_bin and win_rate_bin have too many "unknown" values
    # from /api/matches, causing CPD issues. Drop them for now.
    model = DiscreteBayesianNetwork([
        ("enemy_race", "strategy"),
        ("duration_bin", "strategy"),
        ("pool_bin", "strategy"),
    ])

    # Fit parameters with MLE estimator
    train_cols = ["enemy_race", "duration_bin", "pool_bin", "strategy"]
    try:
        estimator = DiscreteMLE()
        model.fit(df_disc[train_cols], estimator=estimator)
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

    # Show timing feature coverage
    for field in ["pool_start", "gas_time", "ling_seen", "nat_start"]:
        known = sum(1 for _, r in df.iterrows() if r.get(field, -1) != -1)
        print(f"  {field}: {known}/{len(df)} rows have data")

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