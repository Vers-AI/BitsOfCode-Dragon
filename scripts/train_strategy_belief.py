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
import time
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

# Structure type name → normalized name mapping (from API scouted_enemy_structures)
STRUCT_ALIASES = {
    "BARRACKS": "barracks", "BARRACKSREACTOR": "barracks", "BARRACKSTECHLAB": "barracks",
    "FACTORY": "factory", "FACTORYREACTOR": "factory", "FACTORYTECHLAB": "factory",
    "STARPORT": "starport", "STARPORTREACTOR": "starport", "STARPORTTECHLAB": "starport",
    "GATEWAY": "gateway", "WARPGATE": "gateway",
    "FORGE": "forge", "CYBERNETICSCORE": "cyber_core",
    "PHOTONCANNON": "cannon", "SHIELDBATTERY": "shield_battery",
    "STARGATE": "stargate", "FLEETBEACON": "fleet_beacon",
    "ROBOTICSFACILITY": "robotics_facility", "ROBOTICSBAY": "robotics_bay",
    "TWILIGHTCOUNCIL": "twilight_council", "TEMPLARARCHIVE": "templar_archive",
    "DARKSHRINE": "dark_shrine",
    "SPAWNINGPOOL": "spawning_pool", "EXTRACTOR": "extractor",
    "BANELINGNEST": "baneling_nest", "ROACHWARREN": "roach_warren",
    "HYDRALISKDEN": "hydralisk_den", "SPIRE": "spire",
    "INFESTATIONPIT": "infestation_pit", "ULTRALISKCAVERN": "ultralisk_cavern",
    "HATCHERY": "hatchery", "LAIR": "lair", "HIVE": "hive",
    "NEXUS": "nexus", "PYLON": "pylon",
    "COMMANDCENTER": "command_center", "ORBITALCOMMAND": "orbital_command",
    "PLANETARYFORTRESS": "planetary_fortress",
    "SUPPLYDEPOT": "supply_depot", "SUPPLYDEPOTLOWERED": "supply_depot",
    "REFINERY": "refinery", "ASSIMILATOR": "assimilator",
    "BUNKER": "bunker", "ENGINEERINGBAY": "engineering_bay",
    "MISSILETURRET": "missile_turret", "SENSORTOWER": "sensor_tower",
    "GHOSTACADEMY": "ghost_academy", "FUSIONCORE": "fusion_core",
    "CREEPTUMORBURROWED": "creep_tumor",
}

# Unit type name → normalized name mapping (from API scouted_enemy_units)
UNIT_ALIASES = {
    "MARINE": "marine", "MARAUDER": "marauder", "REAPER": "reaper",
    "GHOST": "ghost", "HELLION": "hellion", "SIEGETANK": "siege_tank",
    "CYCLONE": "cyclone", "THOR": "thor", "BATTLECRUISER": "battlecruiser",
    "MEDIVAC": "medivac", "RAVEN": "raven", "BANSHEE": "banshee",
    "VIKING": "viking", "WIDOWMINE": "widow_mine",
    "ZEALOT": "zealot", "STALKER": "stalker", "SENTRY": "sentry",
    "ADEPT": "adept", "HIGHTEMPLAR": "high_templar", "DARKTEMPLAR": "dark_templar",
    "IMMORTAL": "immortal", "COLOSSUS": "colossus", "DISRUPTOR": "disruptor",
    "OBSERVER": "observer", "WARPPRISM": "warp_prism",
    "PHOENIX": "phoenix", "VOIDRAY": "void_ray", "CARRIER": "carrier",
    "ORACLE": "oracle", "TEMPEST": "tempest", "MOTHERSHIP": "mothership",
    "ZERGLING": "zergling", "DRONE": "drone", "QUEEN": "queen",
    "HYDRALISK": "hydralisk", "MUTALISK": "mutalisk", "CORRUPTOR": "corruptor",
    "BROODLORD": "broodlord", "INFESTOR": "infestor", "SWARMHOST": "swarm_host",
    "ULTRALISK": "ultralisk", "ROACH": "roach", "RAVAGER": "ravager",
    "BANELING": "baneling", "OVERLORD": "overlord", "OVERSEER": "overseer",
    "SCV": "scv", "PROBE": "probe", "MULE": "mule",
}

RACE_MAP = {"Terran": 0, "Zerg": 1, "Protoss": 2, "Random": 3}

MODEL_FILE = Path("bot/models/strategy_belief_model.pkl")
NPZ_FILE = Path("bot/models/strategy_belief_model.npz")
OPPONENT_PRIORS_FILE = Path("bot/models/opponent_priors.json")
CATEGORY_PRIOR_FILE = Path("bot/models/strategy_category_prior.json")

# Minimum games per opponent to include in priors (avoids overfitting to 1-game samples)
MIN_GAMES_PER_OPPONENT = 3

# When True, only rows with API-provided strategy_category (replay ground truth)
# are used for training. Heuristic-derived labels are dropped entirely, breaking
# the self-referential loop where the BN learns from its own rule-based outputs.
GROUND_TRUTH_ONLY_DEFAULT = True

# Self-correction: when the previous BN model disagrees with the ground-truth
# label on a training row, and a deterministic timing guard also agrees with
# the ground truth (not the BN), the row's label is trusted as-is and the BN
# is forced to learn from it. This makes each retraining cycle correct the
# CPD entries that were wrong last time. Stats are printed so you can see how
# many rows the previous BN got wrong.
SELF_CORRECT_DEFAULT = True


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
    """Derive strategy_category from available labels + structure/unit heuristics.

    Priority:
      1. cheese_type → Level-1 mapping (Zerg ground truth, cannon_rush)
      2. Structure/unit count heuristics (all races)
      3. Game-level metrics (used_cheese_response, under_attack, game length)
      4. Default: macro

    Key rule: always check _last_scout_time. If the bot hasn't scouted since
    before the threshold, "haven't seen it" doesn't mean "it's not there."
    For training data we use struct_counts as a proxy for scout coverage.

    IMPORTANT: game_time is the TOTAL game duration (always > 360s in our data).
    Time-based heuristics use struct_first_seen (when structures were first
    observed) and first_under_attack_time instead.
    """
    # Priority 0: API strategy_category (ground truth from enriched endpoint)
    raw_api = row.get("strategy_category_api")
    if raw_api is None or (isinstance(raw_api, float) and pd.isna(raw_api)):
        raw_api = ""
    api_category = str(raw_api).strip().lower()
    if api_category == "timing":
        api_category = "timing_attack"
    if api_category in ("cheese", "all_in", "timing_attack", "macro"):
        return api_category

    # Priority 1: cheese_type → Level-1 (Zerg games with ground truth)
    # Only use cheese_type mapping for known cheese types (not "none")
    cheese_type = row.get("cheese_type", "none") or "none"
    if cheese_type != "none" and cheese_type in CHEESE_TYPE_TO_CATEGORY:
        return CHEESE_TYPE_TO_CATEGORY[cheese_type]

    # Extract structure/unit counts (from events)
    structs = row.get("struct_counts", {}) or {}
    units = row.get("unit_counts", {}) or {}
    struct_first_seen = row.get("struct_first_seen", {}) or {}
    unit_first_seen = row.get("unit_first_seen", {}) or {}
    game_time = row.get("game_time", 0)
    enemy_race = row.get("enemy_race", "Unknown")
    first_under_attack = row.get("first_under_attack_time")

    # Helper: get structure count
    def sc(name: str) -> int:
        return structs.get(name, 0)

    # Helper: get unit count
    def uc(name: str) -> int:
        return units.get(name, 0)

    # Helper: when was a structure first seen (seconds)?
    def sf(name: str) -> float | None:
        return struct_first_seen.get(name)

    # Helper: when was a unit first seen (seconds)?
    def uf(name: str) -> float | None:
        return unit_first_seen.get(name)

    # Helper: count bases (townhalls)
    def base_count() -> int:
        return max(
            sc("hatchery") + sc("lair") + sc("hive"),
            sc("nexus"),
            sc("command_center") + sc("orbital_command") + sc("planetary_fortress"),
        )

    # Helper: count gateways + warpgates
    def gateway_total() -> int:
        return sc("gateway") + sc("warpgate")

    # ── CHEESE ──────────────────────────────────────────────────────

    # worker_rush: workers approaching + no production structures
    if row.get("worker_rush_active"):
        return "cheese"

    # 12_pool: pool_seen_time < 35s
    pool_start = row.get("pool_start", -1)
    if pool_start is not None and pool_start > 0 and pool_start < 35:
        return "cheese"

    # cannon_rush: forge seen + cannon present + no gateway/cyber_core
    # (Training data: forge present + cannon present + no gateway/cyber_core)
    if sc("forge") > 0 and sc("cannon") > 0 and sc("cyber_core") == 0:
        return "cheese"

    # proxy_rax: barracks seen very early + 1 base
    # Standard 1-rax FE hits at ~52s; proxy rax is typically <30s (worker cut).
    # Use 45s as the threshold — anything later is likely standard play.
    # When position data is available (rax_near_base), only flag as cheese
    # when rax_near_base == "yes". Until then, use the tighter timing threshold.
    rax_time = sf("barracks")
    rax_near = row.get("rax_near_base", "unknown")
    if enemy_race == "Terran" and sc("barracks") >= 1 and base_count() == 1:
        if rax_near == "yes":
            return "cheese"
        if rax_near == "unknown" and rax_time is not None and rax_time < 45:
            return "cheese"
        # rax_near == "no" or rax_time >= 45 → not proxy, fall through

    # proxy_gateway: gateway seen very early + 1 base
    # Standard gateway is ~16 supply (~65s). Proxy gateway is <20s.
    gw_time = sf("gateway")
    gw_near = row.get("gw_near_base", "unknown")
    if enemy_race == "Protoss" and sc("gateway") >= 1 and sc("nexus") <= 1:
        if gw_near == "yes":
            return "cheese"
        if gw_near == "unknown" and gw_time is not None and gw_time < 30:
            return "cheese"
        # gw_near == "no" or gw_time >= 30 → not proxy, fall through

    # ── ALL-IN ──────────────────────────────────────────────────────

    # marine_rush: ≥2 barracks + 1 base + early
    if sc("barracks") >= 2 and base_count() == 1 and rax_time is not None and rax_time < 300:
        return "all_in"

    # four_gate: ≥4 gateways + 1 base
    if gateway_total() >= 4 and base_count() == 1:
        return "all_in"

    # roach_rush: roach warren early + 1 base
    rw_time = sf("roach_warren")
    if sc("roach_warren") > 0 and base_count() == 1 and rw_time is not None and rw_time < 300:
        return "all_in"

    # ravager_rush: ravagers seen + 1 base
    rav_time = uf("ravager")
    if uc("ravager") > 0 and base_count() == 1 and rav_time is not None and rav_time < 420:
        return "all_in"

    # six_gate: ≥6 gateways + 2 bases
    if gateway_total() >= 6 and base_count() == 2:
        return "all_in"

    # two_base_colossus: robotics bay + 2 bases + early
    rb_time = sf("robotics_bay")
    if sc("robotics_bay") > 0 and base_count() == 2 and rb_time is not None and rb_time < 480:
        return "all_in"

    # one_base_all_in: 1 base + game > 4min + no natural
    if base_count() == 1 and game_time > 240 and sc("nexus") <= 1 and sc("hatchery") <= 1 and sc("orbital_command") <= 1:
        return "all_in"

    # two_base_all_in: 2 bases + lots of production + short game
    if base_count() == 2 and game_time < 600 and (
        sc("barracks") >= 4 or gateway_total() >= 4
    ):
        return "all_in"

    # ── TIMING ATTACK ────────────────────────────────────────────────

    # stargate_timing: stargate early
    sg_time = sf("stargate")
    if sc("stargate") > 0 and sg_time is not None and sg_time < 270:
        return "timing_attack"

    # bio_timing: ≥3 barracks + medivacs
    if sc("barracks") >= 3 and uc("medivac") > 0:
        return "timing_attack"

    # roach_timing: roach warren 150-270s + ≥2 bases
    if sc("roach_warren") > 0 and base_count() >= 2 and rw_time is not None and 150 <= rw_time <= 270:
        return "timing_attack"

    # tank_timing: factory + siege tanks
    if sc("factory") > 0 and uc("siege_tank") > 0:
        return "timing_attack"

    # widow_mine_drop: starport + widow mines
    if sc("starport") > 0 and uc("widow_mine") > 0:
        return "timing_attack"

    # ── GAME-LEVEL HEURISTICS ────────────────────────────────────────

    # used_cheese_response = bot detected cheese → strong signal
    if row.get("used_cheese_response"):
        if first_under_attack is not None and first_under_attack < 360:
            return "cheese"
        elif first_under_attack is not None and first_under_attack < 600:
            return "all_in"
        elif game_time < 600:
            return "cheese"

    # Under attack early → likely cheese/all_in
    if first_under_attack is not None:
        if first_under_attack < 360:
            return "cheese"
        elif first_under_attack < 600:
            return "all_in"

    # Short game losses → likely cheese or all_in
    result = row.get("result", "")
    if result == "loss" and game_time < 480:
        return "cheese"
    if result == "loss" and game_time < 720:
        return "all_in"

    # ── MACRO ────────────────────────────────────────────────────────

    # mech: factory ≥ barracks
    if sc("factory") >= sc("barracks") and sc("factory") > 0 and sc("barracks") > 0:
        return "macro"

    # three_base_macro: ≥3 bases
    if base_count() >= 3:
        return "macro"

    # bio_macro: ≥3 barracks + medivacs + ≥3 bases
    if sc("barracks") >= 3 and uc("medivac") > 0 and base_count() >= 3:
        return "macro"

    # standard: pool > 60s + natural exists
    if pool_start > 60 and row.get("nat_start", -1) > 0:
        return "macro"

    # Default: macro
    return "macro"


def _extract_timing_from_events(events: list[dict]) -> dict:
    """Extract timing features and structure/unit counts from per-match events.

    Takes the latest non-null value for timing fields.
    Aggregates structure/unit counts across all events (max seen).
    """
    import ast
    import json

    timing = {}
    timing_fields = [
        "pool_start", "speed_start", "queen_time", "gas_time",
        "ling_seen", "ling_contact", "gas_workers", "ling_has_speed",
        "last_nat_scout_time", "nat_present_on_last_scout",
    ]
    rush_fields = ["score_12p", "score_speed", "auto_true_fired", "cheese_label"]

    # Track max structure/unit counts across all events
    struct_counts: dict[str, int] = {}
    unit_counts: dict[str, int] = {}
    # Track when each structure was FIRST seen (game_steps from event)
    struct_first_seen: dict[str, float] = {}
    unit_first_seen: dict[str, float] = {}
    under_attack_count = 0
    used_cheese_response = False
    worker_rush_active = False
    # Track earliest under_attack time
    first_under_attack_time: float | None = None

    def _parse_dict_field(val) -> dict | None:
        """Parse a dict field that may be a dict, JSON string, or Python repr string."""
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val.strip():
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return None

    for event in events:
        # Timing fields: take latest non-null
        for field in timing_fields + rush_fields:
            val = event.get(field)
            if val is not None and val != "" and val != -1.0:
                timing[field] = val

        # nat_start
        nat = event.get("nat_start")
        if nat is not None and nat != "" and nat > 0:
            timing["nat_start"] = nat

        # Structure counts: merge (take max across events)
        raw_structs = _parse_dict_field(event.get("scouted_enemy_structures"))
        if raw_structs:
            # Use 'ts' (per-event game time in seconds), NOT game_steps (match total)
            event_time = event.get("ts")
            if not isinstance(event_time, (int, float)):
                event_time = 0.0
            for raw_name, count in raw_structs.items():
                norm = STRUCT_ALIASES.get(raw_name, raw_name.lower())
                struct_counts[norm] = max(struct_counts.get(norm, 0), count)
                # Track first seen time
                if norm not in struct_first_seen and event_time > 0:
                    struct_first_seen[norm] = event_time

        # Unit counts: merge (take max across events)
        raw_units = _parse_dict_field(event.get("scouted_enemy_units"))
        if raw_units:
            event_time = event.get("ts")
            if not isinstance(event_time, (int, float)):
                event_time = 0.0
            for raw_name, count in raw_units.items():
                norm = UNIT_ALIASES.get(raw_name, raw_name.lower())
                unit_counts[norm] = max(unit_counts.get(norm, 0), count)
                if norm not in unit_first_seen and event_time > 0:
                    unit_first_seen[norm] = event_time

        # Boolean flags
        if event.get("under_attack"):
            under_attack_count += 1
            event_time = event.get("ts")
            if not isinstance(event_time, (int, float)):
                event_time = 0.0
            if first_under_attack_time is None and event_time > 0:
                first_under_attack_time = event_time
        if event.get("used_cheese_response"):
            used_cheese_response = True
        if event.get("worker_rush_active"):
            worker_rush_active = True

    # Store aggregated data
    timing["struct_counts"] = struct_counts
    timing["unit_counts"] = unit_counts
    timing["struct_first_seen"] = struct_first_seen
    timing["unit_first_seen"] = unit_first_seen
    timing["under_attack_count"] = under_attack_count
    timing["used_cheese_response"] = used_cheese_response
    timing["worker_rush_active"] = worker_rush_active
    timing["first_under_attack_time"] = first_under_attack_time

    return timing


def build_training_data(matches: pd.DataFrame, api_url: str = API_BASE) -> pd.DataFrame:
    """Build training data from match records + per-match events.

    Uses /api/matches for basic match info, then enriches each match with
    timing features from /api/matches/{id}/events.
    """
    rows = []
    fetched_events = 0
    max_event_fetches = min(len(matches), 500)

    for idx, (_, match) in enumerate(matches.iterrows()):
        match_id = match.get("arena_match_id", 0)
        # game_steps → seconds: SC2 runs at 16 steps/sec on fast speed
        game_steps = match.get("game_steps", 0)
        game_time = game_steps / 16.0 if game_steps else 0.0

        row = {
            "match_id": match_id,
            "opponent_id": match.get("opponent_id", "") or "",
            "enemy_race": match.get("enemy_race", "Unknown"),
            "enemy_race_int": RACE_MAP.get(match.get("enemy_race", "Unknown"), 3),
            "map": match.get("map", "") or "",
            "game_time": game_time,
            "cheese_type": match.get("cheese_type", "none") or "none",
            "strategy_category_api": match.get("strategy_category", ""),
            "build_label_api": match.get("build_label", ""),
            "result": match.get("result", ""),
            # Match-level fields available from /api/matches
            "used_cheese_response": match.get("used_cheese_response", False),
            "rush_time_seconds": match.get("rush_time_seconds", -1),
            "sq": match.get("sq", -1),
            # Defaults — overwritten by events if available
            "struct_counts": {},
            "unit_counts": {},
            "struct_first_seen": {},
            "unit_first_seen": {},
            "worker_rush_active": False,
            "first_under_attack_time": None,
        }

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
            time.sleep(0.05)
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
                # Structure/unit counts from events
                row["struct_counts"] = timing.get("struct_counts", {})
                row["unit_counts"] = timing.get("unit_counts", {})
                row["struct_first_seen"] = timing.get("struct_first_seen", {})
                row["unit_first_seen"] = timing.get("unit_first_seen", {})
                row["used_cheese_response"] = timing.get("used_cheese_response", row.get("used_cheese_response", False))
                row["worker_rush_active"] = timing.get("worker_rush_active", False)
                row["first_under_attack_time"] = timing.get("first_under_attack_time")
                # Count under_attack transitions from events
                under_count = sum(
                    1 for e in events
                    if e.get("under_attack") is True
                )
                row["under_attack_count"] = under_count

            if fetched_events % 50 == 0:
                print(f"  Fetched events for {fetched_events}/{max_event_fetches} matches...")

        # Derive ground truth label (needs struct_counts from events)
        row["strategy_label"] = derive_strategy_label(row)

        # Extract scalar structure counts for BN features
        structs = row.get("struct_counts", {}) or {}
        units = row.get("unit_counts", {}) or {}
        row["barracks_count"] = structs.get("barracks", 0)
        row["gateway_count"] = structs.get("gateway", 0) + structs.get("warpgate", 0)
        row["factory_count"] = structs.get("factory", 0)
        row["starport_count"] = structs.get("starport", 0)
        row["forge_seen"] = 1 if structs.get("forge", 0) > 0 else 0
        row["cannon_seen"] = 1 if structs.get("cannon", 0) > 0 else 0
        row["roach_warren_seen"] = 1 if structs.get("roach_warren", 0) > 0 else 0
        row["robotics_bay_seen"] = 1 if structs.get("robotics_bay", 0) > 0 else 0
        row["stargate_seen"] = 1 if structs.get("stargate", 0) > 0 else 0
        row["base_count"] = max(
            structs.get("hatchery", 0) + structs.get("lair", 0) + structs.get("hive", 0),
            structs.get("nexus", 0),
            structs.get("command_center", 0) + structs.get("orbital_command", 0) + structs.get("planetary_fortress", 0),
        )
        row["medivac_seen"] = 1 if units.get("medivac", 0) > 0 else 0
        row["siege_tank_seen"] = 1 if units.get("siege_tank", 0) > 0 else 0
        row["widow_mine_seen"] = 1 if units.get("widow_mine", 0) > 0 else 0
        row["ravager_seen"] = 1 if units.get("ravager", 0) > 0 else 0

        # === Schema v2 features (Step 1: position) ===
        # struct_first_seen is used as fallback for rax/gw timing (see below).
        struct_first = row.get("struct_first_seen", {}) or {}
        # Position booleans come from the API match record (emitted by game_report.py).
        # API sends: 1=yes (near our base), 0=no (seen but not near), -1=unknown (never seen).
        # Map to BN state strings. Unknown stays "unknown" so the BN can learn from
        # the absence of observation rather than treating it as "no".
        def _pos_str(val) -> str:
            if val == 1 or val is True:
                return "yes"
            if val == 0 or val is False:
                return "no"
            return "unknown"

        row["rax_near_base"] = _pos_str(match.get("rax_near_base", -1))
        row["gw_near_base"] = _pos_str(match.get("gw_near_base", -1))
        row["cannon_near_base"] = _pos_str(match.get("cannon_near_base", -1))
        row["bunker_near_base"] = _pos_str(match.get("bunker_near_base", -1))

        # === Schema v2 features (Step 2: timing) ===
        # Use match-level timing fields directly — the event-based struct_first_seen
        # is unreliable (API often returns only the last batch of events at ~714s).
        # pool_start, nat_start, rax_start, and gw_start are all populated at match
        # level from rush_detect/enemy_timings telemetry and are far more accurate
        # for early-game timing than event-derived struct_first_seen.
        pool_time = row.get("pool_start", -1)
        row["pool_timing_raw"] = pool_time if pool_time and pool_time > 0 else -1

        # rax/gw timing: prefer match-level rax_start/gw_start (from enemy_timings.py
        # via game_report.py), fall back to struct_first_seen if missing.
        rax_time = match.get("rax_start", -1)
        if rax_time is None or rax_time < 0:
            rax_time = struct_first.get("barracks", -1)
        row["rax_timing_raw"] = rax_time if rax_time is not None and rax_time > 0 else -1

        gw_time = match.get("gw_start", -1)
        if gw_time is None or gw_time < 0:
            gw_time = struct_first.get("gateway", -1)
        row["gw_timing_raw"] = gw_time if gw_time is not None and gw_time > 0 else -1

        # nat_timing: use nat_start from match level (populated for ~78% of matches)
        nat_time = row.get("nat_start", -1)
        row["nat_timing_raw"] = nat_time if nat_time and nat_time > 0 else -1

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

    # ── New bins for expanded BN ──────────────────────────────────────

    # Barracks count bins: none, few, many
    df["rax_bin"] = pd.cut(
        df["barracks_count"].fillna(0),
        bins=[-1, 0, 2, float("inf")],
        labels=["none", "few", "many"],
    ).astype(str)

    # Gateway count bins: none, few, many
    df["gateway_bin"] = pd.cut(
        df["gateway_count"].fillna(0),
        bins=[-1, 0, 3, float("inf")],
        labels=["none", "few", "many"],
    ).astype(str)

    # Base count bins: one, two, three_plus
    df["bases_bin"] = pd.cut(
        df["base_count"].fillna(0),
        bins=[-1, 1, 2, float("inf")],
        labels=["one", "two", "three_plus"],
    ).astype(str)

    # Factory seen: yes/no
    df["factory_bin"] = df["factory_count"].fillna(0).apply(
        lambda x: "yes" if x > 0 else "no"
    )

    # Starport seen: yes/no
    df["starport_bin"] = df["starport_count"].fillna(0).apply(
        lambda x: "yes" if x > 0 else "no"
    )

    # Tech structures: forge, roach_warren, robotics_bay → yes/no each
    df["forge_bin"] = df["forge_seen"].fillna(0).apply(
        lambda x: "yes" if x > 0 else "no"
    )
    df["roach_warren_bin"] = df["roach_warren_seen"].fillna(0).apply(
        lambda x: "yes" if x > 0 else "no"
    )
    df["robo_bay_bin"] = df["robotics_bay_seen"].fillna(0).apply(
        lambda x: "yes" if x > 0 else "no"
    )

    # === Schema v2: Position features (Step 1) ===
    # Collapse to 2 states: "yes" (near our base — proxy/cannon rush) vs
    # "not_yes" (everything else: seen-not-near OR unknown).
    # The key signal is "yes" — it's the proxy indicator. Collapsing "no" and
    # "unknown" keeps that signal while cutting CPD size 3x per variable.
    # Memory constraint: 15 parents with 3 states each = 5.2 GiB CPD → OOM.
    # Reducing to 2 states per position var keeps the CPD trainable.
    def _pos_collapse(val) -> str:
        return "yes" if val == "yes" or val == 1 or val is True else "not_yes"
    for col in ["rax_near_base", "gw_near_base", "cannon_near_base", "bunker_near_base"]:
        if col not in df.columns:
            df[col] = "not_yes"
        else:
            df[col] = df[col].apply(_pos_collapse)

    # === Schema v2: Timing features (Step 2) ===
    # Collapse from 5 states (none/very_early/early/standard/late) to 3 states
    # (none/early/standard). The key signal is "early" (proxy/cheese) vs
    # "standard" (macro). "very_early" → "early" (same signal: aggressive).
    # "late" → "standard" (late = still macro, not cheese).
    # This cuts timing from 5^4=625 to 3^4=81 combinations — 8x CPD reduction.
    df["rax_timing"] = df["rax_timing_raw"].fillna(-1).apply(
        lambda t: "none" if t < 0 else ("early" if t < 45 else "standard")
    )

    # pool_timing: <25=very_early (12-pool), 25-40=early, 40-80=standard, >80=late, none
    df["pool_timing"] = df["pool_timing_raw"].fillna(-1).apply(
        lambda t: "none" if t < 0 else ("early" if t < 40 else "standard")
    )

    # gw_timing: <20=very_early (proxy), 20-40=early, 40-80=standard, >80=late, none
    df["gw_timing"] = df["gw_timing_raw"].fillna(-1).apply(
        lambda t: "none" if t < 0 else ("early" if t < 40 else "standard")
    )

    # nat_timing: <60=very_early (greedy), 60-120=early (macro), 120-240=late (all-in), >240=none, none=-1
    df["nat_timing"] = df["nat_timing_raw"].fillna(-1).apply(
        lambda t: "none" if t < 0 else ("early" if t < 120 else "standard")
    )

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
    # Schema v2: 15 parent nodes for better discrimination:
    #   enemy_race: 100% coverage — race determines available strategies
    #   duration_bin: 100% coverage — cheese/all_in end early
    #   pool_bin: Zerg signal — very_early = cheese, early = all_in
    #   rax_bin: Terran signal — few+one_base = cheese/all_in
    #   gateway_bin: Protoss signal — many+one_base = four_gate
    #   bases_bin: economy signal — one = cheese/all_in, three_plus = macro
    #   factory_bin: Terran tech signal — yes = timing or mech
    #   rax_near_base: proxy rax vs standard (Step 1)
    #   gw_near_base: proxy gateway vs standard (Step 1)
    #   cannon_near_base: cannon rush vs standard forge (Step 1)
    #   bunker_near_base: bunker rush vs standard (Step 1)
    #   rax_timing: fine-grained barracks timing (Step 2)
    #   pool_timing: fine-grained pool timing (Step 2)
    #   gw_timing: fine-grained gateway timing (Step 2)
    #   nat_timing: natural expansion timing (Step 2)
    model = DiscreteBayesianNetwork([
        ("enemy_race", "strategy"),
        ("duration_bin", "strategy"),
        ("pool_bin", "strategy"),
        ("rax_bin", "strategy"),
        ("gateway_bin", "strategy"),
        ("bases_bin", "strategy"),
        ("factory_bin", "strategy"),
        # Step 1: position features
        ("rax_near_base", "strategy"),
        ("gw_near_base", "strategy"),
        ("cannon_near_base", "strategy"),
        ("bunker_near_base", "strategy"),
        # Step 2: timing features
        ("rax_timing", "strategy"),
        ("pool_timing", "strategy"),
        ("gw_timing", "strategy"),
        ("nat_timing", "strategy"),
    ])

    # Fit parameters with MLE estimator
    train_cols = ["enemy_race", "duration_bin", "pool_bin",
                  "rax_bin", "gateway_bin", "bases_bin", "factory_bin",
                  "rax_near_base", "gw_near_base", "cannon_near_base", "bunker_near_base",
                  "rax_timing", "pool_timing", "gw_timing", "nat_timing",
                  "strategy"]
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
        "schema_version": 2,  # Schema v2 = 15 parent variables
    }


def _load_previous_bn():
    """Load the previous BN model for self-correction comparison.

    Returns a BNInference instance or None if no prior model exists.
    Uses the runtime numpy .npz loader (no pgmpy needed).
    """
    try:
        from bot.belief.bn_inference import BNInference
        bn = BNInference()
        if bn.is_loaded:
            return bn
    except Exception as e:
        print(f"[SelfCorrect] Could not load previous BN: {e}")
    return None


def _build_evidence_from_row(row) -> dict[str, str]:
    """Build BN evidence dict from a training row (post-discretization).

    Mirrors the parent variable names used in train_bn().
    """
    return {
        "enemy_race": str(row.get("enemy_race", "Unknown")),
        "duration_bin": str(row.get("duration_bin", "unknown")),
        "pool_bin": str(row.get("pool_bin", "unknown")),
        "rax_bin": str(row.get("rax_bin", "none")),
        "gateway_bin": str(row.get("gateway_bin", "none")),
        "bases_bin": str(row.get("bases_bin", "one")),
        "factory_bin": str(row.get("factory_bin", "no")),
        "rax_near_base": str(row.get("rax_near_base", "unknown")),
        "gw_near_base": str(row.get("gw_near_base", "unknown")),
        "cannon_near_base": str(row.get("cannon_near_base", "unknown")),
        "bunker_near_base": str(row.get("bunker_near_base", "unknown")),
        "rax_timing": str(row.get("rax_timing", "none")),
        "pool_timing": str(row.get("pool_timing", "none")),
        "gw_timing": str(row.get("gw_timing", "none")),
        "nat_timing": str(row.get("nat_timing", "none")),
    }


def _deterministic_guard_label(row) -> str | None:
    """Return a high-confidence label from timing guards, or None.

    These are the same early-game timing signals the bot uses at runtime.
    When the guard agrees with the ground-truth label (but the previous BN
    disagreed), that row is a correction signal — the BN was wrong and the
    ground truth + guard both say otherwise.
    """
    pool_start = row.get("pool_start", -1)
    pool_timing = str(row.get("pool_timing", "none"))
    nat_timing = str(row.get("nat_timing", "none"))
    bases_bin = str(row.get("bases_bin", "one"))
    duration_bin = str(row.get("duration_bin", "unknown"))
    enemy_race = str(row.get("enemy_race", "Unknown"))

    # 12-pool: pool < 35s is always cheese, regardless of what the replay
    # parser or BN says. This is the strongest deterministic signal in SC2.
    if pool_start is not None and pool_start > 0 and pool_start < 35:
        return "cheese"
    if pool_timing == "very_early":
        return "cheese"

    # Speedling: pool 35-52s + early gas = cheese (Zerg only)
    if enemy_race == "Zerg" and pool_timing == "early":
        gas_time = row.get("gas_time", -1)
        if gas_time is not None and gas_time > 0 and gas_time < 60:
            return "cheese"

    # One base + short game = all_in (no natural, game ends fast)
    if bases_bin == "one" and duration_bin == "short":
        return "all_in"

    # Three+ bases + long game = macro
    if bases_bin == "three_plus" and duration_bin in ("long", "very_long"):
        return "macro"

    # Natural on time + standard pool = macro
    if nat_timing == "early" and pool_timing == "standard":
        return "macro"

    return None


def apply_self_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Compare previous BN predictions to ground-truth labels and report errors.

    This doesn't relabel rows — the ground-truth labels are already correct
    (they come from the replay). Instead it:
      1. Loads the previous BN model
      2. Predicts on each training row
      3. Compares to the ground-truth label
      4. Reports where the BN was wrong, and whether a deterministic guard
         also disagrees with the BN (confirming the ground truth)

    The correction happens implicitly: MLE fitting on these rows will update
    the CPD entries that the previous BN got wrong. Rows where the BN was
    wrong AND a guard confirms the ground truth are the strongest correction
    signal — those evidence combinations will shift the CPD hardest.

    Returns the df unchanged (labels are already ground truth). Prints stats.
    """
    prev_bn = _load_previous_bn()
    if prev_bn is None:
        print("[SelfCorrect] No previous BN model found — first training run.")
        return df

    df = df.copy()
    df_disc = discretize_features(df)
    df_disc = df_disc.rename(columns={"strategy_label": "strategy"})
    df_disc = df_disc[df_disc["strategy"].notna()]

    if len(df_disc) == 0:
        print("[SelfCorrect] No labeled rows to compare.")
        return df

    correct = 0
    wrong = 0
    guard_confirmed = 0
    wrong_by_label: dict[str, int] = {}
    wrong_by_race: dict[str, int] = {}

    for _, row in df_disc.iterrows():
        true_label = row["strategy"]
        evidence = _build_evidence_from_row(row)
        probs = prev_bn.predict(**evidence)
        # predict() returns dict[StrategyCategory, float]; keys are enum members
        pred_cat = max(probs, key=probs.get) if probs else None
        pred_label = pred_cat.value if pred_cat else "unknown"

        if pred_label == true_label:
            correct += 1
        else:
            wrong += 1
            wrong_by_label[true_label] = wrong_by_label.get(true_label, 0) + 1
            race = str(row.get("enemy_race", "Unknown"))
            wrong_by_race[race] = wrong_by_race.get(race, 0) + 1

            # Check if a deterministic guard also disagrees with the BN
            guard_label = _deterministic_guard_label(row)
            if guard_label is not None and guard_label == true_label:
                guard_confirmed += 1

    total = correct + wrong
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n=== Self-Correction: Previous BN vs Ground Truth ===")
    print(f"  Previous BN accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Wrong predictions: {wrong}")
    if guard_confirmed > 0:
        print(f"  Guard-confirmed corrections: {guard_confirmed} "
              f"(BN wrong, guard + ground truth agree)")
    if wrong_by_label:
        print(f"  Errors by true label:")
        for label, count in sorted(wrong_by_label.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")
    if wrong_by_race:
        print(f"  Errors by race:")
        for race, count in sorted(wrong_by_race.items(), key=lambda x: -x[1]):
            print(f"    {race}: {count}")
    print(f"  → These {wrong} rows will correct the CPD via MLE fitting.")

    return df


def filter_ground_truth_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where the API provided a replay-derived strategy_category.

    This drops heuristic-derived labels so the BN CPD is trained purely from
    the replay source of truth, not from the bot's own rule-based predictions.
    Also normalizes the API label ("timing" → "timing_attack").
    """
    before = len(df)

    def _api_label(row) -> str | None:
        raw = row.get("strategy_category_api")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            raw = ""
        raw = str(raw).strip().lower()
        if raw == "timing":
            raw = "timing_attack"
        if raw in ("cheese", "all_in", "timing_attack", "macro"):
            return raw
        return None

    df = df.copy()
    df["_api_label"] = df.apply(_api_label, axis=1)
    df = df[df["_api_label"].notna()].copy()
    df["strategy_label"] = df["_api_label"]
    df = df.drop(columns=["_api_label"])

    dropped = before - len(df)
    print(f"[GroundTruthOnly] Kept {len(df)}/{before} rows "
          f"(dropped {dropped} heuristic-only labels)")
    return df


def build_category_prior(df: pd.DataFrame, output_path: str = str(CATEGORY_PRIOR_FILE)) -> dict:
    """Compute marginal P(strategy) from API ground-truth labels.

    Replaces the hardcoded STRATEGY_CATEGORY_PRIOR in constants.py with a
    data-driven prior derived from replay analysis. Writes a JSON file that
    StrategyBelief loads at runtime.
    """
    import json

    categories = STRATEGY_CATEGORY_VALUES

    if "strategy_label" not in df.columns:
        print("[CategoryPrior] Missing strategy_label column. Skipping.")
        return {}

    valid = df.dropna(subset=["strategy_label"])
    valid = valid[valid["strategy_label"].isin(categories)]

    if len(valid) == 0:
        print("[CategoryPrior] No valid labels. Skipping.")
        return {}

    counts = {cat: 0 for cat in categories}
    for label in valid["strategy_label"]:
        if label in counts:
            counts[label] += 1

    total = sum(counts.values())
    probs = {cat: counts[cat] / total for cat in categories}

    data = {
        "schema_version": 1,
        "categories": categories,
        "counts": {cat: counts[cat] for cat in categories},
        "probs": {cat: round(probs[cat], 4) for cat in categories},
        "n_samples": total,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n=== Category prior saved to {output_path} ===")
    print(f"  Samples: {total}")
    for cat in categories:
        print(f"  {cat}: {counts[cat]} ({probs[cat]:.1%})")

    return data


def build_opponent_priors(df: pd.DataFrame, output_path: str = str(OPPONENT_PRIORS_FILE)) -> dict:
    """Build per-opponent Dirichlet alpha parameters from match data.

    Groups matches by (opponent_id, enemy_race), counts strategy_label outcomes,
    and computes alpha parameters (baseline [1,1,1,1] + observed counts).

    Only includes opponents with >= MIN_GAMES_PER_OPPONENT games to avoid
    overfitting to small samples.

    Args:
        df: Training data DataFrame with columns: opponent_id, enemy_race, strategy_label
        output_path: Path to write the JSON priors file.

    Returns:
        Dict with schema_version, categories, and per-opponent alpha params.
    """
    import json

    categories = STRATEGY_CATEGORY_VALUES  # ["cheese", "all_in", "timing_attack", "macro"]

    # Check required columns
    if "opponent_id" not in df.columns or "strategy_label" not in df.columns:
        print("[OpponentPriors] Missing required columns (opponent_id, strategy_label). Skipping.")
        return {}

    # Filter out rows without opponent_id or strategy_label
    valid = df.dropna(subset=["opponent_id", "strategy_label"])
    valid = valid[valid["opponent_id"].astype(str) != "None"]
    valid = valid[valid["opponent_id"].astype(str) != ""]

    if len(valid) == 0:
        print("[OpponentPriors] No valid opponent data. Skipping.")
        return {}

    # Group by (opponent_id, enemy_race) and count strategy outcomes
    priors = {}
    opponent_counts = {}

    for (opp_id, race), group in valid.groupby(["opponent_id", "enemy_race"]):
        if len(group) < MIN_GAMES_PER_OPPONENT:
            continue

        # Count strategy outcomes
        counts = {cat: 0 for cat in categories}
        for label in group["strategy_label"]:
            if label in counts:
                counts[label] += 1

        # Alpha = baseline (1) + observed counts
        alphas = [1.0 + counts[cat] for cat in categories]
        key = f"{opp_id}:{race}"
        priors[key] = alphas
        opponent_counts[key] = len(group)

    if not priors:
        print(f"[OpponentPriors] No opponents with >= {MIN_GAMES_PER_OPPONENT} games. Skipping.")
        return {}

    # Build output structure
    # NOTE: key must be "profiles" to match OpponentBelief._load_file() which
    # reads data.get("profiles", {})
    data = {
        "schema_version": 1,
        "categories": categories,
        "profiles": priors,
    }

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n=== Opponent priors saved to {output_path} ===")
    print(f"Opponents with >= {MIN_GAMES_PER_OPPONENT} games: {len(priors)}")
    print(f"Total games used: {sum(opponent_counts.values())}")

    # Show top opponents by game count
    top = sorted(opponent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top opponents by game count:")
    for key, count in top:
        alphas = priors[key]
        alpha_str = ", ".join(f"{cat}={a:.0f}" for cat, a in zip(categories, alphas))
        print(f"  {key}: {count} games -> {alpha_str}")

    return data


MAP_PRIORS_FILE = Path("bot/models/map_priors.json")
MIN_GAMES_PER_MAP = 3


def build_map_priors(df: pd.DataFrame, output_path: str = str(MAP_PRIORS_FILE)) -> dict:
    """Build per-map Dirichlet alpha parameters from match data.

    Groups matches by map name, counts strategy_label outcomes, and computes
    alpha parameters (baseline [1,1,1,1] + observed counts). This captures
    patterns like "I get rushed more on Pylon AIE than on other maps."

    Only includes maps with >= MIN_GAMES_PER_MAP games.
    """
    import json

    categories = STRATEGY_CATEGORY_VALUES

    if "map" not in df.columns or "strategy_label" not in df.columns:
        print("[MapPriors] Missing required columns (map, strategy_label). Skipping.")
        return {}

    valid = df.dropna(subset=["map", "strategy_label"])
    valid = valid[valid["map"].astype(str) != "None"]
    valid = valid[valid["map"].astype(str) != ""]

    if len(valid) == 0:
        print("[MapPriors] No valid map data. Skipping.")
        return {}

    priors = {}
    map_counts = {}

    for map_name, group in valid.groupby("map"):
        if len(group) < MIN_GAMES_PER_MAP:
            continue

        counts = {cat: 0 for cat in categories}
        for label in group["strategy_label"]:
            if label in counts:
                counts[label] += 1

        alphas = [1.0 + counts[cat] for cat in categories]
        priors[str(map_name)] = alphas
        map_counts[str(map_name)] = len(group)

    if not priors:
        print(f"[MapPriors] No maps with >= {MIN_GAMES_PER_MAP} games. Skipping.")
        return {}

    data = {
        "schema_version": 1,
        "categories": categories,
        "maps": priors,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n=== Map priors saved to {output_path} ===")
    print(f"Maps with >= {MIN_GAMES_PER_MAP} games: {len(priors)}")
    for map_name, count in sorted(map_counts.items(), key=lambda x: x[1], reverse=True):
        alphas = priors[map_name]
        alpha_str = ", ".join(f"{cat}={a:.0f}" for cat, a in zip(categories, alphas))
        print(f"  {map_name}: {count} games -> {alpha_str}")

    return data


def main():
    parser = argparse.ArgumentParser(description="Train Strategy Belief BN model")
    parser.add_argument("--api-url", default=API_BASE, help="Telemetry API base URL")
    parser.add_argument("--output", default=str(MODEL_FILE), help="Output model path")
    parser.add_argument("--limit", type=int, default=500, help="Max matches to fetch")
    parser.add_argument("--priors-output", default=str(OPPONENT_PRIORS_FILE),
                        help="Output path for opponent priors JSON")
    parser.add_argument("--skip-priors", action="store_true",
                        help="Skip building opponent priors")
    parser.add_argument("--ground-truth-only", dest="ground_truth_only",
                        action=argparse.BooleanOptionalAction,
                        default=GROUND_TRUTH_ONLY_DEFAULT,
                        help="Only train on rows with API replay-derived labels "
                             "(default: True). Use --no-ground-truth-only to also "
                             "include heuristic-derived labels.")
    parser.add_argument("--self-correct", dest="self_correct",
                        action=argparse.BooleanOptionalAction,
                        default=SELF_CORRECT_DEFAULT,
                        help="Compare previous BN predictions to ground-truth labels "
                             "and report errors before retraining (default: True). "
                             "The MLE fit on ground-truth rows automatically corrects "
                             "CPD entries the previous BN got wrong.")
    args = parser.parse_args()

    print("=== Strategy Belief Model Training ===\n")

    # Fetch data
    matches = fetch_matches(limit=args.limit, api_url=args.api_url)

    # Build training data
    print("\nBuilding training data...")
    df = build_training_data(matches, api_url=args.api_url)

    # Ground-truth-only filtering: drop heuristic-derived labels so the BN
    # learns purely from replay-derived strategy_category.
    if args.ground_truth_only:
        print("\n=== Ground-Truth-Only Mode ===")
        print("Filtering to rows with API replay-derived strategy_category...")
        df = filter_ground_truth_only(df)
        if len(df) < 20:
            print(f"WARNING: Only {len(df)} ground-truth rows after filtering.")
            print("Consider collecting more replays or use --no-ground-truth-only.")
    else:
        print("\n=== Mixed-Label Mode (heuristic fallbacks included) ===")
        print("WARNING: BN will be partly trained on heuristic-derived labels.")

    # Self-correction: compare previous BN to ground truth before retraining.
    # The MLE fit on these ground-truth rows will correct CPD entries the
    # previous BN got wrong — each retraining cycle fixes the previous errors.
    if args.self_correct:
        print("\n=== Self-Correction Mode ===")
        df = apply_self_correction(df)

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

        # Auto-export .npz for runtime (no pgmpy dependency)
        npz_path = Path(args.output).with_suffix(".npz")
        try:
            from scripts.export_bn_model import export_model
            export_model(Path(args.output), npz_path)
            print(f"=== Also exported .npz to {npz_path} ===")
        except Exception as e:
            print(f"Warning: .npz export failed ({e}). Run export_bn_model.py manually.")
    else:
        print("\n=== BN model NOT saved — rule-based guards will be used ===")
        print("Collect more data (50+ games per race) and re-run this script.")

    # Build data-driven marginal prior P(strategy) from API ground truth
    print("\n=== Building Category Prior (marginal) ===")
    build_category_prior(df)

    # Build opponent priors from match data
    if not args.skip_priors:
        print("\n=== Building Opponent Priors ===")
        build_opponent_priors(df, output_path=args.priors_output)
        print("\n=== Building Map Priors ===")
        build_map_priors(df)
    else:
        print("\n=== Skipping priors (--skip-priors) ===")


if __name__ == "__main__":
    main()