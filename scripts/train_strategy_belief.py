"""Train sklearn GradientBoostingClassifier for Strategy Belief classification.

Purpose: Train a strategy classifier from telemetry API data to predict
         opponent strategy into cheese/all_in/timing_attack/macro. Falls back
         to rule-based guards when the model file is missing.

Key Decisions: Queries /api/features/match-level-full for enriched data.
                Uses strategy_category column (when populated) as ground truth.
                Phase B hard cutover (2026-09-08): sklearn GBC replaces the
                pgmpy BN — BN scored 16.9% on holdout vs 67.7% for sklearn.
                The artifact is a single joblib .pkl dict {model, feature_cols,
                classes}; runtime loads it via SklearnInference with no export step.

Limitations: Requires 50+ games per race for reliable training.
              Class imbalance (macro-heavy) — consider class_weight if
              per-class recall shows cheese/all_in starving.

Usage:
    python scripts/train_strategy_belief.py [--api-url URL] [--output PATH]

Output:
    bot/models/strategy_model_sklearn.pkl
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


def fetch_matches(limit: int = 2000, api_url: str = API_BASE) -> pd.DataFrame:
    """Fetch match records from the telemetry API.

    Tries /api/features/match-level-full first (enriched endpoint).
    Falls back to /api/matches (basic endpoint) if the full one fails.

    Note: API max limit is 2000 and the offset param is ignored —
    the reachable set is the newest 2000 matches.
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


_EVENTS_CACHE_PATH = Path("data/events_cache.pkl")
_events_cache: dict | None = None


def _events_cache_lookup(match_id: int) -> list[dict] | None:
    """Return cached events for a match id, or None if not cached.

    Loads data/events_cache.pkl once (populated by eval_bn_vs_sklearn.py's
    parallel prefetch). Missing cache or missing id → None (serial fetch path).
    """
    global _events_cache
    if _events_cache is None:
        if not _EVENTS_CACHE_PATH.exists():
            _events_cache = {}
            return None
        try:
            import pickle
            with open(_EVENTS_CACHE_PATH, "rb") as f:
                _events_cache = pickle.load(f)
        except Exception:
            _events_cache = {}
            return None
    return _events_cache.get(match_id)


def fetch_match_events(match_id: int, api_url: str = API_BASE) -> list[dict]:
    """Fetch all events for a specific match.

    The first event in the response typically contains rush_detect timing
    features (pool_start, gas_time, ling_seen, etc.) even for non-Zerg games.

    Uses the shared events cache when present (data/events_cache.pkl) — the
    endpoint is ~2.5s per request server-side, so training runs prefetch in
    parallel via scripts/eval_bn_vs_sklearn.py and reuse the cache here.
    """
    cached = _events_cache_lookup(match_id)
    if cached is not None:
        return cached

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
    # Event fetches are rate-limited (0.05s sleep each) — cap at 2000 (full window)
    max_event_fetches = min(len(matches), 2000)

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

        # Default missing timing features; prefer match-level values (populated
        # since the 2026-06-19 API fix) — event enrichment fills the rest
        _fill_missing_timing(row)
        row["pool_start"] = match.get("pool_start", -1) if match.get("pool_start") is not None else -1
        row["gas_time"] = match.get("gas_time", -1) if match.get("gas_time") is not None else -1
        row["ling_seen"] = match.get("ling_seen", -1) if match.get("ling_seen") is not None else -1
        row["ling_contact"] = match.get("ling_contact", -1) if match.get("ling_contact") is not None else -1
        row["nat_start"] = match.get("nat_start", -1) if match.get("nat_start") is not None else -1
        row["last_nat_scout_time"] = match.get("last_nat_scout_time", -1) if match.get("last_nat_scout_time") is not None else -1
        row["nat_present_on_last_scout"] = match.get("nat_present_on_last_scout", -1) if match.get("nat_present_on_last_scout") is not None else -1
        # Zerg detail fields — match-level since the 2026-09-09 endpoint fix;
        # None for old games (0.10-0.12 gate era), -1 = never observed in-game
        row["queen_time"] = match.get("queen_time", -1) if match.get("queen_time") is not None else -1
        row["speed_start"] = match.get("speed_start", -1) if match.get("speed_start") is not None else -1
        row["ling_has_speed"] = match.get("ling_has_speed", 0) if match.get("ling_has_speed") is not None else 0
        row["gas_workers"] = match.get("gas_workers", 0) if match.get("gas_workers") is not None else 0
        # Tech structure timings (v0.13.0 emissions — -1 until new games arrive)
        for f in ["factory_start", "starport_start", "stargate_start",
                  "robotics_facility_start", "robotics_bay_start",
                  "baneling_nest_start", "roach_warren_start", "spire_start"]:
            row[f] = match.get(f, -1) if match.get(f) is not None else -1

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

    # === Schema v3: Disambiguation features ===
    # Pool timing alone can't separate "safe pool → drone" from "pool → ling flood"
    # (62% of pool<42s games on this ladder are replay-labeled macro). These
    # features ARE collected in training rows and at runtime — they were left out
    # of the BN-era feature set only because of the CPD memory constraint.
    # Bins tuned via per-label quartile analysis of the 1,997 labeled games
    # (v3c variant: 73.2% holdout vs 70.2% baseline; queen/nat_present dropped —
    # queen has only 3 cheese samples, nat_present is redundant with nat_timing).
    # gas_timing: all_in takes gas ~38-42s, cheese ~52-60s — the <50 cut separates them
    df["gas_timing"] = df["gas_time"].fillna(-1).apply(
        lambda t: "none" if t < 0 else ("early" if t < 50 else ("mid" if t < 90 else "late"))
    )

    # ling_timing: macro's first-ling median (134s) is EARLIER than rush (147-151s)
    # because rush lings die attacking before being scouted — so 'mid' (120-160)
    # is the rush band, not 'early'. Three bins keep this signal.
    df["ling_timing"] = df["ling_seen"].fillna(-1).apply(
        lambda t: "none" if t < 0 else ("early" if t < 120 else ("mid" if t < 160 else "late"))
    )

    return df


SKLEARN_MODEL_FILE = Path("bot/models/strategy_model_sklearn.pkl")

SKLEARN_EVIDENCE_COLS = [
    "enemy_race", "duration_bin", "pool_bin", "rax_bin", "gateway_bin",
    "bases_bin", "factory_bin",
    "rax_near_base", "gw_near_base", "cannon_near_base", "bunker_near_base",
    "rax_timing", "pool_timing", "gw_timing", "nat_timing",
    # Schema v3 disambiguation: gas/ling timing (see discretize_features
    # for bin rationale; queen_timing + nat_present evaluated and dropped)
    "gas_timing", "ling_timing",
]


def train_sklearn(df: pd.DataFrame) -> dict:
    """Train sklearn GradientBoostingClassifier on discretized features.

    Replaces the pgmpy BN (Phase B hard cutover). The model is a Pipeline
    (OneHotEncoder + GBC) so runtime inference is a single joblib load +
    predict_proba — no encoder bookkeeping, no .npz export.

    Returns an artifact dict for joblib.dump. Empty dict on failure.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    df_disc = discretize_features(df)
    df_disc = df_disc.rename(columns={"strategy_label": "strategy"})
    df_disc = df_disc[df_disc["strategy"].notna()]

    if len(df_disc) < 20:
        print(f"WARNING: Only {len(df_disc)} labeled samples — model will be unreliable.")
        if len(df_disc) < 10:
            print("Falling back: model file will NOT be saved.")
            return {}

    print(f"\nTraining data: {len(df_disc)} rows")
    print(f"Label distribution:\n{df_disc['strategy'].value_counts()}")
    print(f"Race distribution:\n{df_disc['enemy_race'].value_counts()}")

    X = df_disc[SKLEARN_EVIDENCE_COLS].astype(str)
    y = df_disc["strategy"]

    model = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    try:
        model.fit(X, y)
    except Exception as e:
        print(f"sklearn fitting error: {e}")
        print("Falling back: model file will NOT be saved.")
        return {}

    # Validate: a synthetic prediction must succeed before we save
    try:
        synthetic = pd.DataFrame([{col: "unknown" for col in SKLEARN_EVIDENCE_COLS}])
        pred = model.predict(synthetic)[0]
        print(f"\n=== Sanity check: P(strategy | all-unknown evidence) = {pred} ===")
    except Exception as e:
        print(f"sklearn sanity check failed: {e}")
        return {}

    # Feature importances from the fitted GBC (post-onehot, aggregated per raw feature)
    try:
        clf = model.named_steps["clf"]
        ohe = model.named_steps["onehot"]
        importances = clf.feature_importances_
        # ohe.categories_ aligns with SKLEARN_EVIDENCE_COLS: category i belongs
        # to raw feature SKLEARN_EVIDENCE_COLS[i]; count states per feature to
        # slice the flat importances array correctly
        per_feature: dict[str, float] = {}
        pos = 0
        for col, cats in zip(SKLEARN_EVIDENCE_COLS, ohe.categories_):
            n = len(cats)
            per_feature[col] = float(np.sum(importances[pos:pos + n]))
            pos += n
        print("\n=== Feature importances (aggregated per raw feature) ===")
        for col, imp in sorted(per_feature.items(), key=lambda kv: -kv[1]):
            print(f"  {col:20s} {imp:.3f}")
    except Exception as e:
        print(f"(feature importance report skipped: {e})")

    return {
        "model": model,
        "feature_cols": SKLEARN_EVIDENCE_COLS,
        "classes": list(model.classes_),
        "n_samples": len(df_disc),
        # Model generation stamp — OpponentBelief compares this against the
        # runtime profile file's stored epoch at load; a mismatch discards the
        # accumulated runtime profiles and re-seeds from the training baseline.
        # Makes retrain-resets automatic: new artifact → new epoch → stale
        # runtime data invalidated on first game, forever.
        "model_epoch": int(time.time()),
    }


def _load_previous_model():
    """Load the previous sklearn model for self-correction comparison.

    Returns a (model, feature_cols, classes) tuple or None if no prior
    artifact exists. Loads the .pkl directly — no bot package import
    (which would pull ares and fail outside the game environment).
    """
    if not SKLEARN_MODEL_FILE.exists():
        return None
    try:
        import joblib as _joblib
        artifact = _joblib.load(SKLEARN_MODEL_FILE)
        return artifact["model"], list(artifact["feature_cols"]), list(artifact["classes"])
    except Exception as e:
        print(f"[SelfCorrect] Could not load previous sklearn model: {e}")
        return None


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
      1. Loads the previous sklearn model
      2. Predicts on each training row
      3. Compares to the ground-truth label
      4. Reports where the model was wrong, and whether a deterministic guard
          also disagrees with the model (confirming the ground truth)

    The correction happens implicitly: fitting on these rows will update
    the model where the previous one was wrong. Rows where the model was
    wrong AND a guard confirms the ground truth are the strongest correction
    signal.

    Returns the df unchanged (labels are already ground truth). Prints stats.
    """
    prev = _load_previous_model()
    if prev is None:
        print("[SelfCorrect] No previous sklearn model found — first training run.")
        return df
    prev_model, feature_cols, _classes = prev

    df = df.copy()
    df_disc = discretize_features(df)
    df_disc = df_disc.rename(columns={"strategy_label": "strategy"})
    df_disc = df_disc[df_disc["strategy"].notna()]

    if len(df_disc) == 0:
        print("[SelfCorrect] No labeled rows to compare.")
        return df

    X = df_disc[feature_cols].astype(str)
    y = df_disc["strategy"].values

    correct = 0
    wrong = 0
    guard_confirmed = 0
    wrong_by_label: dict[str, int] = {}
    wrong_by_race: dict[str, int] = {}

    preds = prev_model.predict(X)
    for (idx, row), pred_label in zip(df_disc.iterrows(), preds):
        true_label = row["strategy"]

        if pred_label == true_label:
            correct += 1
        else:
            wrong += 1
            wrong_by_label[true_label] = wrong_by_label.get(true_label, 0) + 1
            race = str(row.get("enemy_race", "Unknown"))
            wrong_by_race[race] = wrong_by_race.get(race, 0) + 1

            # Check if a deterministic guard also disagrees with the model
            guard_label = _deterministic_guard_label(row)
            if guard_label is not None and guard_label == true_label:
                guard_confirmed += 1

    total = correct + wrong
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n=== Self-Correction: Previous Model vs Ground Truth ===")
    print(f"  Previous model accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Wrong predictions: {wrong}")
    if guard_confirmed > 0:
        print(f"  Guard-confirmed corrections: {guard_confirmed} "
              f"(model wrong, guard + ground truth agree)")
    if wrong_by_label:
        print(f"  Errors by true label:")
        for label, count in sorted(wrong_by_label.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")
    if wrong_by_race:
        print(f"  Errors by race:")
        for race, count in sorted(wrong_by_race.items(), key=lambda x: -x[1]):
            print(f"    {race}: {count}")
    print(f"  >> These {wrong} rows will correct the model via re-fitting.")

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
    parser = argparse.ArgumentParser(description="Train Strategy Belief sklearn model")
    parser.add_argument("--api-url", default=API_BASE, help="Telemetry API base URL")
    parser.add_argument("--output", default=str(SKLEARN_MODEL_FILE), help="Output model path")
    parser.add_argument("--limit", type=int, default=2000, help="Max matches to fetch (API max: 2000)")
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
    parser.add_argument("--from-cache", action="store_true",
                        help="Load training data from data/eval_cache_training_data.pkl.gz "
                             "(built by eval_bn_vs_sklearn.py) instead of fetching from the API")
    parser.add_argument("--self-correct", dest="self_correct",
                        action=argparse.BooleanOptionalAction,
                        default=SELF_CORRECT_DEFAULT,
                        help="Compare previous model predictions to ground-truth labels "
                             "and report errors before retraining (default: True). "
                             "The fit on ground-truth rows automatically corrects "
                             "what the previous model got wrong.")
    args = parser.parse_args()

    print("=== Strategy Belief Model Training ===\n")

    if args.from_cache:
        cache_path = Path("data/eval_cache_training_data.pkl.gz")
        if not cache_path.exists():
            print(f"FATAL: {cache_path} not found — run eval_bn_vs_sklearn.py first "
                  f"or drop --from-cache to fetch from the API.")
            return 1
        print(f"Loading training data from {cache_path}")
        df = pd.read_pickle(cache_path)
    else:
        # Fetch data
        matches = fetch_matches(limit=args.limit, api_url=args.api_url)

        # Pre-filter to labeled matches when in ground-truth-only mode: unlabeled
        # rows are dropped later anyway, and fetching their events is pure waste
        # (the events endpoint is ~2.5s per request server-side).
        if args.ground_truth_only:
            labeled_mask = matches["strategy_category"].notna() & (matches["strategy_category"] != "")
            n_labeled = int(labeled_mask.sum())
            print(f"\n{n_labeled}/{len(matches)} matches have ground-truth labels — "
                  f"building training data for those only")
            matches = matches[labeled_mask]

        # Build training data
        print("\nBuilding training data...")
        df = build_training_data(matches, api_url=args.api_url)

    # Ground-truth-only filtering: drop heuristic-derived labels so the model
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
        print("WARNING: Model will be partly trained on heuristic-derived labels.")

    # Self-correction: compare previous model to ground truth before retraining.
    # The fit on these ground-truth rows will correct what the previous model
    # got wrong — each retraining cycle fixes the previous errors.
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

    # Train sklearn model
    print("\nTraining sklearn GradientBoostingClassifier...")
    result = train_sklearn(df)

    if result and "model" in result:
        SKLEARN_MODEL_FILE.parent.mkdir(exist_ok=True)
        joblib.dump(result, args.output)
        print(f"\n=== Model saved to {args.output} ===")
        print(f"Classes: {result['classes']}")
        print(f"Training samples: {result['n_samples']}")
    else:
        print("\n=== Model NOT saved — rule-based guards will be used ===")
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