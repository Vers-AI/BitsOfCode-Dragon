"""Strategy detection — interpret observations into strategy classifications.

Purpose: Multi-detector system for classifying opponent strategies across all
         races. Consumes timing signals from enemy_timings.py and produces
         boolean + label outputs. Used as Layer 1 (auto-TRUE guards) by
         StrategyBelief, and as fallback when strategy belief is disabled.

Key Decisions: Observation ≠ interpretation — this module reads bot._* attributes
               set by enemy_timings.py and classifies them. Auto-TRUE guards
               fire first, then rule-based scoring.

Limitations: All classifiers are rule-based (no ML models). The BN model in
             StrategyBelief provides probabilistic coverage for cases the
             rules miss. Cannon rush is detected here, not in a separate module.
"""

from typing import TYPE_CHECKING

import numpy as np
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from ares.consts import UnitRole

from cython_extensions import cy_dijkstra, cy_distance_to
from bot.constants import RUSH_SPEED, RUSH_DISTANCE_CALIBRATION

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


# ── RUSH DISTANCE ──────────────────────────────────────────────────────────

def compute_rush_distance_tier(bot: "PiG_Bot") -> str:
    """Compute rush distance tier for logging purposes only.

    Uses Dijkstra pathfinding from our start to enemy start, offset by 3 tiles.
    Stores rush_time_seconds on bot object. Returns tier string.

    NOTE: Detection logic uses FIXED timing constants, not map-aware offsets.
    """
    our_offset_pos = bot.start_location.towards(bot.enemy_start_locations[0], 3)
    enemy_pos = bot.enemy_start_locations[0]

    our_x, our_y = int(our_offset_pos.x), int(our_offset_pos.y)
    enemy_x, enemy_y = int(enemy_pos.x), int(enemy_pos.y)

    cost_grid = np.where(
        bot.game_info.pathing_grid.data_numpy.T == 1,
        1.0,
        np.inf,
    ).astype(np.float64)

    targets = np.array([[our_x, our_y]], dtype=np.intp)
    dijkstra_result = cy_dijkstra(cost_grid, targets, checks_enabled=True)

    path = dijkstra_result.get_path(enemy_pos)
    ground_distance = len(path) if path else 0

    if ground_distance <= 0 or ground_distance == float('inf'):
        bot._rush_time_seconds = 0.0
        return "medium"

    rush_time = (ground_distance / RUSH_SPEED) * RUSH_DISTANCE_CALIBRATION
    bot._rush_time_seconds = rush_time

    if rush_time <= 36:
        return "short"
    elif rush_time <= 45:
        return "medium"
    return "long"


# ── SCOUT STATUS ────────────────────────────────────────────────────────────

def _probe_scout_status(bot: "PiG_Bot") -> dict:
    """Track probe scout status for adjusting heuristics."""
    scout_units = bot.mediator.get_units_from_role(
        role=UnitRole.BUILD_RUNNER_SCOUT,
        unit_type=bot.worker_type,
    )
    scout_active = len(scout_units) > 0

    if not hasattr(bot, '_scout_died_early'):
        bot._scout_died_early = False
    if not hasattr(bot, '_scout_was_active'):
        bot._scout_was_active = False

    if bot._scout_was_active and not scout_active:
        if bot.time < 120.0 and not hasattr(bot, '_saw_enemy_natural_before_scout_died'):
            bot._scout_died_early = True

    bot._scout_was_active = scout_active

    saw_enemy_natural = any(
        bot.enemy_structures.closer_than(15, bot.mediator.get_enemy_nat)
    )
    if saw_enemy_natural:
        bot._saw_enemy_natural_before_scout_died = True

    return {
        'scout_active': scout_active,
        'scout_died_early': bot._scout_died_early,
        'saw_enemy_natural': saw_enemy_natural,
    }


# ── LING RUSH SIGNALS ─────────────────────────────────────────────────────

def get_ling_rush_signals(bot: "PiG_Bot") -> dict:
    """Gather observable signals for Zerg rush detection.

    Returns dict with: tier, natural_absent, lings_near_base, total_lings,
    speed_seen, scout_died_early, saw_natural.
    """
    scout_status = _probe_scout_status(bot)
    tier = bot.rush_distance_tier

    enemy_nat_pos = bot.mediator.get_enemy_nat

    if not hasattr(bot, '_natural_ever_scouted'):
        bot._natural_ever_scouted = False

    visibility_state = bot.state.visibility[enemy_nat_pos.rounded]
    currently_visible = visibility_state == 2
    previously_seen = visibility_state >= 1

    if currently_visible or previously_seen:
        bot._natural_ever_scouted = True

    structures_at_natural = bot.enemy_structures.closer_than(15, enemy_nat_pos)
    has_natural_structure = len(structures_at_natural) > 0
    natural_absent = bot._natural_ever_scouted and not has_natural_structure

    enemy_lings = bot.mediator.get_enemy_army_dict.get(UnitTypeId.ZERGLING, [])
    lings_near_main = [
        ling for ling in enemy_lings
        if cy_distance_to(ling.position, bot.start_location) < 50
    ]

    return {
        'tier': tier,
        'natural_absent': natural_absent,
        'lings_near_base': len(lings_near_main),
        'total_lings': len(enemy_lings),
        'speed_seen': bot._ling_has_speed if hasattr(bot, '_ling_has_speed') else False,
        'scout_died_early': scout_status['scout_died_early'],
        'saw_natural': scout_status['saw_enemy_natural'],
    }


# ── MAIN ENTRY POINT ───────────────────────────────────────────────────────

def detect_cheese(bot: "PiG_Bot") -> bool:
    """Multi-detector strategy detection entry point.

    Dispatches to race-specific detectors. Returns True if any
    cheese/all-in is detected.
    """
    race = bot.enemy_race.name

    # Worker rush (all races, ARES mediator)
    if detect_worker_rush(bot):
        return True

    if race == "Zerg" or (race == "Random" and not bot.mediator.get_enemy_nat):
        return _detect_zerg_ling_rush(bot) or _detect_zerg_allin(bot)
    if race == "Terran":
        return _detect_terran_strategy(bot)
    if race == "Protoss":
        return _detect_protoss_strategy(bot)

    # Random with known race — try all detectors
    return (
        _detect_zerg_ling_rush(bot)
        or _detect_zerg_allin(bot)
        or _detect_terran_strategy(bot)
        or _detect_protoss_strategy(bot)
    )


# ── WORKER RUSH ─────────────────────────────────────────────────────────────

def detect_worker_rush(bot: "PiG_Bot") -> bool:
    """Detect worker rush via ARES mediator (all races)."""
    if bot.mediator.get_enemy_worker_rushed and bot.game_state == 0:
        if not hasattr(bot, '_worker_rush_detected'):
            bot._worker_rush_detected = True
            bot._worker_rush_detected_time = bot.time
            bot._not_worker_rush = False
            bot._cheese_source = "auto-TRUE:ARES"
            bot._strategy_chat_pending = "(worker rush detected)"
            print(f"{bot.time_formatted}: Worker rush detected (ARES)")
        return True
    return False


# ── ZERG DETECTOR ──────────────────────────────────────────────────────────

def _detect_zerg_ling_rush(bot: "PiG_Bot") -> bool:
    """Three-label zergling rush detection with Auto-TRUE guards.

    Labels: 12_pool, speedling, none.
    """
    if not hasattr(bot, '_cheese_detected'):
        bot._cheese_detected = False
        bot._cheese_label = "none"
        bot._score_12p = 0
        bot._score_speed = 0
        bot._auto_true_fired = False

    if bot._cheese_detected:
        return True

    signals = get_ling_rush_signals(bot)

    # === TIMING CONSTANTS ===
    T_POOL_12P_START_MIN = 38.0
    T_POOL_12P_START_MAX = 42.0
    T_POOL_12P_DONE = 65.0
    T_POOL_SPEED_START_MIN = 48.0
    T_POOL_SPEED_START_MAX = 52.0
    T_NAT_CHECK = 80.0
    T_NAT_CONFIRM_MISSING = 105.0
    T_LING_EARLY = 105.0
    T_LING_12P_SEEN = 115.0
    T_CONTACT_SLOW = 120.0
    T_CONTACT_SPEED = 160.0
    T_SPEED_START_EARLY = 85.0
    T_QUEEN_CHECK = 120.0

    time_now = bot.time

    # === AUTO-TRUE GUARDS ===
    first_ling = getattr(bot, '_first_ling_seen_time', None)
    first_contact = getattr(bot, '_first_ling_contact_nat_time', None)
    has_speed = getattr(bot, '_ling_has_speed', False)

    # Guard A: Any ling seen ≤ 1:45 → 12_pool
    if first_ling is not None and first_ling <= T_LING_EARLY:
        bot._cheese_detected = True
        bot._cheese_label = "12_pool"
        bot._auto_true_fired = True
        bot._cheese_source = "auto-TRUE"
        bot._strategy_chat_pending = "(early lings detected)"
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE A): "
              f"Ling seen at {first_ling:.1f}s → 12_pool")
        return True

    # Guard B: Slow-ling contact ≤ 2:00 → 12_pool
    if (first_contact is not None and
            first_contact <= T_CONTACT_SLOW and not has_speed):
        bot._cheese_detected = True
        bot._cheese_label = "12_pool"
        bot._auto_true_fired = True
        bot._cheese_source = "auto-TRUE"
        bot._strategy_chat_pending = "(slow-ling contact)"
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE B): "
              f"Slow-ling contact at {first_contact:.1f}s → 12_pool")
        return True

    # Guard C: Speed-ling contact ≤ 2:40 → speedling
    if (first_contact is not None and
            first_contact <= T_CONTACT_SPEED and has_speed):
        bot._cheese_detected = True
        bot._cheese_label = "speedling"
        bot._auto_true_fired = True
        bot._cheese_source = "auto-TRUE"
        bot._strategy_chat_pending = "(speed-ling contact)"
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE C): "
              f"Speed-ling contact at {first_contact:.1f}s → speedling")
        return True

    # === SCORING ===
    score_12p = 0
    score_speed = 0
    pool_start = getattr(bot, '_pool_seen_time', None)
    pool_state = getattr(bot, '_pool_seen_state', "none")

    # 12_POOL SIGNALS
    if pool_start is not None and T_POOL_12P_START_MIN <= pool_start <= T_POOL_12P_START_MAX:
        score_12p += 4
    if pool_state == "done" and pool_start is not None and pool_start <= T_POOL_12P_START_MAX:
        score_12p += 3

    scouted_nat_late = (
        getattr(bot, '_last_nat_scout_time', None) is not None
        and getattr(bot, '_last_nat_scout_time', None) >= T_NAT_CONFIRM_MISSING
    )
    confirmed_no_nat = (
        getattr(bot, '_nat_present_on_last_scout', None) is False and scouted_nat_late
    )
    if confirmed_no_nat and getattr(bot, '_enemy_nat_started_at', None) is None:
        score_12p += 3

    if first_ling is not None and first_ling <= T_LING_12P_SEEN:
        score_12p += 4

    no_queen = (
        pool_state in {"morphing", "done"}
        and getattr(bot, '_queen_started_time', None) is None
        and time_now >= T_QUEEN_CHECK
    )
    if no_queen:
        score_12p += 2

    # SPEEDLING SIGNALS
    if pool_start is not None and T_POOL_SPEED_START_MIN <= pool_start <= T_POOL_SPEED_START_MAX:
        score_speed += 3

    extractor_time = getattr(bot, '_extractor_seen_time', None)
    gas_workers = getattr(bot, '_gas_workers_count', 0)
    if extractor_time is not None and gas_workers >= 2:
        score_speed += 3

    speed_started = getattr(bot, '_speed_research_started', False)
    speed_time = getattr(bot, '_speed_research_time', None)
    if speed_started and speed_time is not None and speed_time <= T_SPEED_START_EARLY:
        score_speed += 4

    if first_ling is not None and 125.0 <= first_ling <= 135.0:
        score_speed += 2

    if (first_contact is not None and
            first_contact <= T_CONTACT_SPEED and has_speed):
        score_speed += 4

    # DAMPERS
    nat_started = getattr(bot, '_enemy_nat_started_at', None)
    if nat_started is not None and nat_started <= T_NAT_CHECK:
        score_speed -= 1

    queen_time = getattr(bot, '_queen_started_time', None)
    if queen_time is not None and 110.0 <= queen_time <= 120.0:
        score_speed -= 1

    bot._score_12p = score_12p
    bot._score_speed = score_speed

    # === RULE-BASED CLASSIFICATION ===
    if score_12p >= 5:
        bot._cheese_detected = True
        bot._cheese_label = "12_pool"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(rule score={score_12p})"
        print(f"{bot.time_formatted}: Rush detected (rules)! "
              f"12_pool (score={score_12p})")
        return True

    if score_speed >= 5:
        bot._cheese_detected = True
        bot._cheese_label = "speedling"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(rule score={score_speed})"
        print(f"{bot.time_formatted}: Rush detected (rules)! "
              f"speedling (score={score_speed})")
        return True

    bot._cheese_label = "none"
    bot._cheese_source = None
    return False


def _detect_zerg_allin(bot: "PiG_Bot") -> bool:
    """Zerg all-in and timing detection beyond ling rush.

    Uses ARES mediator booleans + timing signals for patterns we track.
    Labels: roach_rush, ravager_push, roach_timing, roach_ravager_push,
            one_base_all_in, two_base_all_in.
    """
    if bot.enemy_race.name not in ("Zerg", "Random"):
        return False

    if not hasattr(bot, '_zerg_allin_detected'):
        bot._zerg_allin_detected = False
        bot._zerg_allin_label = "none"

    if bot._zerg_allin_detected:
        return True

    bases = getattr(bot, '_enemy_bases_count', 0)
    rw_time = getattr(bot, '_roach_warren_seen_time', None)
    spire_time = getattr(bot, '_spire_seen_time', None)
    nat_started = getattr(bot, '_enemy_nat_started_at', None)
    time_now = bot.time

    # === ARES AUTO-TRUE GUARDS ===

    # ARES roach rush
    if bot.mediator.get_enemy_roach_rushed:
        bot._zerg_allin_detected = True
        bot._zerg_allin_label = "roach_rush"
        bot._cheese_source = "auto-TRUE:ARES"
        bot._strategy_chat_pending = "(roach rush detected)"
        print(f"{bot.time_formatted}: Zerg all-in detected (ARES): roach_rush")
        return True

    # ARES ravager rush
    if bot.mediator.get_enemy_ravager_rush:
        bot._zerg_allin_detected = True
        bot._zerg_allin_label = "ravager_push"
        bot._cheese_source = "auto-TRUE:ARES"
        bot._strategy_chat_pending = "(ravager rush detected)"
        print(f"{bot.time_formatted}: Zerg all-in detected (ARES): ravager_push")
        return True

    # === RULE-BASED DETECTION ===

    # roach_rush: roach warren early + 1 base
    if rw_time is not None and rw_time < 180.0 and bases == 1:
        bot._zerg_allin_detected = True
        bot._zerg_allin_label = "roach_rush"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(roach warren early + 1 base)"
        print(f"{bot.time_formatted}: Zerg all-in detected (rules): "
              f"roach_rush (rw@{rw_time:.0f}s, bases={bases})")
        return True

    # ravager_rush: ravagers seen + 1 base
    ravager_seen = any(
        u.type_id == UnitTypeId.RAVAGER for u in bot.enemy_units
    )
    if ravager_seen and bases == 1 and time_now < 420.0:
        bot._zerg_allin_detected = True
        bot._zerg_allin_label = "ravager_push"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(ravager + 1 base)"
        print(f"{bot.time_formatted}: Zerg all-in detected (rules): "
              f"ravager_push (bases={bases})")
        return True

    # roach_ravager_push: roach warren < 270s + 2 bases + no Spire
    if (rw_time is not None and rw_time < 270.0
            and bases == 2 and spire_time is None and time_now > 180.0):
        bot._zerg_allin_detected = True
        bot._zerg_allin_label = "roach_ravager_push"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(2-base roach/ravager push)"
        print(f"{bot.time_formatted}: Zerg all-in detected (rules): "
              f"roach_ravager_push (rw@{rw_time:.0f}s, bases={bases})")
        return True

    # one_base_all_in: 1 base + game > 4min + no natural
    if bases == 1 and time_now > 240.0 and nat_started is None:
        scouted = getattr(bot, '_last_nat_scout_time', None)
        if scouted is not None and scouted > 180.0:
            bot._zerg_allin_detected = True
            bot._zerg_allin_label = "one_base_all_in"
            bot._cheese_source = "rules"
            bot._strategy_chat_pending = "(1 base all-in)"
            print(f"{bot.time_formatted}: Zerg all-in detected (rules): "
              f"one_base_all_in (bases={bases}, t={time_now:.0f}s)")
            return True

    # two_base_all_in: 2 bases + roach warren + no Spire + early push
    if (bases == 2 and rw_time is not None and spire_time is None
            and time_now > 240.0 and time_now < 600.0
            and nat_started is not None and nat_started < 120.0):
        bot._zerg_allin_detected = True
        bot._zerg_allin_label = "two_base_all_in"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(2 base all-in)"
        print(f"{bot.time_formatted}: Zerg all-in detected (rules): "
              f"two_base_all_in (bases={bases}, t={time_now:.0f}s)")
        return True

    # roach_timing: roach warren 150-270s + ≥2 bases (not all-in)
    if (rw_time is not None and 150.0 <= rw_time <= 270.0
            and bases >= 2 and spire_time is None
            and not bot._zerg_allin_detected):
        # This is a timing attack, not an all-in — set label but don't fire
        # detect_cheese() returns True for non-macro strategies
        if not hasattr(bot, '_zerg_timing_label'):
            bot._zerg_timing_label = "roach_timing"
            bot._strategy_chat_pending = "(roach timing)"
            print(f"{bot.time_formatted}: Zerg timing detected (rules): "
                  f"roach_timing (rw@{rw_time:.0f}s, bases={bases})")
            return True

    bot._zerg_allin_label = "none"
    return False


# ── TERRAN DETECTOR ────────────────────────────────────────────────────────

def _detect_terran_strategy(bot: "PiG_Bot") -> bool:
    """Terran strategy detection — proxy rax, bunker rush, all-in, timing.

    Labels: proxy_rax, bunker_rush, marauder_push, marine_rush,
            all_in, one_base_all_in, two_base_all_in,
            bio_timing, tank_timing, widow_mine_drop.
    """
    if not hasattr(bot, '_terran_strategy_detected'):
        bot._terran_strategy_detected = False
        bot._terran_strategy_label = "none"

    if bot._terran_strategy_detected:
        return True

    rax_time = getattr(bot, '_barracks_seen_time', None)
    rax_count = getattr(bot, '_barracks_count', 0)
    rax_proxy = getattr(bot, '_barracks_near_our_base', False)
    bunker_near = getattr(bot, '_bunker_near_base', False)
    bunker_time = getattr(bot, '_bunker_seen_time', None)
    nat_started = getattr(bot, '_enemy_nat_started_at', None)
    gas_time = getattr(bot, '_enemy_gas_seen_time', None)
    gas_workers = getattr(bot, '_enemy_gas_workers_count', 0)
    factory_time = getattr(bot, '_factory_seen_time', None)
    factory_count = getattr(bot, '_factory_count', 0)
    starport_time = getattr(bot, '_starport_seen_time', None)
    bases = getattr(bot, '_enemy_bases_count', 0)
    marauder_time = getattr(bot, '_marauder_seen_time', None)
    medivac_time = getattr(bot, '_medivac_seen_time', None)
    tank_time = getattr(bot, '_siege_tank_seen_time', None)
    mine_time = getattr(bot, '_widow_mine_seen_time', None)
    stim_time = getattr(bot, '_stimpack_seen_time', None)
    time_now = bot.time

    # === AUTO-TRUE GUARDS ===

    # Guard A: Proxy barracks near our base
    if rax_proxy and rax_time is not None and rax_time < 90.0:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "proxy_rax"
        bot._cheese_source = "auto-TRUE"
        bot._strategy_chat_pending = "(proxy barracks detected)"
        print(f"{bot.time_formatted}: Terran strategy detected (Auto-TRUE): "
              f"Proxy rax at {rax_time:.1f}s")
        return True

    # Guard B: Bunker near our base early
    if bunker_near and bunker_time is not None and bunker_time < 120.0:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "bunker_rush"
        bot._cheese_source = "auto-TRUE"
        bot._strategy_chat_pending = "(bunker rush detected)"
        print(f"{bot.time_formatted}: Terran strategy detected (Auto-TRUE): "
              f"Bunker rush at {bunker_time:.1f}s")
        return True

    # Guard C: ARES marauder rush
    if bot.mediator.get_enemy_marauder_rush and time_now < 150.0:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "marauder_push"
        bot._cheese_source = "auto-TRUE:ARES"
        bot._strategy_chat_pending = "(marauder rush detected)"
        print(f"{bot.time_formatted}: Terran strategy detected (Auto-TRUE): "
              f"marauder_push (ARES)")
        return True

    # === RULE-BASED DETECTION ===

    # marine_rush: ≥2 barracks + 1 base + early, but NOT proxy (proxies → proxy_rax via scoring)
    if rax_count >= 2 and bases == 1 and time_now < 300.0 and not rax_proxy:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "marine_rush"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(marine rush: rax + 1 base)"
        print(f"{bot.time_formatted}: Terran strategy detected (rules): "
              f"marine_rush (rax={rax_count}, bases={bases})")
        return True

    # one_base_all_in: 1 base + game > 4min + no natural
    if bases == 1 and time_now > 240.0 and nat_started is None:
        scouted = getattr(bot, '_last_nat_scout_time', None)
        if scouted is not None and scouted > 180.0:
            bot._terran_strategy_detected = True
            bot._terran_strategy_label = "one_base_all_in"
            bot._cheese_source = "rules"
            bot._strategy_chat_pending = "(1 base all-in)"
            print(f"{bot.time_formatted}: Terran strategy detected (rules): "
                  f"one_base_all_in (bases={bases}, t={time_now:.0f}s)")
            return True

    # two_base_all_in: 2 bases + lots of production + early
    if bases == 2 and time_now < 600.0 and rax_count >= 4:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "two_base_all_in"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(2 base all-in)"
        print(f"{bot.time_formatted}: Terran strategy detected (rules): "
              f"two_base_all_in (rax={rax_count}, bases={bases})")
        return True

    # === SCORING (cheese-level) ===
    score_proxy = 0
    score_bunker = 0
    score_allin = 0

    # Proxy rax signals
    if rax_proxy:
        score_proxy += 5
    if rax_time is not None and rax_time < 65.0:  # Rax before 1:05
        score_proxy += 3
    if rax_count >= 2 and rax_proxy:
        score_proxy += 2

    # Bunker rush signals
    if bunker_near:
        score_bunker += 4
    if bunker_time is not None and bunker_time < 120.0:
        score_bunker += 2

    # All-in signals (no natural, early gas, many rax)
    if nat_started is None and time_now > 120.0:
        scouted = getattr(bot, '_last_nat_scout_time', None)
        if scouted is not None and scouted > 105.0:
            score_allin += 3
    if gas_time is not None and gas_time < 60.0 and gas_workers >= 2:
        score_allin += 2
    if rax_count >= 3 and nat_started is None and time_now < 180.0:
        score_allin += 3

    # === CLASSIFICATION (cheese-level) ===
    if score_proxy >= 5:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "proxy_rax"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(proxy score={score_proxy})"
        print(f"{bot.time_formatted}: Terran strategy detected (rules): "
              f"proxy_rax (score={score_proxy})")
        return True

    if score_bunker >= 5:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "bunker_rush"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(bunker score={score_bunker})"
        print(f"{bot.time_formatted}: Terran strategy detected (rules): "
              f"bunker_rush (score={score_bunker})")
        return True

    if score_allin >= 5:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "all_in"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(all-in score={score_allin})"
        print(f"{bot.time_formatted}: Terran strategy detected (rules): "
              f"all_in (score={score_allin})")
        return True

    # === TIMING ATTACK DETECTION ===
    # These return True (non-macro) but set timing labels

    # bio_timing: ≥3 barracks + medivacs
    if rax_count >= 3 and medivac_time is not None:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "bio_timing"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(bio timing: rax + medivac)"
        print(f"{bot.time_formatted}: Terran timing detected (rules): "
              f"bio_timing (rax={rax_count}, medivac@{medivac_time:.0f}s)")
        return True

    # tank_timing: factory + siege tanks
    if factory_count > 0 and tank_time is not None:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "tank_timing"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(tank timing)"
        print(f"{bot.time_formatted}: Terran timing detected (rules): "
              f"tank_timing (factory + tanks)")
        return True

    # widow_mine_drop: starport + widow mines
    if starport_time is not None and mine_time is not None:
        bot._terran_strategy_detected = True
        bot._terran_strategy_label = "widow_mine_drop"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(widow mine drop)"
        print(f"{bot.time_formatted}: Terran timing detected (rules): "
              f"widow_mine_drop (starport + mines)")
        return True

    bot._terran_strategy_label = "none"
    return False


# ── PROTOSS DETECTOR ───────────────────────────────────────────────────────

def _detect_protoss_strategy(bot: "PiG_Bot") -> bool:
    """Protoss strategy detection — cannon rush, proxy gates, four-gate, all-in, timing.

    Labels: cannon_rush, proxy_gate, four_gate, six_gate, all_in,
            two_base_colossus, two_base_all_in, stargate_timing.
    """
    if not hasattr(bot, '_protoss_strategy_detected'):
        bot._protoss_strategy_detected = False
        bot._protoss_strategy_label = "none"

    if bot._protoss_strategy_detected:
        return True

    gw_time = getattr(bot, '_gateway_seen_time', None)
    gw_count = getattr(bot, '_gateway_count', 0)
    wg_count = getattr(bot, '_warpgate_count', 0)
    gw_proxy = getattr(bot, '_gateway_near_our_base', False)
    forge_time = getattr(bot, '_forge_seen_time', None)
    cannon_time = getattr(bot, '_cannon_seen_time', None)
    core_time = getattr(bot, '_cyber_core_seen_time', None)
    nat_started = getattr(bot, '_enemy_nat_started_at', None)
    gas_time = getattr(bot, '_enemy_gas_seen_time', None)
    gas_workers = getattr(bot, '_enemy_gas_workers_count', 0)
    stargate_time = getattr(bot, '_stargate_seen_time', None)
    robo_time = getattr(bot, '_robotics_facility_seen_time', None)
    robo_bay_time = getattr(bot, '_robotics_bay_seen_time', None)
    bases = getattr(bot, '_enemy_bases_count', 0)
    time_now = bot.time
    total_gates = gw_count + wg_count

    # === AUTO-TRUE GUARDS ===

    # Guard A: Cannon rush — forge + cannon near our base early
    if cannon_time is not None and cannon_time < 150.0:
        # Check if cannon is near our base
        cannons = [s for s in bot.enemy_structures
                   if s.type_id == UnitTypeId.PHOTONCANNON]
        our_nat = bot.mediator.get_own_nat
        cannon_near = any(
            cy_distance_to(c.position, bot.start_location) < 25.0
            or cy_distance_to(c.position, our_nat) < 25.0
            for c in cannons
        )
        if cannon_near:
            bot._protoss_strategy_detected = True
            bot._protoss_strategy_label = "cannon_rush"
            bot._cheese_source = "auto-TRUE"
            bot._strategy_chat_pending = "(cannon rush detected)"
            print(f"{bot.time_formatted}: Protoss strategy detected (Auto-TRUE): "
                  f"cannon_rush at {cannon_time:.1f}s")
            return True

    # Guard B: Proxy gateways near our base
    if gw_proxy and gw_time is not None and gw_time < 80.0:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "proxy_gate"
        bot._cheese_source = "auto-TRUE"
        bot._strategy_chat_pending = "(proxy gates detected)"
        print(f"{bot.time_formatted}: Protoss strategy detected (Auto-TRUE): "
              f"proxy_gate at {gw_time:.1f}s")
        return True

    # Guard C: ARES proxy zealot
    if bot.mediator.get_is_proxy_zealot:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "proxy_gate"
        bot._cheese_source = "auto-TRUE:ARES"
        bot._strategy_chat_pending = "(proxy zealot detected)"
        print(f"{bot.time_formatted}: Protoss strategy detected (Auto-TRUE): "
              f"proxy_gate (ARES)")
        return True

    # Guard D: ARES four gate
    if bot.mediator.get_enemy_four_gate:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "four_gate"
        bot._cheese_source = "auto-TRUE:ARES"
        bot._strategy_chat_pending = "(four gate detected)"
        print(f"{bot.time_formatted}: Protoss strategy detected (Auto-TRUE): "
              f"four_gate (ARES)")
        return True

    # === RULE-BASED DETECTION ===

    # four_gate: ≥4 gates + 1 base
    if total_gates >= 4 and bases == 1:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "four_gate"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(4+ gates + 1 base)"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"four_gate (gates={total_gates}, bases={bases})")
        return True

    # six_gate: ≥6 gates + 2 bases
    if total_gates >= 6 and bases == 2:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "six_gate"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(6+ gates + 2 bases)"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"six_gate (gates={total_gates}, bases={bases})")
        return True

    # two_base_colossus: robotics bay + 2 bases + early
    if robo_bay_time is not None and bases == 2 and time_now < 480.0:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "two_base_colossus"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(2-base colossus)"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"two_base_colossus (robo_bay@{robo_bay_time:.0f}s)")
        return True

    # two_base_all_in: 2 bases + lots of gates + early
    if bases == 2 and time_now < 600.0 and total_gates >= 4:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "two_base_all_in"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(2 base all-in)"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"two_base_all_in (gates={total_gates}, bases={bases})")
        return True

    # one_base_all_in: 1 base + game > 4min + no natural
    if bases == 1 and time_now > 240.0 and nat_started is None:
        scouted = getattr(bot, '_last_nat_scout_time', None)
        if scouted is not None and scouted > 180.0:
            bot._protoss_strategy_detected = True
            bot._protoss_strategy_label = "one_base_all_in"
            bot._cheese_source = "rules"
            bot._strategy_chat_pending = "(1 base all-in)"
            print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
                  f"one_base_all_in (bases={bases}, t={time_now:.0f}s)")
            return True

    # === SCORING (cheese-level) ===
    score_cannon = 0
    score_proxy = 0
    score_four_gate = 0
    score_allin = 0

    # Cannon rush signals
    if forge_time is not None and forge_time < 60.0:
        score_cannon += 3
    if cannon_time is not None and cannon_time < 150.0:
        score_cannon += 3
    # Forge before gateway = strong cannon rush indicator
    if (forge_time is not None and gw_time is not None
            and forge_time < gw_time):
        score_cannon += 3

    # Proxy gate signals
    if gw_proxy:
        score_proxy += 4
    if gw_time is not None and gw_time < 65.0:
        score_proxy += 3
    if gw_count >= 2 and gw_proxy:
        score_proxy += 2

    # Four-gate signals (4+ gates, no natural, early gas)
    if gw_count >= 4 and time_now < 300.0:
        score_four_gate += 3
    if gw_count >= 4 and nat_started is None and time_now > 180.0:
        score_four_gate += 3
    if gas_time is not None and gas_time < 80.0 and gas_workers >= 2:
        score_four_gate += 2
    # No stargate/robo by 4:00 with 4+ gates = four-gate commitment
    if (gw_count >= 4 and time_now > 240.0
            and stargate_time is None and robo_time is None):
        score_four_gate += 2

    # All-in signals (no natural, early gas, many gates)
    if nat_started is None and time_now > 150.0:
        scouted = getattr(bot, '_last_nat_scout_time', None)
        if scouted is not None and scouted > 105.0:
            score_allin += 2
    if gw_count >= 3 and nat_started is None and time_now < 180.0:
        score_allin += 3

    # === CLASSIFICATION (cheese-level) ===
    if score_cannon >= 5:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "cannon_rush"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(cannon score={score_cannon})"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"cannon_rush (score={score_cannon})")
        return True

    if score_proxy >= 5:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "proxy_gate"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(proxy score={score_proxy})"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"proxy_gate (score={score_proxy})")
        return True

    if score_four_gate >= 5:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "four_gate"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(4gate score={score_four_gate})"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"four_gate (score={score_four_gate})")
        return True

    if score_allin >= 5:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "all_in"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = f"(all-in score={score_allin})"
        print(f"{bot.time_formatted}: Protoss strategy detected (rules): "
              f"all_in (score={score_allin})")
        return True

    # === TIMING ATTACK DETECTION ===

    # stargate_timing: stargate early
    if stargate_time is not None and stargate_time < 270.0:
        bot._protoss_strategy_detected = True
        bot._protoss_strategy_label = "stargate_timing"
        bot._cheese_source = "rules"
        bot._strategy_chat_pending = "(stargate timing)"
        print(f"{bot.time_formatted}: Protoss timing detected (rules): "
              f"stargate_timing (stargate@{stargate_time:.0f}s)")
        return True

    bot._protoss_strategy_label = "none"
    return False


# ── BACKWARD-COMPATIBLE HELPERS ─────────────────────────────────────────────

def get_enemy_cannon_rushed(bot: "PiG_Bot", detection_radius: float = 25.0) -> bool:
    """Check if the enemy is cannon rushing.

    Backward-compatible wrapper that delegates to the Protoss detector.
    The detection_radius parameter is kept for API compatibility but the
    actual detection now uses the full Protoss strategy classifier.
    """
    if bot.enemy_race != Race.Protoss:
        return False

    # If Protoss detector has already classified as cannon_rush, return True
    if getattr(bot, '_protoss_strategy_label', 'none') == 'cannon_rush':
        return True

    # Run the Protoss detector if not yet triggered
    _detect_protoss_strategy(bot)
    return getattr(bot, '_protoss_strategy_label', 'none') == 'cannon_rush'