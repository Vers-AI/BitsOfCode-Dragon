"""Enemy timing observations — record what we see and when.

Purpose: Race-agnostic observation layer. Tracks building start times, unit
         first-seen times, expansion timings, and proxy detection for ALL races.
         No classification logic lives here — this only records facts.

Key Decisions: Location is first-class — every structure gets a near_our_base
               boolean for proxy detection. Absence is meaningful — pair with
               _last_nat_scout_time so "didn't see X" is a real signal.

Limitations: Only records what we actually see. If we haven't scouted an area,
             we can't distinguish "not built yet" from "built but not seen".
"""

from typing import TYPE_CHECKING

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from cython_extensions import cy_distance_to

if TYPE_CHECKING:
    from bot.bot import PiG_Bot

# Distance threshold for "near our base" proxy detection (game units)
_PROXY_DISTANCE_THRESHOLD = 55.0


def _estimate_building_start_time(bot: "PiG_Bot", building) -> float:
    """Estimate when a building started based on current build progress.

    Formula: start_time = current_time - (build_progress * build_duration / 22.4)
    22.4 = game loops per second at Faster speed.
    """
    build_duration = building._type_data.cost.time
    build_progress = building.build_progress
    elapsed_build_time = (build_progress * build_duration) / 22.4
    return bot.time - elapsed_build_time


def _is_near_our_base(bot: "PiG_Bot", position: Point2) -> bool:
    """Check if a position is close enough to our base to be a proxy."""
    return cy_distance_to(position, bot.start_location) < _PROXY_DISTANCE_THRESHOLD


def track_enemy_timings(bot: "PiG_Bot") -> None:
    """Track enemy building/unit timings for ALL races.

    Records timestamps on bot object. No classification — just observations.
    Called once per frame during the build phase.

    Cross-cutting attributes (all races):
        _enemy_nat_started_at, _last_nat_scout_time, _nat_present_on_last_scout
        _enemy_worker_count, _enemy_worker_count_history, _enemy_bases_count
        _enemy_army_supply, _enemy_gas_workers_count, _enemy_gas_seen_time
        _structures_near_our_base, _enemy_gas_on_our_base_time

    Zerg attributes:
        _pool_seen_time, _pool_seen_state, _extractor_seen_time, _queen_started_time
        _first_ling_seen_time, _first_ling_contact_nat_time, _ling_has_speed
        _ling_pos_history, _baneling_nest_seen_time, _speed_research_started
        _speed_research_time, _gas_workers_count

    Terran attributes:
        _barracks_seen_time, _barracks_count, _barracks_near_our_base
        _factory_seen_time, _factory_count, _starport_seen_time, _starport_count
        _bunker_near_base, _bunker_seen_time

    Protoss attributes:
        _gateway_seen_time, _gateway_count, _gateway_near_our_base
        _forge_seen_time, _cyber_core_seen_time, _cannon_seen_time
        _stargate_seen_time, _robotics_facility_seen_time, _robotics_bay_seen_time
        _warpgate_count
    """
    _init_cross_cutting_attrs(bot)
    _track_cross_cutting(bot)

    if bot.enemy_race.name == "Zerg" or (
        bot.enemy_race.name == "Random" and not bot.mediator.get_enemy_nat
    ):
        _init_zerg_attrs(bot)
        _track_zerg(bot)

    if bot.enemy_race.name == "Terran":
        _init_terran_attrs(bot)
        _track_terran(bot)

    if bot.enemy_race.name == "Protoss":
        _init_protoss_attrs(bot)
        _track_protoss(bot)

    # For Random opponents, once we see race-specific structures, track those too
    if bot.enemy_race.name == "Random":
        _init_zerg_attrs(bot)
        _init_terran_attrs(bot)
        _init_protoss_attrs(bot)
        _track_zerg(bot)
        _track_terran(bot)
        _track_protoss(bot)


# ── CROSS-CUTTING (all races) ──────────────────────────────────────────────

def _init_cross_cutting_attrs(bot: "PiG_Bot") -> None:
    """Initialize cross-cutting tracking attributes (all races)."""
    if not hasattr(bot, '_enemy_nat_started_at'):
        bot._enemy_nat_started_at = None
    if not hasattr(bot, '_last_nat_scout_time'):
        bot._last_nat_scout_time = None
    if not hasattr(bot, '_nat_present_on_last_scout'):
        bot._nat_present_on_last_scout = None
    if not hasattr(bot, '_enemy_worker_count'):
        bot._enemy_worker_count = 0
    if not hasattr(bot, '_enemy_worker_count_history'):
        bot._enemy_worker_count_history = []  # [(time, count), ...]
    if not hasattr(bot, '_enemy_bases_count'):
        bot._enemy_bases_count = 0
    if not hasattr(bot, '_enemy_army_supply'):
        bot._enemy_army_supply = 0
    if not hasattr(bot, '_enemy_gas_workers_count'):
        bot._enemy_gas_workers_count = 0
    if not hasattr(bot, '_enemy_gas_seen_time'):
        bot._enemy_gas_seen_time = None
    if not hasattr(bot, '_structures_near_our_base'):
        bot._structures_near_our_base = []  # [(unit_type, time, distance)]
    if not hasattr(bot, '_enemy_gas_on_our_base_time'):
        bot._enemy_gas_on_our_base_time = None


def _track_cross_cutting(bot: "PiG_Bot") -> None:
    """Track observations that apply to all races."""
    enemy_nat_pos = bot.mediator.get_enemy_nat

    # ── Natural expansion tracking (all races) ──
    if bot.is_visible(enemy_nat_pos):
        bot._last_nat_scout_time = bot.time
        # Check for any enemy townhall at natural position
        nat_townhalls = [
            s for s in bot.enemy_structures
            if s.type_id in {
                UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
                UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND,
                UnitTypeId.PLANETARYFORTRESS,
                UnitTypeId.NEXUS,
            }
            and cy_distance_to(s.position, enemy_nat_pos) < 10
        ]
        bot._nat_present_on_last_scout = len(nat_townhalls) > 0

    if bot._enemy_nat_started_at is None:
        nat_townhalls = [
            s for s in bot.enemy_structures
            if s.type_id in {
                UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
                UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND,
                UnitTypeId.PLANETARYFORTRESS,
                UnitTypeId.NEXUS,
            }
            and cy_distance_to(s.position, enemy_nat_pos) < 10
        ]
        if nat_townhalls:
            bot._enemy_nat_started_at = _estimate_building_start_time(bot, nat_townhalls[0])

    # ── Worker count tracking (all races) ──
    from ares.consts import WORKER_TYPES
    enemy_workers = [u for u in bot.enemy_units if u.type_id in WORKER_TYPES]
    bot._enemy_worker_count = len(enemy_workers)
    # Rolling history for delta detection (cap at 60 samples)
    bot._enemy_worker_count_history.append((bot.time, len(enemy_workers)))
    if len(bot._enemy_worker_count_history) > 60:
        bot._enemy_worker_count_history = bot._enemy_worker_count_history[-60:]

    # ── Base count (all races) ──
    from sc2.ids.unit_typeid import UnitTypeId as UID
    townhall_types = {
        UID.HATCHERY, UID.LAIR, UID.HIVE,
        UID.COMMANDCENTER, UID.ORBITALCOMMAND, UID.PLANETARYFORTRESS,
        UID.NEXUS,
    }
    bot._enemy_bases_count = sum(
        1 for s in bot.enemy_structures if s.type_id in townhall_types
    )

    # ── Army supply (all races) ──
    enemy_army = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES and not u.is_structure
    ]
    bot._enemy_army_supply = sum(u.food_used for u in enemy_army if hasattr(u, 'food_used') and u.food_used)

    # ── Gas tracking (all races) ──
    gas_types = {UnitTypeId.EXTRACTOR, UnitTypeId.REFINERY, UnitTypeId.ASSIMILATOR}
    gas_structures = [s for s in bot.enemy_structures if s.type_id in gas_types]
    if gas_structures and bot._enemy_gas_seen_time is None:
        bot._enemy_gas_seen_time = _estimate_building_start_time(bot, gas_structures[0])

    # Count workers on gas (near any enemy gas structure)
    ready_gas = [s for s in gas_structures if s.is_ready]
    if ready_gas:
        worker_types = {UnitTypeId.DRONE, UnitTypeId.SCV, UnitTypeId.PROBE}
        gas_workers = [
            w for w in bot.enemy_units
            if w.type_id in worker_types
            and cy_distance_to(w.position, ready_gas[0].position) < 3
        ]
        bot._enemy_gas_workers_count = len(gas_workers)

    # ── Gas steal detection ──
    if bot._enemy_gas_on_our_base_time is None:
        our_geysers = bot.vespene_geyser.closer_than(15, bot.start_location)
        for geyser in our_geysers:
            steal_structures = [
                s for s in bot.enemy_structures
                if s.type_id in gas_types
                and cy_distance_to(s.position, geyser.position) < 3
            ]
            if steal_structures:
                bot._enemy_gas_on_our_base_time = bot.time
                break

    # ── Proxy structure detection (all races) ──
    # Track any enemy structure built suspiciously close to our base
    proxy_types = {
        UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT,
        UnitTypeId.GATEWAY, UnitTypeId.WARPGATE,
        UnitTypeId.PYLON, UnitTypeId.FORGE,
        UnitTypeId.BUNKER,
        UnitTypeId.SPAWNINGPOOL,
        UnitTypeId.HATCHERY,  # Proxy hatch
    }
    for s in bot.enemy_structures:
        if s.type_id in proxy_types:
            dist = cy_distance_to(s.position, bot.start_location)
            if dist < _PROXY_DISTANCE_THRESHOLD:
                # Only add if not already tracked
                already_tracked = any(
                    t == s.type_id and abs(d - dist) < 1.0
                    for t, _, d in bot._structures_near_our_base
                )
                if not already_tracked:
                    bot._structures_near_our_base.append(
                        (s.type_id, bot.time, dist)
                    )


# ── ZERG ───────────────────────────────────────────────────────────────────

def _init_zerg_attrs(bot: "PiG_Bot") -> None:
    """Initialize Zerg-specific tracking attributes."""
    if not hasattr(bot, '_pool_seen_state'):
        bot._pool_seen_state = "none"
    if not hasattr(bot, '_pool_seen_time'):
        bot._pool_seen_time = None
    if not hasattr(bot, '_extractor_seen_time'):
        bot._extractor_seen_time = None
    if not hasattr(bot, '_queen_started_time'):
        bot._queen_started_time = None
    if not hasattr(bot, '_first_ling_seen_time'):
        bot._first_ling_seen_time = None
    if not hasattr(bot, '_first_ling_contact_nat_time'):
        bot._first_ling_contact_nat_time = None
    if not hasattr(bot, '_gas_workers_count'):
        bot._gas_workers_count = 0
    if not hasattr(bot, '_speed_research_started'):
        bot._speed_research_started = False
    if not hasattr(bot, '_speed_research_time'):
        bot._speed_research_time = None
    if not hasattr(bot, '_baneling_nest_seen_time'):
        bot._baneling_nest_seen_time = None
    if not hasattr(bot, '_ling_has_speed'):
        bot._ling_has_speed = False
    if not hasattr(bot, '_ling_pos_history'):
        bot._ling_pos_history = {}


def _track_zerg(bot: "PiG_Bot") -> None:
    """Track Zerg-specific enemy timings."""
    # ── Spawning pool ──
    pools = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.SPAWNINGPOOL]
    if pools and bot._pool_seen_state == "none":
        pool = pools[0]
        bot._pool_seen_time = _estimate_building_start_time(bot, pool)
        bot._pool_seen_state = "done" if pool.is_ready else "morphing"
    elif pools and bot._pool_seen_state == "morphing":
        if pools[0].is_ready:
            bot._pool_seen_state = "done"

    # ── Extractor (first one) ──
    if bot._extractor_seen_time is None:
        extractors = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.EXTRACTOR]
        if extractors:
            bot._extractor_seen_time = _estimate_building_start_time(bot, extractors[0])

    # ── Queen (first one) ──
    if bot._queen_started_time is None:
        queens = [u for u in bot.enemy_units if u.type_id == UnitTypeId.QUEEN]
        if queens:
            bot._queen_started_time = bot.time

    # ── Zerglings ──
    enemy_lings = bot.mediator.get_enemy_army_dict.get(UnitTypeId.ZERGLING, [])
    if enemy_lings:
        if bot._first_ling_seen_time is None:
            bot._first_ling_seen_time = bot.time

        if bot._first_ling_contact_nat_time is None:
            our_nat_pos = bot.mediator.get_own_nat
            lings_at_nat = [
                ling for ling in enemy_lings
                if cy_distance_to(ling.position, our_nat_pos) < 15
            ]
            if lings_at_nat:
                bot._first_ling_contact_nat_time = bot.time

        # Detect Metabolic Boost via position deltas
        # Base speed=4.13, base+creep=5.37, speed=6.58
        if not bot._ling_has_speed:
            now = bot.time
            for ling in enemy_lings:
                if not ling.is_visible:
                    continue
                tag = ling.tag
                pos = ling.position
                if tag in bot._ling_pos_history:
                    prev_pos, prev_time = bot._ling_pos_history[tag]
                    dt = now - prev_time
                    if dt >= 0.3:
                        dist = cy_distance_to(pos, prev_pos)
                        observed_speed = dist / dt
                        if observed_speed > 5.5:  # Above base+creep(5.37)
                            bot._ling_has_speed = True
                            if bot.debug:
                                print(f"{bot.time_formatted}: Speed detected! "
                                      f"Ling {tag} speed={observed_speed:.2f}")
                            break
                bot._ling_pos_history[tag] = (pos, now)
            if len(bot._ling_pos_history) > 50:
                bot._ling_pos_history = {
                    t: v for t, v in bot._ling_pos_history.items()
                    if now - v[1] < 10.0
                }

    # ── Gas workers (Zerg-specific: drones on extractor) ──
    extractors = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.EXTRACTOR and s.is_ready]
    if extractors:
        workers_on_gas = [
            w for w in bot.enemy_units
            if w.type_id == UnitTypeId.DRONE and cy_distance_to(w.position, extractors[0].position) < 3
        ]
        bot._gas_workers_count = len(workers_on_gas)

    # ── Speed research (Metabolic Boost) ──
    if not bot._speed_research_started and pools:
        pool = pools[0]
        if pool.is_ready and pool.is_active:
            bot._speed_research_started = True
            bot._speed_research_time = bot.time

    # ── Baneling nest ──
    if bot._baneling_nest_seen_time is None:
        nests = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.BANELINGNEST]
        if nests:
            bot._baneling_nest_seen_time = bot.time


# ── TERRAN ──────────────────────────────────────────────────────────────────

def _init_terran_attrs(bot: "PiG_Bot") -> None:
    """Initialize Terran-specific tracking attributes."""
    if not hasattr(bot, '_barracks_seen_time'):
        bot._barracks_seen_time = None
    if not hasattr(bot, '_barracks_count'):
        bot._barracks_count = 0
    if not hasattr(bot, '_barracks_near_our_base'):
        bot._barracks_near_our_base = False
    if not hasattr(bot, '_factory_seen_time'):
        bot._factory_seen_time = None
    if not hasattr(bot, '_factory_count'):
        bot._factory_count = 0
    if not hasattr(bot, '_starport_seen_time'):
        bot._starport_seen_time = None
    if not hasattr(bot, '_starport_count'):
        bot._starport_count = 0
    if not hasattr(bot, '_bunker_near_base'):
        bot._bunker_near_base = False
    if not hasattr(bot, '_bunker_seen_time'):
        bot._bunker_seen_time = None


def _track_terran(bot: "PiG_Bot") -> None:
    """Track Terran-specific enemy timings."""
    # ── Barracks ──
    rax = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.BARRACKS]
    bot._barracks_count = len(rax)
    if rax and bot._barracks_seen_time is None:
        bot._barracks_seen_time = _estimate_building_start_time(bot, rax[0])
    # Check for proxy barracks
    if rax and not bot._barracks_near_our_base:
        bot._barracks_near_our_base = any(
            _is_near_our_base(bot, r.position) for r in rax
        )

    # ── Factory ──
    factories = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.FACTORY]
    bot._factory_count = len(factories)
    if factories and bot._factory_seen_time is None:
        bot._factory_seen_time = _estimate_building_start_time(bot, factories[0])

    # ── Starport ──
    starports = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.STARPORT]
    bot._starport_count = len(starports)
    if starports and bot._starport_seen_time is None:
        bot._starport_seen_time = _estimate_building_start_time(bot, starports[0])

    # ── Bunker ──
    bunkers = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.BUNKER]
    if bunkers:
        if bot._bunker_seen_time is None:
            bot._bunker_seen_time = bot.time
        if not bot._bunker_near_base:
            our_nat = bot.mediator.get_own_nat
            bot._bunker_near_base = any(
                cy_distance_to(b.position, our_nat) < 15
                or cy_distance_to(b.position, bot.start_location) < 25
                for b in bunkers
            )


# ── PROTOSS ────────────────────────────────────────────────────────────────

def _init_protoss_attrs(bot: "PiG_Bot") -> None:
    """Initialize Protoss-specific tracking attributes."""
    if not hasattr(bot, '_gateway_seen_time'):
        bot._gateway_seen_time = None
    if not hasattr(bot, '_gateway_count'):
        bot._gateway_count = 0
    if not hasattr(bot, '_gateway_near_our_base'):
        bot._gateway_near_our_base = False
    if not hasattr(bot, '_forge_seen_time'):
        bot._forge_seen_time = None
    if not hasattr(bot, '_cyber_core_seen_time'):
        bot._cyber_core_seen_time = None
    if not hasattr(bot, '_cannon_seen_time'):
        bot._cannon_seen_time = None
    if not hasattr(bot, '_stargate_seen_time'):
        bot._stargate_seen_time = None
    if not hasattr(bot, '_robotics_facility_seen_time'):
        bot._robotics_facility_seen_time = None
    if not hasattr(bot, '_robotics_bay_seen_time'):
        bot._robotics_bay_seen_time = None
    if not hasattr(bot, '_warpgate_count'):
        bot._warpgate_count = 0


def _track_protoss(bot: "PiG_Bot") -> None:
    """Track Protoss-specific enemy timings."""
    # ── Gateways + Warpgates ──
    gateways = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.GATEWAY]
    warpgates = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.WARPGATE]
    bot._gateway_count = len(gateways)
    bot._warpgate_count = len(warpgates)
    if gateways and bot._gateway_seen_time is None:
        bot._gateway_seen_time = _estimate_building_start_time(bot, gateways[0])
    # Check for proxy gateways
    if gateways and not bot._gateway_near_our_base:
        bot._gateway_near_our_base = any(
            _is_near_our_base(bot, g.position) for g in gateways
        )

    # ── Forge ──
    forges = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.FORGE]
    if forges and bot._forge_seen_time is None:
        bot._forge_seen_time = _estimate_building_start_time(bot, forges[0])

    # ── Cyber Core ──
    cores = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.CYBERNETICSCORE]
    if cores and bot._cyber_core_seen_time is None:
        bot._cyber_core_seen_time = _estimate_building_start_time(bot, cores[0])

    # ── Photon Cannon ──
    cannons = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.PHOTONCANNON]
    if cannons and bot._cannon_seen_time is None:
        bot._cannon_seen_time = bot.time

    # ── Stargate ──
    stargates = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.STARGATE]
    if stargates and bot._stargate_seen_time is None:
        bot._stargate_seen_time = _estimate_building_start_time(bot, stargates[0])

    # ── Robotics Facility ──
    robos = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.ROBOTICSFACILITY]
    if robos and bot._robotics_facility_seen_time is None:
        bot._robotics_facility_seen_time = _estimate_building_start_time(bot, robos[0])

    # ── Robotics Bay ──
    bays = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.ROBOTICSBAY]
    if bays and bot._robotics_bay_seen_time is None:
        bot._robotics_bay_seen_time = _estimate_building_start_time(bot, bays[0])