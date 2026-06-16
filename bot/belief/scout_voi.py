"""Scout VOI — Staleness × relevance ranking for mid-late-game scouting.

Purpose: When strategy has converged and the army is lost, rank scout destinations
         by how stale (long-unseen) and how decision-relevant they are. This replaces
         the default expansion cycling with an information-driven approach.

Key Decisions: No scipy.stats.entropy — uses fixed relevance weights × game-time staleness.
               Composition uncertainty bonus is a small additive constant, not full entropy.
               Staleness tracking requires _location_last_seen on bot instance.

Limitations: Only active when enable_scout_voi config flag is True.
             Level-2 routing (proxy/cannon/rush tables) lives in scouting.py, not here.
"""

from typing import Optional

from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

from ares.consts import UnitRole
from bot.constants import (
    SCOUT_VOI_COMPOSITION_BONUS,
    SCOUT_VOI_RELEVANCE,
    VOI_VISION_RADIUS,
)
from cython_extensions import cy_distance_to


def get_voi_destinations(bot) -> Optional[list[tuple[Point2, float]]]:
    """Rank candidate scout destinations by staleness × relevance.

    Returns a list of (Point2, score) tuples sorted by score descending,
    or None if scout VOI is disabled or no locations have been tracked yet.

    Only active when strategy has converged (P(macro) > STRATEGY_HUNT_THRESHOLD)
    and cached army is stale (no recent army position).
    """
    if not bot.config.get("Belief", {}).get("enable_scout_voi", False):
        return None

    location_last_seen = getattr(bot, "_location_last_seen", {})
    if not location_last_seen:
        return None

    game_time = bot.time

    from bot.managers.scouting import _resolve_hunt_accessor

    scored = []
    for key, relevance in SCOUT_VOI_RELEVANCE.items():
        if key == "last_army_pos":
            cached_army = bot.mediator.get_cached_enemy_army
            non_worker = [u for u in cached_army
                          if u.type_id not in {UnitTypeId.DRONE, UnitTypeId.SCV, UnitTypeId.PROBE}]
            if non_worker:
                centroid = Point2((
                    sum(u.position.x for u in non_worker) / len(non_worker),
                    sum(u.position.y for u in non_worker) / len(non_worker),
                ))
                pos = centroid
            else:
                continue
        else:
            pos = _resolve_hunt_accessor(bot, key)
            if pos is None:
                continue

        last_seen = location_last_seen.get(key, 0.0)
        staleness = game_time - last_seen
        score = staleness * relevance

        # Composition uncertainty bonus: small boost for locations that would
        # confirm expected-but-unseen unit types
        composition = getattr(
            getattr(getattr(bot, "belief_state", None), "composition", None),
            "get_expected_unit_types", None
        )
        if composition is not None and key in ("enemy_nat", "enemy_main"):
            expected = composition()
            if expected and max(expected.values()) > 0.5:
                score += SCOUT_VOI_COMPOSITION_BONUS

        scored.append((pos, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored if scored else None


def update_location_sightings(bot) -> None:
    """Update _location_last_seen for all candidate VOI locations.

    Called once per step from bot.on_step(). Checks if any friendly scout
    (SCOUTING role, BUILD_RUNNER_SCOUT role, or observer) is within
    VOI_VISION_RADIUS of each candidate location, and updates the
    last-seen timestamp if so.
    """
    location_last_seen = getattr(bot, "_location_last_seen", None)
    if location_last_seen is None:
        return

    game_time = bot.time

    # Collect all friendly scout positions
    scout_positions = []

    for role in (UnitRole.SCOUTING, UnitRole.BUILD_RUNNER_SCOUT):
        units = bot.mediator.get_units_from_role(role=role)
        for u in units:
            scout_positions.append(u.position)

    # Observers on scouting duty
    for obs in bot.units(UnitTypeId.OBSERVER):
        scout_positions.append(obs.position)

    if not scout_positions:
        return

    from bot.managers.scouting import _resolve_hunt_accessor

    for key in SCOUT_VOI_RELEVANCE:
        if key == "last_army_pos":
            # last_army_pos staleness is based on cached army, not scout proximity
            cached_army = bot.mediator.get_cached_enemy_army
            non_worker = [u for u in cached_army
                          if u.type_id not in {UnitTypeId.DRONE, UnitTypeId.SCV, UnitTypeId.PROBE}]
            if non_worker:
                location_last_seen["last_army_pos"] = game_time
            continue

        pos = _resolve_hunt_accessor(bot, key)
        if pos is None:
            continue

        # Initialize unseen locations with 0.0 (max staleness)
        # so they get high VOI score and are scouted first
        if key not in location_last_seen:
            location_last_seen[key] = 0.0

        for scout_pos in scout_positions:
            if cy_distance_to(scout_pos, pos) < VOI_VISION_RADIUS:
                location_last_seen[key] = game_time
                break