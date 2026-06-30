# bot/managers/structure_manager.py
# Purpose: Nexus ability management (Chronoboost, Energy Recharge, Mass Recall)
# Key Decisions: All Nexus-cast abilities consolidated here for consistent energy management;
#   urgent Observer production gets chrono priority over the build profile list
# Limitations: Mass Recall combat-trigger logic remains in combat.py

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2
from sc2.units import Units
from cython_extensions import cy_distance_to

from bot.constants import (
    MASS_RECALL_COOLDOWN,
    MASS_RECALL_ENERGY_COST,
    RECHARGE_RANGE,
    get_active_profile,
)

# When intel urgency exceeds this, observer production is considered urgent
# (same threshold used by scouting to dispatch hunters)
_URGENT_OBSERVER_URGENCY = 0.5


def _get_recall_nexus_tag(bot) -> int | None:
    """Return the tag of the Nexus designated to preserve energy for Mass Recall.

    Only designates a recall Nexus once we've expanded (2+ Nexuses), since
    the first Nexus needs to spend freely early on. Uses the main/starting
    Nexus so it preserves 50 energy for emergency recall.
    Returns None if we haven't expanded yet or no suitable Nexus exists.
    """
    # Only designate a recall Nexus after expansion
    if len(bot.townhalls.ready) < 2:
        return None
    for nexus in bot.townhalls.ready:
        if nexus.type_id == UnitTypeId.NEXUS and nexus.position == bot.start_location:
            return nexus.tag
    # Fallback: first available Nexus
    for nexus in bot.townhalls.ready:
        if nexus.type_id == UnitTypeId.NEXUS:
            return nexus.tag
    return None


def use_recharge(bot, main_army: Units) -> bool:
    """
    Uses Energy Recharge on the highest-priority unit within range of any eligible Nexus.

    Iterates all Nexuses with enough energy, and for each scans for units within
    RECHARGE_RANGE that need energy. Picks the best (Nexus, target) pair across
    all Nexuses using the unit-type priority order, breaking ties by lowest energy %.

    Target priority: Shield Battery > Mothership > High Templar > Oracle > Sentry.

    Args:
        bot: The bot instance
        main_army: Unused — kept for call-site compatibility

    Returns:
        bool: True if Energy Recharge was used, False otherwise
    """
    # Priority order: Shield Battery > Mothership > High Templar > Oracle > Sentry
    priority_unit_types = [
        UnitTypeId.SHIELDBATTERY,
        UnitTypeId.MOTHERSHIP,
        UnitTypeId.HIGHTEMPLAR,
        UnitTypeId.ORACLE,
        UnitTypeId.SENTRY
    ]

    # Recall Nexus requires 100 energy (50 reserve + 50 cost) to spend
    recall_nexus_tag = _get_recall_nexus_tag(bot)
    eligible_nexuses = []
    for nexus in bot.structures(UnitTypeId.NEXUS).ready:
        min_energy = 100 if nexus.tag == recall_nexus_tag else 50
        if nexus.energy >= min_energy:
            eligible_nexuses.append(nexus)

    if not eligible_nexuses:
        return False

    # Scan all eligible Nexuses for the best (Nexus, target) pair.
    # Best = highest-priority unit type found, then lowest energy % within that type.
    best_nexus = None
    best_target = None
    best_priority_idx = len(priority_unit_types)  # lower is better
    best_energy_pct = float("inf")

    for nexus in eligible_nexuses:
        for prio_idx, unit_type in enumerate(priority_unit_types):
            # Can't improve on a higher-priority target already found
            if prio_idx > best_priority_idx:
                break

            # Shield Battery is in structures collection, others are in units
            collection = bot.structures if unit_type == UnitTypeId.SHIELDBATTERY else bot.units
            for unit in collection:
                if (unit.type_id == unit_type and
                    cy_distance_to(unit.position, nexus.position) <= RECHARGE_RANGE and
                    unit.energy_percentage < 1.0):
                    # Pick this if: higher priority type, or same type but lower energy
                    if (prio_idx < best_priority_idx or
                        (prio_idx == best_priority_idx and unit.energy_percentage < best_energy_pct)):
                        best_priority_idx = prio_idx
                        best_energy_pct = unit.energy_percentage
                        best_nexus = nexus
                        best_target = unit

    if not best_target:
        return False

    best_nexus(AbilityId.ENERGYRECHARGE_ENERGYRECHARGE, best_target)
    return True


def _is_urgent_observer_needed(bot) -> bool:
    """Return True if detection is urgently needed right now.

    Triggers when either:
    1. Cloaked/burrowed enemies are near our bases (set per-frame by threat_detection)
    2. Intel urgency is high — we're blind and need scouting coverage
    """
    # Cloaked threats near bases — need detection immediately
    if getattr(bot, '_cloaked_threat_positions', None):
        return True
    # Intel urgency — we've lost sight of the enemy army
    if getattr(bot, '_intel_urgency', 0.0) > _URGENT_OBSERVER_URGENCY:
        return True
    return False


def use_chronoboost(bot) -> bool:
    """
    Uses Chronoboost on the highest priority structure that would benefit from it.

    Priority order comes from the active BuildProfile's chrono_priority field,
    so each build can define its own chrono target order (e.g., Twilight first
    for blink-heavy builds, RoboBay first for robo-centric builds).

    Only boosts structures that are actively producing/researching and don't already have chronoboost.

    Args:
        bot: The bot instance

    Returns:
        bool: True if Chronoboost was used, False otherwise
    """
    # Recall Nexus requires 100 energy (50 reserve + 50 cost) to spend
    recall_nexus_tag = _get_recall_nexus_tag(bot)
    available_nexuses = [
        nexus for nexus in bot.townhalls.ready
        if nexus.type_id == UnitTypeId.NEXUS
        and nexus.energy >= (100 if nexus.tag == recall_nexus_tag else 50)
    ]

    if not available_nexuses:
        return False

    # Urgent observer: if a Robo is building an Observer and detection is urgently
    # needed (cloaked threats near bases or high intel urgency), chrono the Robo
    # ahead of the normal priority list. Observers take ~22s; chrono cuts that to ~15s,
    # which can be decisive when we're blind or under cloaked attack.
    if _is_urgent_observer_needed(bot):
        # Observer training ability
        observer_train_ability = (
            bot.game_data.units[UnitTypeId.OBSERVER.value].creation_ability.id
        )
        for robo in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
            if (not robo.has_buff(BuffId.CHRONOBOOSTENERGYCOST)
                    and robo.is_active):
                # Verify the Robo is actually training an Observer
                if robo.orders and any(
                    order.ability.id == observer_train_ability
                    for order in robo.orders
                ):
                    closest_nexus = min(available_nexuses,
                                        key=lambda n: cy_distance_to(n.position, robo.position))
                    closest_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, robo)
                    return True

    # Chrono priority from active BuildProfile (each build defines its own order)
    priority_structures = get_active_profile(bot).chrono_priority

    for structure_type in priority_structures:
        # Only boost active structures without existing chronoboost and >30s remaining
        candidates = []
        for structure in bot.structures(structure_type).ready:
            # Skip Gateways morphing into Warpgates — chrono on a morph is wasted
            if structure.type_id == UnitTypeId.GATEWAY and structure.orders:
                if any(order.ability.id == AbilityId.MORPH_WARPGATE for order in structure.orders):
                    continue
            if not structure.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and structure.is_active:
                remaining_time = get_remaining_activity_time(bot, structure)
                
                # Nexus producing probes: always chrono (probes warp fast, chrono = extra probe per cycle)
                # Research/production: only chrono if >30s remaining (shorter durations waste energy)
                is_probe_production = (structure.type_id == UnitTypeId.NEXUS and 
                                     structure.orders and 
                                     any(order.ability.id == AbilityId.NEXUSTRAIN_PROBE for order in structure.orders))
                
                if is_probe_production or remaining_time > 30.0:
                    candidates.append(structure)

        if candidates:
            target_structure = candidates[0]
            closest_nexus = min(available_nexuses,
                                key=lambda nexus: cy_distance_to(nexus.position, target_structure.position))
            closest_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target_structure)
            return True

    return False


def get_remaining_activity_time(bot, structure) -> float:
    """
    Calculate the remaining time for a structure's current activity (building, researching, or producing).
    Uses actual game data for accurate time calculation.

    Args:
        bot: The bot instance
        structure: The structure to check

    Returns:
        float: Remaining time in seconds, or 0.0 if no significant activity
    """

    if structure.orders:
        for order in structure.orders:
            if hasattr(order, 'progress') and order.progress < 1.0:
                try:
                    ability_data = bot.game_data.abilities[order.ability.value]
                    total_time = ability_data.cost.time / 22.4  # Game time to seconds
                    remaining = (1.0 - order.progress) * total_time
                    return remaining
                except (KeyError, AttributeError):
                    # Fallback estimate when game data unavailable
                    return (1.0 - order.progress) * 45.0

    return 0.0


def use_mass_recall(bot, target_position: Point2, avoid_nexus_tag: int | None = None) -> bool:
    """
    Cast Nexus Mass Recall at target_position.

    Handles Nexus-side concerns: global cooldown, energy check, Nexus selection.
    Combat-side decision (devastating loss, retreat blocked) is in combat.py.

    Prefers a Nexus that is NOT the one closest to the army (safer, further from
    enemy pressure). Falls back to any Nexus with enough energy.

    Args:
        bot: The bot instance
        target_position: Where to cast the recall (typically army center)
        avoid_nexus_tag: Tag of the Nexus nearest to army — prefer not to use this one

    Returns:
        bool: True if recall was cast, False otherwise
    """
    # Global cooldown check (130s across all Nexuses)
    if bot.time < bot._mass_recall_last_cast_time + MASS_RECALL_COOLDOWN:
        if bot.debug:
            print(f"[MASS RECALL] Blocked by cooldown at {bot.time:.1f}s (last cast {bot._mass_recall_last_cast_time:.1f}s)")
        return False

    # Prefer the designated recall Nexus (guaranteed to have energy)
    designated_tag = _get_recall_nexus_tag(bot)
    recall_nexus = None
    if designated_tag is not None:
        for nexus in bot.structures(UnitTypeId.NEXUS).ready:
            if nexus.tag == designated_tag and nexus.energy >= MASS_RECALL_ENERGY_COST:
                recall_nexus = nexus
                break

    # Fallback: any Nexus with enough energy (prefer not the one closest to army)
    if recall_nexus is None:
        for nexus in bot.structures(UnitTypeId.NEXUS).ready:
            if nexus.energy < MASS_RECALL_ENERGY_COST:
                continue
            if avoid_nexus_tag is not None and nexus.tag != avoid_nexus_tag:
                recall_nexus = nexus
                break
        # Last resort: use the avoided Nexus
        if recall_nexus is None:
            for nexus in bot.structures(UnitTypeId.NEXUS).ready:
                if nexus.energy >= MASS_RECALL_ENERGY_COST:
                    recall_nexus = nexus
                    break

    if recall_nexus is None:
        if bot.debug:
            nexus_info = [(n.tag, f"{n.energy:.0f}e") for n in bot.structures(UnitTypeId.NEXUS).ready]
            print(f"[MASS RECALL] No nexus with {MASS_RECALL_ENERGY_COST} energy | Nexuses: {nexus_info}")
        return False

    if bot.debug:
        print(f"[MASS RECALL] Casting from nexus {recall_nexus.tag} (energy={recall_nexus.energy:.0f}) at {bot.time:.1f}s -> {target_position.round(1)}")
    recall_nexus(AbilityId.EFFECT_MASSRECALL_NEXUS, target_position)
    bot._mass_recall_last_cast_time = bot.time

    if bot.debug:
        print(f"[MASS RECALL] Cast at {bot.time:.1f}s | Nexus: {recall_nexus.tag} | Target: {target_position.round(1)}")

    return True
