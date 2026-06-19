import math

from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId

from cython_extensions import cy_unit_pending, cy_structure_pending_ares, cy_distance_to

from sc2.position import Point2
from sc2.units import Units

from ares.behaviors.macro import (
    ProductionController,
    SpawnController,
    MacroPlan,
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    UpgradeController,
    BuildStructure,
)
from ares.consts import UnitRole, WORKER_TYPES
from ares.consts import LOSS_MARGINAL_OR_BETTER, ID, TARGET

from bot.utilities.performance_monitor import get_economy_state
from bot.intel import get_enemy_intel_quality
from bot.utilities.debug import render_detection_cannon_debug
from bot.combat.target_scoring import COUNTER_TABLE
from bot.constants import (
    MEMORY_EXPIRY_TIME,
    STALE_INTEL_THRESHOLD,
    RESOURCE_PRESSURE_MAX_NUDGE,
    RESOURCE_IMBALANCE_RATIO,
    FREEFLOW_INCOME_RATIO_THRESHOLD,
    BuildProfile,
    BUILD_PROFILES,
    PVT_STANDARD_2023_PROFILE,
    PVT_STALKER_2021_PROFILE,
    PVZ_STANDARD_PROFILE,
    PVP_2GATE_PROFILE,
    _resolve,
    get_active_profile,
    _needs_detection_cannons,
    DETECTION_CANNON_RANGE,
    PYLON_POWER_RANGE,
    MINERAL_CLEARANCE,
)
from ares.dicts.cost_dict import COST_DICT


def get_freeflow_mode(bot) -> bool:
    """
    Freeflow calculation: detects when SpawnController should `continue` past
    unaffordable units instead of `break`ing the loop.
    
    Uses a layered approach:
      1. Early-game pressure checks (one base, PvP)
      2. Bank-imbalance checks (always active, no PM dependency)
      3. Income-aware checks (when PerformanceMonitor data is available)
    
    Args:
        bot: The bot instance
        
    Returns:
        bool: True if should use freeflow mode
    """
    minerals = bot.minerals
    vespene = bot.vespene
    
    # --- Bank-imbalance checks (always active, catches resource skew) ---
    # Both resources high — just spend
    if minerals > 500 and vespene > 500:
        return True
    
    # Significant bank with clear imbalance — one resource is 3x+ the other
    # and the dominant resource exceeds 400 (enough that we should be spending it)
    if minerals > 400 and minerals > 3 * max(vespene, 1):
        return True
    if vespene > 400 and vespene > 3 * max(minerals, 1):
        return True
    
    # --- Income-aware checks (more precise, needs PM data) ---
    pm = getattr(bot, 'performance_monitor', None)
    if pm and pm.tracking_enabled and pm.samples_taken >= 5:
        income_min = pm.avg_income_minerals
        income_gas = pm.avg_income_vespene
        
        # Severe income imbalance: one resource accumulates faster than we can
        # spend it. Trigger freeflow to keep production flowing with whatever
        # resources we DO have. No bank threshold — the imbalance itself is
        # the signal, and our Layer 1 priority reorder handles what gets built.
        if income_gas > 0 and income_min / (income_gas + 1) > FREEFLOW_INCOME_RATIO_THRESHOLD:
            return True
        if income_min > 0 and income_gas / (income_min + 1) > FREEFLOW_INCOME_RATIO_THRESHOLD:
            return True
    
    # Normal proportional production
    return False


def get_optimal_gas_workers(bot) -> int:
    """
    Dynamically adjust gas workers based on economy state.
    Prevents gas flooding and mineral starvation.
    
    Args:
        bot: The bot instance
        
    Returns:
        int: Number of workers per gas (0-3)
    """
    # Get current gatherers (workers actually mining)
    gatherers = bot.mediator.get_units_from_role(role=UnitRole.GATHERING)
    
    # Critical mineral starvation - stop gas completely
    if (bot.minerals < 100 
        and bot.vespene > 300 
        and bot.supply_used < 84):
        return 0
    
    # Early game with few workers - stop gas temporarily
    if ((bot.reaction_manager.is_cheese_response and len(gatherers) < 21)
        or len(gatherers) < 12):
        return 0
    
    # Vespene flooding - toggle off gas mining
    if bot._gas_worker_toggle and bot.vespene > 1500 and bot.minerals < 100:
        bot._gas_worker_toggle = False
        return 0
    
    # Vespene starved - toggle gas mining back on
    if bot.vespene < 400 or bot.minerals > 1200:
        bot._gas_worker_toggle = True
    
    # Low gas - max workers
    if bot.vespene < 100:
        return 3
    
    return 3 if bot._gas_worker_toggle else 0

# Army composition constants 
STANDARD_ARMY_0 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 4},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.4, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 5},
    UnitTypeId.STALKER: {"proportion": 0.0, "priority": 2},
    UnitTypeId.ZEALOT: {"proportion": 0.2, "priority": 3},
}
STANDARD_ARMY_1 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.35, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.2, "priority": 4},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 3},
    UnitTypeId.STALKER: {"proportion": 0.0, "priority": 2},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 0},
}

PVP_ARMY_0 = {
    UnitTypeId.STALKER: {"proportion": 0.3, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.2, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 3},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.2, "priority": 2},
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 2},
}

PVP_ARMY_1 = {
    UnitTypeId.STALKER: {"proportion": 0.3, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.25, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 2},
    UnitTypeId.IMMORTAL: {"proportion": 0.3, "priority": 3},
}

CHEESE_DEFENSE_ARMY = {
    UnitTypeId.ADEPT: {"proportion": 0.25, "priority": 2},
    UnitTypeId.STALKER: {"proportion": 0.15, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.6, "priority": 1},
    
    
}

# 2021 PvT Stalker-Centric army: pure Stalker + Zealot, no Robo units
PVT_STALKER_2021_ARMY = {
    UnitTypeId.STALKER: {"proportion": 0.65, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 1},
}

# 2021 PvT Stalker-Centric moderate economy army: adds HT + Sentry for Storm/Feedback
# Proportions: Stalker/Zealot scaled down to 80% of base, HT 15%, Sentry 5%
PVT_STALKER_2021_ARMY_MODERATE = {
    UnitTypeId.STALKER: {"proportion": 0.52, "priority": 0},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.13, "priority": 1},
    UnitTypeId.ZEALOT: {"proportion": 0.33, "priority": 2},
    UnitTypeId.SENTRY: {"proportion": 0.02, "priority": 3},
}

# 2021 PvT Stalker-Centric post-Archon army: HTs have merged into Archons,
# redistribute HT proportion to Stalker/Zealot. Keep small HT production for Storm.
# Mirrors the 2023 profile's Archon switch pattern: stop heavy HT production,
# let existing HTs merge, keep a small HT line for Storm/Feedback.
PVT_STALKER_2021_ARMY_FULL = {
    UnitTypeId.STALKER: {"proportion": 0.50, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 1},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.10, "priority": 2},
    UnitTypeId.SENTRY: {"proportion": 0.05, "priority": 3},
}

# Wire army composition dicts into BuildProfile instances
# (profiles are defined in constants.py with empty dicts to avoid circular imports)
PVT_STANDARD_2023_PROFILE.army_composition_0 = STANDARD_ARMY_0
PVT_STANDARD_2023_PROFILE.army_composition_1 = STANDARD_ARMY_1
PVT_STALKER_2021_PROFILE.army_composition_0 = PVT_STALKER_2021_ARMY
PVT_STALKER_2021_PROFILE.army_composition_1 = PVT_STALKER_2021_ARMY_MODERATE  # Economy switch at moderate
PVT_STALKER_2021_PROFILE.army_composition_2 = PVT_STALKER_2021_ARMY_FULL  # Archon switch at 15%
PVZ_STANDARD_PROFILE.army_composition_0 = STANDARD_ARMY_0  # Same robo-centric comp as PvT
PVZ_STANDARD_PROFILE.army_composition_1 = STANDARD_ARMY_1
PVP_2GATE_PROFILE.army_composition_0 = PVP_ARMY_0
PVP_2GATE_PROFILE.army_composition_1 = PVP_ARMY_1

# Wire conditional upgrades (reactive, game-state dependent)
# These predicates reference helper functions defined below, so they're set up here.
_thermal_lance_predicate = lambda bot: bool(
    bot.structures(UnitTypeId.ROBOTICSBAY)
    and (bot.units(UnitTypeId.COLOSSUS).amount or cy_unit_pending(bot, UnitTypeId.COLOSSUS))
)
_storm_predicate = lambda bot: bool(
    bot.structures(UnitTypeId.TEMPLARARCHIVE)
    and (
        (bot.enemy_race in {Race.Terran, Race.Zerg} and _count_enemy_anti_air(bot) >= 2)
        or (bot.enemy_race == Race.Protoss and _count_enemy_carriers(bot) >= 1)
    )
)
PVT_STANDARD_2023_PROFILE.conditional_upgrades = [
    (UpgradeId.EXTENDEDTHERMALLANCE, _thermal_lance_predicate),
    (UpgradeId.PSISTORMTECH, _storm_predicate),
]
PVZ_STANDARD_PROFILE.conditional_upgrades = [
    (UpgradeId.EXTENDEDTHERMALLANCE, _thermal_lance_predicate),
    (UpgradeId.PSISTORMTECH, _storm_predicate),
]
PVP_2GATE_PROFILE.conditional_upgrades = [
    (UpgradeId.EXTENDEDTHERMALLANCE, _thermal_lance_predicate),
    (UpgradeId.PSISTORMTECH, _storm_predicate),
]
# 2021 profile: conditional upgrades are economy-gated (Weapons/Armor 2-3)

# ===== PRODUCTION NUDGING =====
# Uses COUNTER_TABLE to shift army proportions toward unit types that are
# effective against the observed enemy composition. See COUNTER_TABLE docstring
# in target_scoring.py for how to extend.
#
# HOW IT WORKS:
#   1. Read cached enemy army (filtered: no workers, structures, expired ghosts)
#   2. For each unit type in our composition, sum its COUNTER_TABLE bonuses
#      against all observed enemy units → "effectiveness score"
#   3. Normalize effectiveness scores to a [-MAX_NUDGE, +MAX_NUDGE] range
#   4. Add nudges to base proportions, clamp to MIN_PROPORTION, re-normalize to 1.0
#
# HOW TO EXTEND:
#   - Adding a new unit to an army composition dict? Add COUNTER_TABLE entries
#     in target_scoring.py. The nudging picks them up automatically.
#   - Adding a new army composition dict? Pass it through select_army_composition()
#     and nudging applies to it like any other.

PRODUCTION_MAX_NUDGE = 0.10
"""Maximum proportion shift per unit type (±10%). Keeps changes conservative."""

PRODUCTION_MIN_PROPORTION = 0.05
"""Floor: never drop a unit type below 5% proportion (prevents starvation)."""

PRODUCTION_MIN_ENEMY_UNITS = 3
"""Don't nudge if fewer than this many enemy combat units seen (too little data)."""


ANTI_COLOSSUS_TYPES = {
    UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT, UnitTypeId.CORRUPTOR,
    UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.TEMPEST,
}
"""Enemy unit types that hard-counter Colossus. Drives nudging in all matchups."""


def _count_enemy_anti_air(bot) -> int:
    """Count enemy anti-Colossus air units from cached intel."""
    return sum(1 for u in _get_enemy_combat_units(bot) if u.type_id in ANTI_COLOSSUS_TYPES)


def _count_enemy_carriers(bot) -> int:
    """Count enemy Carriers from cached intel. Used for PvP Storm trigger."""
    return sum(1 for u in _get_enemy_combat_units(bot) if u.type_id == UnitTypeId.CARRIER)


def _get_enemy_combat_units(bot) -> list:
    """Get filtered enemy combat units for production analysis.
    
    Excludes workers, structures, and expired ghost units (age >= MEMORY_EXPIRY_TIME).
    """
    cached = bot.mediator.get_cached_enemy_army or []
    return [
        u for u in cached
        if u.type_id not in WORKER_TYPES
        and not u.is_structure
        and u.age < MEMORY_EXPIRY_TIME
    ]


def _compute_effectiveness(composition: dict, enemy_units: list) -> dict[UnitTypeId, float]:
    """Score each own unit type's effectiveness against the enemy army.
    
    For each unit type in our composition, sums COUNTER_TABLE bonuses against
    every observed enemy unit. Returns average bonus per enemy unit.
    
    Args:
        composition: Army composition dict {UnitTypeId: {"proportion": ..., "priority": ...}}
        enemy_units: Filtered enemy combat units
        
    Returns:
        {UnitTypeId: avg_effectiveness_score} for each type in composition
    """
    n_enemies = len(enemy_units)
    if n_enemies == 0:
        return {}
    
    scores: dict[UnitTypeId, float] = {}
    for own_type in composition:
        total = sum(
            COUNTER_TABLE.get((own_type, e.type_id), 0.0)
            for e in enemy_units
        )
        scores[own_type] = total / n_enemies
    return scores


def nudge_proportions(
    base_composition: dict,
    enemy_units: list,
) -> dict:
    """Apply counter-table-driven proportion nudges to a base army composition.
    
    Returns a new composition dict with adjusted proportions. The base composition
    is never mutated. If there's insufficient enemy data, returns base unchanged.
    
    Perf note: O(k * m) where k = unit types in comp (~5), m = enemy units.
    Runs once per macro cycle — negligible frame cost.
    
    Args:
        base_composition: The base army composition dict
        enemy_units: Filtered enemy combat units
        
    Returns:
        New composition dict with nudged proportions (sums to 1.0)
    """
    if len(enemy_units) < PRODUCTION_MIN_ENEMY_UNITS:
        return base_composition
    
    effectiveness = _compute_effectiveness(base_composition, enemy_units)
    if not effectiveness:
        return base_composition
    
    # Find the range of effectiveness scores to normalize nudges
    scores = list(effectiveness.values())
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score
    
    # All scores equal (or only one type) → no nudge needed
    if score_range < 0.1:
        return base_composition
    
    # Normalize each score to [-1, +1] range, then scale by MAX_NUDGE
    # Highest effectiveness → +MAX_NUDGE, lowest → -MAX_NUDGE
    mid = (max_score + min_score) / 2.0
    nudges: dict[UnitTypeId, float] = {}
    for unit_type, score in effectiveness.items():
        normalized = (score - mid) / (score_range / 2.0)  # -1 to +1
        nudges[unit_type] = normalized * PRODUCTION_MAX_NUDGE
    
    # Reorder priorities by effectiveness: most effective → priority 0 (built first)
    ranked_types = sorted(effectiveness, key=lambda t: effectiveness[t], reverse=True)
    priority_map: dict[UnitTypeId, int] = {ut: i for i, ut in enumerate(ranked_types)}
    
    # Apply nudges to base proportions and reordered priorities
    nudged: dict = {}
    for unit_type, info in base_composition.items():
        nudge = nudges.get(unit_type, 0.0)
        base_prop = info["proportion"]
        new_proportion = base_prop + nudge
        # Only apply min floor if unit has base proportion > 0 or got a positive nudge
        # Units at 0.0 base with no positive nudge should stay at 0.0
        if base_prop > 0 or nudge > 0:
            new_proportion = max(new_proportion, PRODUCTION_MIN_PROPORTION)
        else:
            new_proportion = 0.0
        nudged[unit_type] = {
            "proportion": new_proportion,
            "priority": priority_map.get(unit_type, info["priority"]),
        }
    
    # Re-normalize proportions to sum to exactly 1.0
    # Round each to 4 decimal places, then fix the last to absorb rounding error
    total = sum(v["proportion"] for v in nudged.values())
    if total > 0:
        items = list(nudged.values())
        for info in items:
            info["proportion"] = round(info["proportion"] / total, 4)
        # Force exact 1.0 sum (ARES SpawnController asserts isclose to 1.0)
        rounding_error = 1.0 - sum(v["proportion"] for v in items)
        items[-1]["proportion"] = round(items[-1]["proportion"] + rounding_error, 4)
    
    return nudged


def _get_gas_ratio(unit_type: UnitTypeId) -> float:
    """Get the gas fraction of a unit's total cost using ARES COST_DICT.
    
    Returns 0.0 for mineral-only units, 1.0 for gas-only units,
    ~0.5 for balanced cost units. Uses dynamic lookup — no hardcoded costs.
    
    Perf note: dict lookup, called once per unit type per macro cycle.
    """
    cost = COST_DICT.get(unit_type)
    if cost is None:
        return 0.5  # Unknown unit, assume balanced
    total = cost.minerals + cost.vespene
    if total == 0:
        return 0.0
    return cost.vespene / total


def reorder_priorities_by_resources(composition: dict, bot) -> dict:
    """Layer 1: Reorder unit priorities based on current resource balance.
    
    When mineral-rich/gas-poor: low-gas units (Zealots) get priority 0.
    When gas-rich/mineral-poor: high-gas units (HT, Disruptor) get priority 0.
    When balanced: return composition unchanged.
    
    This directly addresses the SpawnController break-on-unaffordable problem:
    by putting affordable units first in priority order, the controller builds
    them before hitting the unaffordable gas-heavy unit and breaking.
    
    The composition dict is never mutated — returns a new dict.
    
    Perf note: O(k log k) sort where k = unit types in comp (~5). Negligible.
    
    Args:
        composition: Army composition dict {UnitTypeId: {"proportion": ..., "priority": ...}}
        bot: Bot instance for resource access
        
    Returns:
        New composition dict with reordered priorities (proportions unchanged)
    """
    minerals = bot.minerals
    vespene = bot.vespene
    
    # Detect imbalance direction
    mineral_rich = minerals > RESOURCE_IMBALANCE_RATIO * max(vespene, 1)
    gas_rich = vespene > RESOURCE_IMBALANCE_RATIO * max(minerals, 1)
    
    if not mineral_rich and not gas_rich:
        return composition  # Balanced — keep original priorities
    
    # Sort unit types by gas_ratio: ascending if mineral-rich (cheap-gas first),
    # descending if gas-rich (expensive-gas first)
    unit_types = list(composition.keys())
    unit_types.sort(
        key=lambda ut: _get_gas_ratio(ut),
        reverse=gas_rich,
    )
    
    # Assign new priorities: 0 = highest (first in sorted order)
    reordered: dict = {}
    for new_priority, unit_type in enumerate(unit_types):
        info = composition[unit_type]
        reordered[unit_type] = {
            "proportion": info["proportion"],
            "priority": new_priority,
        }
    
    return reordered


def resource_pressure_nudge(composition: dict, bot) -> dict:
    """Layer 2: Shift proportions toward units whose cost matches available resources.
    
    When gas-starved (minerals >> gas): boost low-gas-ratio units, reduce high-gas ones.
    When mineral-starved (gas >> minerals): boost high-gas-ratio units, reduce low-gas ones.
    
    Chained after counter-table nudge_proportions() — both nudges compose additively.
    Max shift ±RESOURCE_PRESSURE_MAX_NUDGE (15%), clamped to PRODUCTION_MIN_PROPORTION floor,
    re-normalized to 1.0.
    
    Uses COST_DICT for dynamic gas-ratio lookup — no hardcoded costs.
    
    Perf note: O(k) where k = unit types in comp (~5). Negligible.
    
    Args:
        composition: Army composition dict (already counter-nudged)
        bot: Bot instance for resource access
        
    Returns:
        New composition dict with resource-pressure-adjusted proportions
    """
    minerals = bot.minerals
    vespene = bot.vespene
    
    # Detect imbalance
    gas_starved = minerals > RESOURCE_IMBALANCE_RATIO * max(vespene, 1)
    mineral_starved = vespene > RESOURCE_IMBALANCE_RATIO * max(minerals, 1)
    
    if not gas_starved and not mineral_starved:
        return composition  # Balanced — no pressure nudge needed
    
    # Compute gas ratios for all unit types in composition
    gas_ratios: dict[UnitTypeId, float] = {}
    for unit_type in composition:
        gas_ratios[unit_type] = _get_gas_ratio(unit_type)
    
    if not gas_ratios:
        return composition
    
    # Normalize gas ratios to [-1, +1] range for nudge direction
    ratios = list(gas_ratios.values())
    max_r, min_r = max(ratios), min(ratios)
    ratio_range = max_r - min_r
    
    if ratio_range < 0.05:
        return composition  # All units cost roughly the same — no nudge
    
    mid_r = (max_r + min_r) / 2.0
    
    # Build nudge per unit type
    nudged: dict = {}
    for unit_type, info in composition.items():
        gas_r = gas_ratios[unit_type]
        normalized = (gas_r - mid_r) / (ratio_range / 2.0)  # -1 to +1
        
        if gas_starved:
            # Boost low-gas units (negative normalized), penalize high-gas
            nudge = -normalized * RESOURCE_PRESSURE_MAX_NUDGE
        else:
            # mineral_starved: Boost high-gas units, penalize low-gas
            nudge = normalized * RESOURCE_PRESSURE_MAX_NUDGE
        
        new_proportion = info["proportion"] + nudge
        new_proportion = max(new_proportion, PRODUCTION_MIN_PROPORTION)
        nudged[unit_type] = {
            "proportion": new_proportion,
            "priority": info["priority"],
        }
    
    # Re-normalize to 1.0
    total = sum(v["proportion"] for v in nudged.values())
    if total > 0:
        items = list(nudged.values())
        for info in items:
            info["proportion"] = round(info["proportion"] / total, 4)
        rounding_error = 1.0 - sum(v["proportion"] for v in items)
        items[-1]["proportion"] = round(items[-1]["proportion"] + rounding_error, 4)
    
    return nudged


def strategy_nudge_proportions(composition: dict, bot) -> dict:
    """Layer 1.5: Shift proportions toward units effective against predicted enemy composition.

    Uses strategy belief (P(cheese/all_in/timing/macro)) to predict what enemy units
    we'll face, then scores our composition against those predicted units using the
    existing COUNTER_TABLE. This is data-driven — the only "opinion" is what units
    each strategy/race typically produces (STRATEGY_EXPECTED_UNITS).

    Chained after counter-table nudge, before resource-pressure nudge.
    Probability-weighted across categories above STRATEGY_NUDGE_THRESHOLD.
    Max shift ±STRATEGY_NUDGE_MAX (10%), clamped to PRODUCTION_MIN_PROPORTION floor,
    re-normalized to 1.0.

    Perf note: O(k * m) where k = unit types in comp (~5), m = expected unit types (~5).
    Runs once per macro cycle — negligible frame cost.

    Args:
        composition: Army composition dict (already counter-nudged)
        bot: Bot instance for strategy belief access

    Returns:
        New composition dict with strategy-adjusted proportions
    """
    from bot.constants import (
        STRATEGY_EXPECTED_UNITS,
        STRATEGY_NUDGE_MAX,
        STRATEGY_NUDGE_THRESHOLD,
    )

    # Guard: strategy belief must be enabled and have a prediction
    if not bot.config.get("Belief", {}).get("enable_strategy", False):
        return composition

    prediction = getattr(
        getattr(bot, "belief_state", None), "strategy", None
    )
    if prediction is None:
        return composition
    prediction = prediction.last_prediction
    if prediction is None:
        return composition

    enemy_race = bot.enemy_race

    # Accumulate probability-weighted effectiveness scores across categories
    # Each category above threshold contributes proportionally to its P(category)
    effectiveness: dict[UnitTypeId, float] = {}
    total_weight = 0.0

    for category, prob in prediction.probs.items():
        if prob < STRATEGY_NUDGE_THRESHOLD:
            continue
        expected_units = STRATEGY_EXPECTED_UNITS.get((category, enemy_race))
        if expected_units is None:
            continue

        # Convert expected unit proportions to a fake "enemy list" for
        # _compute_effectiveness. Weight each unit type by its proportion.
        # _compute_effectiveness sums COUNTER_TABLE scores per enemy unit,
        # so we create weighted "virtual" enemies.
        fake_enemies = []
        for unit_type, proportion in expected_units.items():
            # Skip non-unit entries (Pylon, PhotonCannon, etc.) — not in COUNTER_TABLE
            if unit_type in (UnitTypeId.PYLON, UnitTypeId.PHOTONCANNON,
                             UnitTypeId.BUNKER, UnitTypeId.SPINECRAWLER,
                             UnitTypeId.SCV, UnitTypeId.DRONE, UnitTypeId.PROBE,
                             UnitTypeId.MULE):
                continue
            # Create a minimal fake unit with type_id for _compute_effectiveness
            # We need enough copies to represent the proportion weight
            count = max(1, round(proportion * 10))  # Scale to integer counts
            for _ in range(count):
                fake_enemies.append(_FakeEnemy(unit_type))

        if not fake_enemies:
            continue

        # Score our composition against these predicted enemies
        cat_effectiveness = _compute_effectiveness(composition, fake_enemies)
        if not cat_effectiveness:
            continue

        # Weight by P(category) and accumulate
        for unit_type, score in cat_effectiveness.items():
            effectiveness[unit_type] = effectiveness.get(unit_type, 0.0) + score * prob
        total_weight += prob

    if not effectiveness or total_weight < 0.01:
        return composition

    # Normalize effectiveness scores to [-1, +1] range, then scale by STRATEGY_NUDGE_MAX
    scores = list(effectiveness.values())
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score

    if score_range < 0.1:
        return composition  # All scores equal — no nudge needed

    mid = (max_score + min_score) / 2.0
    nudged: dict = {}
    for unit_type, info in composition.items():
        score = effectiveness.get(unit_type, mid)  # Default to mid if not scored
        normalized = (score - mid) / (score_range / 2.0)  # -1 to +1
        nudge = normalized * STRATEGY_NUDGE_MAX
        new_proportion = info["proportion"] + nudge
        # Only apply min floor if unit has base proportion > 0 or got a positive nudge
        if info["proportion"] > 0 or nudge > 0:
            new_proportion = max(new_proportion, PRODUCTION_MIN_PROPORTION)
        else:
            new_proportion = 0.0
        nudged[unit_type] = {
            "proportion": new_proportion,
            "priority": info["priority"],
        }

    # Re-normalize to 1.0
    total = sum(v["proportion"] for v in nudged.values())
    if total > 0:
        items = list(nudged.values())
        for info in items:
            info["proportion"] = round(info["proportion"] / total, 4)
        rounding_error = 1.0 - sum(v["proportion"] for v in items)
        items[-1]["proportion"] = round(items[-1]["proportion"] + rounding_error, 4)

    return nudged


class _FakeEnemy:
    """Minimal stand-in for an enemy unit with just a type_id.

    Used by strategy_nudge_proportions() to pass predicted enemy types
    through _compute_effectiveness() without needing real Unit objects.
    """
    __slots__ = ("type_id",)

    def __init__(self, type_id: UnitTypeId):
        self.type_id = type_id


# No static upgrade list - use get_desired_upgrades() instead

def calculate_optimal_worker_count(bot) -> int:
    """
    Calculates the optimal number of workers based on available resources.
    Checks mineral contents and vespene contents to determine if mining positions are viable.
    
    Returns:
        int: The optimal number of workers to build
    """
    mineral_spots = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        for mineral in nearby_minerals:
            if mineral.mineral_contents > 100:  
                mineral_spots += 2
    
    gas_spots = 0
    for gas_building in bot.gas_buildings:
        if gas_building.vespene_contents > 100:
            gas_spots += 3
    
    # Add a small buffer for worker replacements
    worker_count = mineral_spots + gas_spots + 4
    
    return worker_count


def is_base_depleted(bot) -> bool:
    """Accurate base depletion using ARES data + python-sc2 mineral contents"""
    
    # Method 1: Check mineral contents directly (python-sc2)
    depleted_bases = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        if nearby_minerals:
            depleted_patches = len([m for m in nearby_minerals if m.mineral_contents < 300])
            if depleted_patches / len(nearby_minerals) > 0.5:
                depleted_bases += 1
    
    # Method 2: Use ARES worker assignment data
    mineral_assignments = bot.mediator.get_mineral_patch_to_list_of_workers
    available_patches = bot.mediator.get_num_available_mineral_patches
    
    # Count oversaturated patches (3+ workers per patch)
    oversaturated_count = sum(1 for workers in mineral_assignments.values() 
                             if len(workers) >= 3)
    
    gas_workers = len(bot.mediator.get_worker_to_vespene_dict)
    mining_workers = bot.workers.amount - gas_workers
    
    return (
        depleted_bases > 0                          # Some bases have depleted patches
        or available_patches < 4                    # Very few available patches  
        or (mining_workers > 0 and available_patches == 0)  # Workers but no patches
        or oversaturated_count > 2                  # Too many oversaturated patches
    )


def expansion_checker(bot, main_army) -> int:
    """
    Evaluates multiple factors to determine when to expand:
    1. Resource starvation (production idle due to lack of income)
    2. Base depletion (worker saturation with declining income)
    3. Army safety for expansion
    4. Spending efficiency using python-sc2 score metrics
    
    Returns the recommended expansion count.
    """
    current_bases = len(bot.townhalls)
    current_workers = bot.workers.amount
    optimal_workers = calculate_optimal_worker_count(bot)
    worker_saturation = current_workers / optimal_workers if optimal_workers > 0 else 0
    
    mineral_collection_rate = bot.state.score.collection_rate_minerals
    idle_production_time = bot.state.score.idle_production_time
    expansion_count = current_bases
    
    # Calculate spending efficiency (SQ-style)
    current_unspent = bot.minerals
    spending_efficiency = mineral_collection_rate / (current_unspent + 1) if mineral_collection_rate > 0 else 0
    
    # Safety check - only expand if safe (filter workers from both armies)
    own_combat_units = [u for u in bot.own_army if u.type_id not in WORKER_TYPES]
    enemy_combat_units = [u for u in bot.enemy_army if u.type_id not in WORKER_TYPES]
    army_safe = bot.mediator.can_win_fight(
        own_units=own_combat_units,
        enemy_units=enemy_combat_units,
        timing_adjust=True,
        good_positioning=False,
        workers_do_no_damage=True,
    ) in LOSS_MARGINAL_OR_BETTER
    
    if not army_safe:
        return expansion_count
    
    resource_starved = (
        idle_production_time > 30.0  # Production buildings idle for 30+ seconds
        and current_unspent < 500    # Low mineral bank
        and mineral_collection_rate > 0  # But we do have some income
    )
    
    bases_depleting = is_base_depleted(bot)
    
    # Spending efficiency thresholds (dynamic by game state)
    if bot.game_state == 0:      # Early: should spend quickly
        efficiency_threshold = 1.5
    elif bot.game_state == 1:    # Mid: more complex economy
        efficiency_threshold = 1.0  
    else:                        # Late: can bank for big investments
        efficiency_threshold = 0.5
    
    # Low spending efficiency = banking too much relative to income
    inefficient_spending = spending_efficiency < efficiency_threshold
    
    if resource_starved:
        expansion_count = current_bases + 1
    elif bases_depleting:
        expansion_count = current_bases + 1
    elif inefficient_spending and worker_saturation > 0.7:
        expansion_count = current_bases + 1
    
    # Game state-based fallback expansions (minimal safety net)
    if bot.game_state >= 1 and current_bases < 2:
        expansion_count = max(expansion_count, 2)
    elif bot.game_state >= 2 and current_bases < 3:
        expansion_count = max(expansion_count, 3)
    
    # Limit early expansion with small army
    if current_bases == 1 and len(main_army) < 5 and bot.game_state == 0:
        expansion_count = 1
    
    return expansion_count


def get_desired_upgrades(bot) -> list[UpgradeId]:
    """
    Returns dynamic upgrade list based on the active BuildProfile and game state.
    UpgradeController will automatically build required tech structures.
    """
    profile = get_active_profile(bot)
    upgrades: list[UpgradeId] = []
    
    # Static upgrade order from profile
    for upgrade in profile.upgrade_order:
        if not bot.pending_or_complete_upgrade(upgrade):
            upgrades.append(upgrade)
    
    # Conditional upgrades (reactive, game-state dependent)
    for upgrade, predicate in profile.conditional_upgrades:
        if not bot.pending_or_complete_upgrade(upgrade) and predicate(bot):
            upgrades.append(upgrade)
    
    # Gate early upgrades if economy not ready (use centralized economy state)
    economy_state = get_economy_state(bot)
    if economy_state in ("recovery", "reduced"):
        return upgrades
    
    # Also gate if army is too small (need units before upgrades)
    if bot.supply_army < 15:
        return upgrades
    
    return upgrades


def _train_warp_prism(bot) -> None:
    """Train Warp Prisms to target count from the active BuildProfile.
    Supports dynamic warp_prism_target (e.g., 2023 build returns 1 when tech/supply met, 0 otherwise).
    Returns early if profile says 0 warp prisms.
    """
    profile = get_active_profile(bot)
    target_count = _resolve(profile.warp_prism_target, bot)
    if target_count == 0:
        return

    total_prisms = (bot.units(UnitTypeId.WARPPRISM).amount +
                    bot.units(UnitTypeId.WARPPRISMPHASING).amount +
                    cy_unit_pending(bot, UnitTypeId.WARPPRISM))
    if total_prisms >= target_count:
        return
    if not bot.can_afford(UnitTypeId.WARPPRISM):
        return

    robotics_facilities = bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle
    for facility in robotics_facilities:
        facility.train(UnitTypeId.WARPPRISM)
        return


def require_shield_battery(bot) -> bool:
    """
    Returns True if shield battery should be built after cheese response.
    Limits to one battery total. Tech requirements checked by BuildStructure.
    """
    total_batteries = (bot.structures(UnitTypeId.SHIELDBATTERY).amount + 
                      cy_structure_pending_ares(bot, UnitTypeId.SHIELDBATTERY))
    
    return bot.reaction_manager.is_cheese_response and total_batteries == 0


def get_shield_battery_base_location(bot) -> Point2:
    """
    Determines which base location to build shield battery at:
    - At natural if buildings or townhall exist there
    - At main base otherwise (uses static_defence placement)
    """
    natural_location = bot.mediator.get_own_nat
    
    natural_has_base = bot.townhalls.closer_than(10, natural_location).amount > 0
    natural_has_structures = bot.structures.closer_than(10, natural_location).amount > 0
    
    if natural_has_base or natural_has_structures:
        return natural_location
    else:
        # Use main base - BuildStructure with static_defence=True will find defensive spot
        return bot.start_location


def get_desired_gateway_count(bot) -> int:
    """
    Returns desired gateway/warpgate count based on the active BuildProfile.
    Walks gateway_thresholds in reverse (highest nexus req first), returns first match.
    """
    profile = get_active_profile(bot)
    bases = len(bot.townhalls)
    for nexus_req, gate_count in reversed(profile.gateway_thresholds):
        if bases >= nexus_req:
            return gate_count
    return profile.gateway_thresholds[0][1]  # fallback to first entry


def get_desired_forge_count(bot) -> int:
    """
    Returns desired forge count based on the active BuildProfile.
    Supports int (static) or callable (dynamic) values.
    """
    profile = get_active_profile(bot)
    return _resolve(profile.forge_count, bot)


def select_army_composition(bot, main_army: Units) -> dict:
    """
    Determines which army composition to use based on the active BuildProfile
    and observed enemy composition.
    
    Steps:
        1. Pick base composition from profile (army_composition_0)
        2. Apply economy switch if threshold met (army_composition_0 → army_composition_1)
        3. Apply archon switch if threshold met (→ army_composition_2, or army_1 fallback)
        4. Apply counter-table-driven proportion nudges based on enemy comp
        5. Cache the nudged result on bot._last_nudged_comp for debug display
    
    Returns:
        dict: The selected (and possibly nudged) army composition dictionary
    """
    profile = get_active_profile(bot)
    
    # All builds now have dedicated profiles — use profile compositions directly
    selected_composition = profile.army_composition_0
    army_1 = profile.army_composition_1
    army_2 = profile.army_composition_2 if profile.army_composition_2 else army_1  # Fallback for profiles without army_2
    threshold = profile.archon_switch_threshold
    
    # Economy switch: one-way transition from army_composition_0 to army_composition_1
    # Once triggered, never reverts even if economy drops (sticky flag)
    if profile.economy_switch_threshold is not None:
        if not hasattr(bot, '_economy_switch_triggered'):
            bot._economy_switch_triggered = False
        
        economy_state = get_economy_state(bot)
        econ_threshold = profile.economy_switch_threshold
        
        if not bot._economy_switch_triggered:
            if ((econ_threshold == "moderate" and economy_state in ("moderate", "full"))
                    or (econ_threshold == "full" and economy_state == "full")):
                bot._economy_switch_triggered = True
        
        if bot._economy_switch_triggered:
            selected_composition = army_1
    
    # Switch composition when Archon percentage exceeds threshold
    # Uses army_2 (post-Archon) if available, otherwise army_1 (legacy behavior)
    if main_army and len(main_army) > 0:
        archon_count = sum(1 for unit in main_army if unit.type_id == UnitTypeId.ARCHON)
        archon_percentage = archon_count / len(main_army) if len(main_army) > 0 else 0
        
        if archon_percentage >= threshold:
            selected_composition = army_2
    
    # === Nudge pipeline: counter-table → strategy → resource-pressure → priority reorder ===
    
    # Step 1: Counter-table nudge (only if intel is fresh enough)
    # When composition belief is enabled, use its probability-weighted freshness
    use_belief = bot.config.get("Belief", {}).get("enable_composition", False)
    if use_belief:
        has_intel = bot._enemy_army_ever_seen
        freshness = bot.belief_state.composition.freshness
    else:
        intel = get_enemy_intel_quality(bot)
        has_intel = intel["has_intel"]
        freshness = intel["freshness"]

    if has_intel and freshness > STALE_INTEL_THRESHOLD:
        enemy_units = _get_enemy_combat_units(bot)
        comp = nudge_proportions(selected_composition, enemy_units)
    else:
        comp = selected_composition
    
    # Step 1.5: Strategy nudge (shifts proportions toward units effective vs predicted enemies)
    # Wrapped in try/except so a bug in strategy nudge can't break army production
    try:
        strategy_comp = strategy_nudge_proportions(comp, bot)
        # Validate: proportions must sum to ~1.0 and all be non-negative
        total = sum(v["proportion"] for v in strategy_comp.values())
        if 0.9 < total < 1.1 and all(v["proportion"] >= 0 for v in strategy_comp.values()):
            comp = strategy_comp
    except Exception as e:
        print(f"[StrategyNudge] Fallback to counter-nudged composition: {e}")
    
    # Step 2: Resource-pressure nudge (shifts proportions toward affordable units)
    comp = resource_pressure_nudge(comp, bot)
    
    # Step 3: Priority reorder (puts affordable unit types first in SpawnController loop)
    comp = reorder_priorities_by_resources(comp, bot)
    
    # Cache for debug overlay
    bot._last_base_comp = selected_composition
    bot._last_nudged_comp = comp
    # Store resource pressure state for debug display
    minerals, vespene = bot.minerals, bot.vespene
    if minerals > RESOURCE_IMBALANCE_RATIO * max(vespene, 1):
        bot._resource_pressure = "GAS_STARVED"
    elif vespene > RESOURCE_IMBALANCE_RATIO * max(minerals, 1):
        bot._resource_pressure = "MIN_STARVED"
    else:
        bot._resource_pressure = "BALANCED"
    
    return comp


def _train_observers(bot) -> None:
    """Train observers to target count from the active BuildProfile.
    Supports dynamic observer_target (e.g., 2021 build returns 1 when detection needed, 0 otherwise).
    Returns early if profile says 0 observers and no detection trigger is active.
    """
    profile = get_active_profile(bot)
    target_count = _resolve(profile.observer_target, bot)
    if target_count == 0:
        return

    observer_count = (
        bot.units(UnitTypeId.OBSERVER).amount
        + bot.units(UnitTypeId.OBSERVERSIEGEMODE).amount
    )
    if cy_unit_pending(bot, UnitTypeId.OBSERVER) or not bot.can_afford(UnitTypeId.OBSERVER):
        return

    robotics_facilities = bot.structures(UnitTypeId.ROBOTICSFACILITY).ready
    if observer_count < 1:
        for facility in robotics_facilities:
            facility.train(UnitTypeId.OBSERVER)
            return

    if bot.game_state == 0:
        return

    if observer_count < target_count:
        for facility in robotics_facilities.idle:
            facility.train(UnitTypeId.OBSERVER)
            return


def build_detection_cannons(bot) -> None:
    """Build Pylon + Photon Cannon behind mineral lines at each owned base when
    detection is needed (cloaked/burrowed threats). Uses a per-base state machine
    to track progress: needs_pylon → pylon_pending → needs_cannon → cannon_pending → complete.

    Purpose: Provide detection coverage for builds without natural Robo/Observer access.
    Key Decisions: Per-base state machine prevents duplicate orders; proximity-based worker
        selection naturally picks probes from the target base. All bases build in parallel
        — no sequential blocking between bases. Mineral clearance ensures structures don't
        block probe pathing.
    Limitations: Only builds 1 cannon per base; relies on Forge being built by macro.
    """
    # Gate: need a completed Forge to build cannons
    if not bot.structures(UnitTypeId.FORGE).ready.exists:
        return

    # Prune state entries for dead Nexuses
    alive_tags = {nexus.tag for nexus in bot.townhalls.ready}
    bot._detection_cannon_state = {
        tag: state for tag, state in bot._detection_cannon_state.items()
        if tag in alive_tags
    }

    for nexus in bot.townhalls.ready:
        # Get behind-mineral-line positions: [left, center, right]
        behind_mineral = bot.mediator.get_behind_mineral_positions(
            th_pos=nexus.position
        )
        if not behind_mineral or len(behind_mineral) < 2:
            continue

        # Get mineral fields at this base for clearance checks
        mineral_fields = bot.mineral_field.closer_than(10, nexus.position)

        mineral_center = behind_mineral[1]  # Center position behind mineral line
        state = bot._detection_cannon_state.get(nexus.tag, "needs_pylon")

        # --- State: needs_pylon ---
        if state == "needs_pylon":
            # Check if a Pylon already exists near this base's mineral line
            existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_pylons.exists:
                bot._detection_cannon_state[nexus.tag] = "needs_cannon"
                continue

            # Check if a Pylon is already pending for THIS specific base
            # (use building tracker to avoid blocking other bases)
            pending_for_this_base = False
            building_tracker = bot.mediator.get_building_tracker_dict
            for worker_tag, info in building_tracker.items():
                if info.get(ID) == UnitTypeId.PYLON:
                    target = info.get(TARGET)
                    if target and cy_distance_to(target.position, mineral_center) < DETECTION_CANNON_RANGE:
                        pending_for_this_base = True
                        break
            if pending_for_this_base:
                continue

            # Find a valid placement position for the Pylon
            placement = _find_pylon_placement(bot, behind_mineral, nexus.position, mineral_fields)

            if placement is None:
                continue

            if not bot.can_afford(UnitTypeId.PYLON):
                continue

            worker = bot.mediator.select_worker(
                target_position=placement, force_close=True
            )
            if worker is None:
                continue

            success = bot.mediator.build_with_specific_worker(
                worker=worker,
                structure_type=UnitTypeId.PYLON,
                pos=placement,
            )
            if success:
                bot._detection_cannon_state[nexus.tag] = "pylon_pending"

        # --- State: pylon_pending ---
        elif state == "pylon_pending":
            existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_pylons.exists:
                ready_pylons = existing_pylons.ready
                if ready_pylons.exists:
                    bot._detection_cannon_state[nexus.tag] = "needs_cannon"
                # Pylon exists but not ready yet — stay in pylon_pending
            else:
                # No pylon at all — build may have been cancelled, retry
                bot._detection_cannon_state[nexus.tag] = "needs_pylon"

        # --- State: needs_cannon ---
        elif state == "needs_cannon":
            # Check if a Cannon already exists near this base's mineral line
            existing_cannons = bot.structures(UnitTypeId.PHOTONCANNON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_cannons.exists:
                bot._detection_cannon_state[nexus.tag] = "complete"
                continue

            # Find the pylon that will power the cannon
            existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if not existing_pylons.ready.exists:
                # Pylon not ready yet — go back to waiting
                bot._detection_cannon_state[nexus.tag] = "pylon_pending"
                continue

            pylon = existing_pylons.ready.closest_to(mineral_center)

            # Find a valid placement position for the Cannon
            placement = _find_cannon_placement(bot, behind_mineral, pylon, mineral_center, nexus.position, mineral_fields)

            if placement is None:
                continue

            if not bot.can_afford(UnitTypeId.PHOTONCANNON):
                continue

            worker = bot.mediator.select_worker(
                target_position=placement, force_close=True
            )
            if worker is None:
                continue

            success = bot.mediator.build_with_specific_worker(
                worker=worker,
                structure_type=UnitTypeId.PHOTONCANNON,
                pos=placement,
            )
            if success:
                bot._detection_cannon_state[nexus.tag] = "cannon_pending"

        # --- State: cannon_pending ---
        elif state == "cannon_pending":
            existing_cannons = bot.structures(UnitTypeId.PHOTONCANNON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_cannons.exists:
                bot._detection_cannon_state[nexus.tag] = "complete"
            else:
                # Check if a cannon is still pending for THIS specific base
                pending_for_this_base = False
                building_tracker = bot.mediator.get_building_tracker_dict
                for worker_tag, info in building_tracker.items():
                    if info.get(ID) == UnitTypeId.PHOTONCANNON:
                        target = info.get(TARGET)
                        if target and cy_distance_to(target.position, mineral_center) < DETECTION_CANNON_RANGE:
                            pending_for_this_base = True
                            break
                if not pending_for_this_base:
                    # No cannon and nothing pending for this base — cancelled, retry
                    bot._detection_cannon_state[nexus.tag] = "needs_cannon"

        # --- State: complete ---
        elif state == "complete":
            # Verify cannon still exists; if destroyed, rebuild
            existing_cannons = bot.structures(UnitTypeId.PHOTONCANNON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if not existing_cannons.exists:
                # Cannon destroyed — check if pylon still exists
                existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                    DETECTION_CANNON_RANGE, mineral_center
                )
                if existing_pylons.exists:
                    bot._detection_cannon_state[nexus.tag] = "needs_cannon"
                else:
                    bot._detection_cannon_state[nexus.tag] = "needs_pylon"

    # Render debug overlay (no-op when bot.debug is False)
    render_detection_cannon_debug(bot)


def _is_clear_of_minerals(pos: Point2, mineral_fields, clearance: float = MINERAL_CLEARANCE) -> bool:
    """Check if a position has enough clearance from mineral fields for probe pathing.

    Args:
        pos: Position to check
        mineral_fields: Units collection of mineral fields at this base
        clearance: Minimum distance from mineral field center (default MINERAL_CLEARANCE)

    Returns:
        True if position is far enough from all mineral fields for probe pathing.
    """
    if not mineral_fields:
        return True
    for mf in mineral_fields:
        if cy_distance_to(pos, mf.position) < clearance:
            return False
    return True


def _find_pylon_placement(bot, behind_mineral: list, nexus_pos: Point2, mineral_fields) -> Point2 | None:
    """Find a valid placement for a Pylon, trying behind-mineral-line first,
    then falling back to positions near the nexus. Ensures clearance from
    mineral fields for probe pathing.

    Args:
        bot: Bot instance for placement validation
        behind_mineral: 3 positions behind mineral line [left, center, right]
        nexus_pos: Nexus position for fallback search
        mineral_fields: Mineral fields at this base (for clearance checks)

    Returns:
        Valid placement Point2, or None if no position found
    """
    # Priority 1: Behind mineral line positions (with mineral clearance)
    for pos in behind_mineral:
        if bot.mediator.can_place_structure(
            position=pos, structure_type=UnitTypeId.PYLON
        ) and _is_clear_of_minerals(pos, mineral_fields):
            return pos

    # Priority 2: Between nexus and mineral center (offset toward minerals)
    mineral_center = behind_mineral[1]
    for offset in (3.0, 4.0, 5.0):
        pos = Point2(nexus_pos.towards(mineral_center, offset))
        if bot.mediator.can_place_structure(
            position=pos, structure_type=UnitTypeId.PYLON
        ) and _is_clear_of_minerals(pos, mineral_fields):
            return pos

    # Priority 3: Around the nexus at increasing radii
    for radius in (4.0, 5.0, 6.0):
        for angle_offset in range(0, 360, 45):
            angle = math.radians(angle_offset)
            pos = Point2((
                nexus_pos.x + radius * math.cos(angle),
                nexus_pos.y + radius * math.sin(angle),
            ))
            if bot.mediator.can_place_structure(
                position=pos, structure_type=UnitTypeId.PYLON
            ) and _is_clear_of_minerals(pos, mineral_fields):
                return pos

    return None


def _find_cannon_placement(
    bot, behind_mineral: list, pylon, mineral_center: Point2, nexus_pos: Point2, mineral_fields
) -> Point2 | None:
    """Find a valid placement for a Photon Cannon within pylon power range.
    Ensures clearance from mineral fields for probe pathing.

    Tries behind-mineral positions first, then positions near the pylon,
    then fallback positions around the nexus.

    Args:
        bot: Bot instance for placement validation
        behind_mineral: 3 positions behind mineral line [left, center, right]
        pylon: The powered Pylon that will supply the cannon
        mineral_center: Center position behind mineral line
        nexus_pos: Nexus position for fallback search
        mineral_fields: Mineral fields at this base (for clearance checks)

    Returns:
        Valid placement Point2, or None if no position found
    """
    # Priority 1: Behind mineral line within pylon power range (with mineral clearance)
    for pos in behind_mineral:
        dist_to_pylon = cy_distance_to(pos, pylon.position)
        if dist_to_pylon <= PYLON_POWER_RANGE:
            if bot.mediator.can_place_structure(
                position=pos, structure_type=UnitTypeId.PHOTONCANNON
            ) and _is_clear_of_minerals(pos, mineral_fields):
                return pos

    # Priority 2: Between mineral center and pylon
    for offset in (1.0, 2.0, 3.0):
        pos = Point2(mineral_center.towards(pylon.position, offset))
        dist_to_pylon = cy_distance_to(pos, pylon.position)
        if dist_to_pylon <= PYLON_POWER_RANGE:
            if bot.mediator.can_place_structure(
                position=pos, structure_type=UnitTypeId.PHOTONCANNON
            ) and _is_clear_of_minerals(pos, mineral_fields):
                return pos

    # Priority 3: Around the pylon at increasing radii (within power range)
    for radius in (2.0, 3.0, 4.0, 5.0):
        for angle_offset in range(0, 360, 45):
            angle = math.radians(angle_offset)
            pos = Point2((
                pylon.position.x + radius * math.cos(angle),
                pylon.position.y + radius * math.sin(angle),
            ))
            dist_to_pylon = cy_distance_to(pos, pylon.position)
            if dist_to_pylon <= PYLON_POWER_RANGE:
                if bot.mediator.can_place_structure(
                    position=pos, structure_type=UnitTypeId.PHOTONCANNON
                ) and _is_clear_of_minerals(pos, mineral_fields):
                    return pos

    # Priority 4: Around the nexus (cannon may be far from minerals but still useful)
    for radius in (4.0, 5.0, 6.0):
        for angle_offset in range(0, 360, 45):
            angle = math.radians(angle_offset)
            pos = Point2((
                nexus_pos.x + radius * math.cos(angle),
                nexus_pos.y + radius * math.sin(angle),
            ))
            # Check if within pylon power range
            dist_to_pylon = cy_distance_to(pos, pylon.position)
            if dist_to_pylon <= PYLON_POWER_RANGE:
                if bot.mediator.can_place_structure(
                    position=pos, structure_type=UnitTypeId.PHOTONCANNON
                ) and _is_clear_of_minerals(pos, mineral_fields):
                    return pos

    return None


async def handle_macro(
    bot,
    main_army: Units,
    warp_prism: Units,
    freeflow: bool,
) -> None:
    """
    Main macro logic: builds units, keeps supply up, reacts to cheese, 
    or transitions to late game. Call this in your bot's on_step.
    """
    spawn_location = bot.natural_expansion
    production_location = bot.start_location
    
    profile = get_active_profile(bot)
    worker_limit = _resolve(profile.worker_cap, bot)
    # Cheese reaction caps probe production until the reaction transitions
    # (keeps the build runner's ConstantWorkerProductionTill and post-build
    #  BuildWorkers in sync — previously BuildWorkers would jump straight
    #  to the standard profile cap, ignoring CategoryConfig.probe_cap)
    cheese_probe_cap = bot.reaction_manager.category_config.probe_cap
    if bot.reaction_manager.is_cheese_response and cheese_probe_cap is not None:
        worker_limit = min(worker_limit, cheese_probe_cap)
    optimal_worker_count = min(calculate_optimal_worker_count(bot), worker_limit)
    
    economy_state = get_economy_state(bot)
    
    # Select army composition based on reaction state
    # ReactionManager handles the one-way transition from cheese → standard
    if not bot.reaction_manager.is_cheese_response:
        army_composition = select_army_composition(bot, main_army)
        expansion_count = expansion_checker(bot, main_army)
    else:
        army_composition = CHEESE_DEFENSE_ARMY
        expansion_count = 3
    
    if require_shield_battery(bot):
        base_location = get_shield_battery_base_location(bot)
        bot.register_behavior(
            BuildStructure(
                base_location=base_location,
                structure_id=UnitTypeId.SHIELDBATTERY,
                static_defence=True
            )
        )
    
    # Scale gateways to desired count (from active BuildProfile)
    # Allow in reduced+ economy so production capacity ramps before the moderate transition,
    # preventing a burst of 2+ gateways when economy recovers
    if economy_state in ("reduced", "moderate", "full"):
        desired_gates = get_desired_gateway_count(bot)
        current_gates = (bot.structures(UnitTypeId.GATEWAY).amount + 
                         bot.structures(UnitTypeId.WARPGATE).amount + 
                         cy_structure_pending_ares(bot, UnitTypeId.GATEWAY))
        
        if current_gates < desired_gates and bot.can_afford(UnitTypeId.GATEWAY):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.GATEWAY))
    
    # Scale forges to desired count (0→1→2) - moderate+ only (lower priority than gates)
    if economy_state in ("moderate", "full"):
        desired_forges = get_desired_forge_count(bot)
        current_forges = (bot.structures(UnitTypeId.FORGE).amount + 
                         cy_structure_pending_ares(bot, UnitTypeId.FORGE))
        
        if current_forges < desired_forges and bot.can_afford(UnitTypeId.FORGE):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.FORGE))
    
    # Reactive structures from BuildProfile (e.g., Robo for detection in Stalker build)
    for structure_type, predicate in profile.conditional_structures:
        if predicate(bot) and bot.can_afford(structure_type):
            bot.register_behavior(BuildStructure(production_location, structure_type))
    
    # Detection cannons behind mineral lines (plug-and-play: only active if profile enables it)
    # Sticky trigger: once a cloaked threat is seen, the system stays active for ALL bases
    # (including new expansions) until every base has a completed cannon. This prevents
    # the harass unit from simply flying to an unprotected base.
    detection_cannon_flag = _resolve(profile.detection_cannons, bot)
    if detection_cannon_flag and _needs_detection_cannons(bot):
        bot._detection_cannon_triggered = True
    if detection_cannon_flag and bot._detection_cannon_triggered:
        build_detection_cannons(bot)
    
    macro_plan: MacroPlan = MacroPlan()
    
    # Determine if we should bank minerals for an expansion this frame.
    #
    # Root cause of the original bug: ExpansionController was at the END of
    # the MacroPlan. MacroPlan.execute() returns True on the first successful
    # behavior, so BuildWorkers (50m), AutoSupply (100m), and GasBuildingController
    # (75m) would succeed and return True before ExpansionController ever ran.
    # The bot never accumulated the 400m for a Nexus — it hovered at 200-350m,
    # perpetually spending on probes/pylons/assimilators.
    #
    # Fix: when banking, add ExpansionController(prioritize=True) at the TOP
    # of the MacroPlan. prioritize=True makes it return True even when it
    # can't afford the Nexus yet, blocking all lower-priority spending.
    #
    # The banking condition stays active while a worker is en route to the
    # expansion (cy_structure_pending_ares > 0) but the Nexus isn't physically
    # under construction yet. This prevents SpawnController from spending
    # minerals before the building manager can place the Nexus.
    #
    # We use expansion_count (from expansion_checker) instead of counting
    # townhalls. expansion_checker already evaluates resource starvation,
    # base depletion, and spending efficiency to decide if we need more
    # bases. This works for the natural (2 bases) AND for late-game
    # expansions when existing bases are mining out.
    nexus_under_construction = bot.structures(UnitTypeId.NEXUS).not_ready.amount > 0
    wants_to_expand = expansion_count > len(bot.townhalls)
    banking_for_expansion = (
        economy_state == "reduced"
        and not bot._under_attack
        and not nexus_under_construction
        and not bot.reaction_manager.is_cheese_response
        and wants_to_expand
    )
    
    if banking_for_expansion:
        # prioritize=True blocks all lower-priority spending until the
        # Nexus is placed. On frame 1, it sends a worker and returns True.
        # On subsequent frames, expansion_pending is True so the controller
        # returns False — but we still skip SpawnController below, so only
        # BuildWorkers (50m) and AutoSupply (100m) can spend. Minerals
        # accumulate and the building manager places the Nexus when affordable.
        macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1, prioritize=True))
    
    # Always: workers and supply (all economy states)
    # These run after ExpansionController, so they only execute if
    # ExpansionController didn't block (i.e. we can afford the Nexus
    # or already have enough bases).
    macro_plan.add(BuildWorkers(to_count=optimal_worker_count))
    macro_plan.add(AutoSupply(base_location=production_location))
    
    if economy_state == "recovery":
        pass
    else:
        # Reduced+: gas buildings and spawn from existing production
        gas_target = _resolve(profile.gas_target, bot)
        macro_plan.add(GasBuildingController(to_count=gas_target, max_pending=2))
        
        spawn_target = warp_prism[0].position if warp_prism else spawn_location
        # Freeflow prevents SpawnController from breaking on unaffordable high-priority
        # units. Use it when: cheese defense active, get_freeflow_mode says yes,
        # or economy is not reduced (moderate/full can spend freely).
        spawn_freeflow = (
            bot.reaction_manager.is_cheese_response
            or freeflow
            or economy_state != "reduced"
        )
        
        # When banking for expansion, SpawnController is skipped entirely
        # so minerals accumulate for the Nexus. Otherwise, produce army.
        if banking_for_expansion:
            pass  # No SpawnController — minerals go to ExpansionController
        else:
            macro_plan.add(SpawnController(army_composition, spawn_target=spawn_target, freeflow_mode=spawn_freeflow))
        
        # Expansion logic: moderate+ gets full expansion, reduced gets safety net to 2 bases
        # Skip expansions when under attack - focus resources on defense
        # No expansions during cheese response — focus entirely on defense
        # Note: when banking_for_expansion, ExpansionController was already
        # added at the top with prioritize=True, so we skip adding it again here.
        if not banking_for_expansion and not bot._under_attack and not bot.reaction_manager.is_cheese_response:
            if economy_state in ("moderate", "full"):
                macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1))
            elif wants_to_expand:
                # Reduced economy but expansion_checker says we need more bases.
                # This covers both the natural (early) and late-game expansions
                # when existing bases are mining out.
                macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1))
        
        if economy_state in ("moderate", "full"):
            # Moderate+: upgrades
            macro_plan.add(UpgradeController(get_desired_upgrades(bot), base_location=production_location))
            
            if economy_state == "full":
                # Full: add new production buildings
                macro_plan.add(ProductionController(army_composition, base_location=production_location, add_production_at_bank=(450,450), ignore_below_proportion=0.3))
    
    bot.register_behavior(macro_plan)
    _train_observers(bot)
    _train_warp_prism(bot)
