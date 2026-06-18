# bot/managers/reactions.py
"""Reaction management — strategy-specific responses to detected threats.

Purpose: Centralize reaction routing, lifecycle, and state management.
The ReactionManager class owns all reaction state and dispatches to
handler functions. Handlers are pure per-frame logic — no flag management.

Architecture:
  - Category layer (CHEESE/ALL_IN/TIMING/MACRO) → macro/build influence
  - Strategy layer (level2 label) → tactical handler selection
  - Category constrains which handlers are valid; strategy picks the specific one.

Key Decisions: Handlers are pure functions. The manager owns activation,
  deactivation, cleanup, and state. No scattered bot._* flags.
Limitations: ALL_IN and TIMING handlers are placeholders (None) until
  implemented. The manager routes correctly but takes no action for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.units import Units
from sc2.position import Point2

# Ares imports
from ares.consts import UnitRole, WORKER_TYPES, UnitTreeQueryType
from ares.behaviors.combat.individual import PathUnitToTarget, WorkerKiteBack
from ares.behaviors.combat import CombatManeuver
from ares.managers.manager_mediator import ManagerMediator
from ares.dicts.unit_data import UNIT_DATA

from cython_extensions import (
    cy_distance_to, cy_distance_to_squared, cy_center, cy_find_units_center_mass,
    cy_closest_to, cy_attack_ready, cy_in_attack_range, cy_pick_enemy_target,
    cy_structure_pending_ares
)

from bot.intel import detect_cheese
from bot.constants import (
    UNDER_ATTACK_VALUE_THRESHOLD,
    UNDER_ATTACK_RATIO_THRESHOLD,
    UNDER_ATTACK_CLEAR_VALUE,
    COMMON_UNIT_IGNORE_TYPES,
    MEMORY_EXPIRY_TIME,
    STRATEGY_THREAT_MULTIPLIER,
    STRATEGY_THREAT_CLEAR_MULTIPLIER,
    REACTION_CATEGORY_CONFIGS,
    CHEESE_THREAT_CLEAR_GRACE,
)
from bot.constants import StrategyCategory

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


# ===== REACTION MANAGER =====

class ReactionManager:
    """Owns all reaction state. Single point of contact for detection →
    reaction routing → lifecycle → consumer queries.

    Two layers:
      - Category (CHEESE/ALL_IN/TIMING/MACRO) → macro/build influence
      - Strategy (level2 label) → tactical handler selection

    Category constrains which handlers are valid; strategy picks the
    specific one. If no specific handler exists for a level2 label,
    the category default handler is used.
    """

    def __init__(self):
        self._active_category: Optional[StrategyCategory] = None
        self._active_strategy: Optional[str] = None
        self._reaction_start_time: float = -1.0
        self._transitioned: bool = False

        # Tracks the last time an enemy combat unit was near our bases.
        # Used by the cheese transition to detect a sustained threat-free window.
        self._last_threat_near_base_time: float = -1.0

        # Handler registry: category → {level2_label → handler}
        # Handlers are set after module init to avoid circular imports.
        self._handlers: dict[StrategyCategory, dict[str, Callable]] = {}
        # Category defaults: category → fallback handler (used when no specific handler matches)
        self._category_defaults: dict[StrategyCategory, Optional[Callable]] = {}
        # Deactivation checks: strategy → should_deactivate function
        self._deactivation_checks: dict[str, Callable] = {}

    def register_handlers(self) -> None:
        """Register handler functions. Called after module init to avoid circular imports."""
        self._handlers = {
            StrategyCategory.CHEESE: {
                "worker_rush": defend_worker_rush,
                "cannon_rush": defend_cannon_rush,
            },
            StrategyCategory.ALL_IN: {
                # Placeholder — no ALL_IN handlers yet
            },
            StrategyCategory.TIMING_ATTACK: {
                # Placeholder — no TIMING handlers yet
            },
            StrategyCategory.MACRO: {
                # No handlers — standard play
            },
        }
        self._category_defaults = {
            StrategyCategory.CHEESE: cheese_reaction,
            StrategyCategory.ALL_IN: None,       # Future: all_in_reaction
            StrategyCategory.TIMING_ATTACK: None, # No-op for now
            StrategyCategory.MACRO: None,         # No-op
        }
        self._deactivation_checks = {
            "worker_rush": _should_deactivate_worker_rush,
            "cannon_rush": _should_deactivate_cannon_rush,
        }

    # --- Consumer API (replaces scattered bot._* flag reads) ---

    @property
    def is_active(self) -> bool:
        """Any reaction currently running?"""
        return self._active_category is not None

    @property
    def is_cheese_response(self) -> bool:
        """Should macro use CHEESE_DEFENSE_ARMY, stop gas, etc.?
        True when cheese is active and not yet transitioned to standard play."""
        return (self._active_category is not None
                and not self._transitioned)

    @property
    def is_early_defensive(self) -> bool:
        """Should combat hold the army back? Same as is_cheese_response for now."""
        return self.is_cheese_response

    @property
    def keep_workers_safe(self) -> bool:
        """Should Mining keep gas workers safe?
        False during worker rush (workers are fighting, not mining)."""
        return self._active_strategy != "worker_rush"

    @property
    def active_reaction_name(self) -> Optional[str]:
        """The level2 label of the active reaction, or None."""
        return self._active_strategy

    @property
    def reaction_start_time(self) -> float:
        """Game time when the current reaction was activated."""
        return self._reaction_start_time

    @property
    def category_config(self):
        """The CategoryConfig for the active category, or MACRO config if inactive."""
        from bot.constants import REACTION_CATEGORY_CONFIGS
        cat = self._active_category or StrategyCategory.MACRO
        return REACTION_CATEGORY_CONFIGS[cat]

    # --- Core lifecycle ---

    # Minimum P(top category) for the BN model to trigger a reaction.
    # Applies uniformly to all non-MACRO categories (cheese, all_in,
    # timing_attack). Below this, the prediction means "not enough evidence
    # yet" and we wait rather than acting. MACRO never triggers a reaction
    # (it's the default/no-op state).
    REACTION_CONFIDENCE_THRESHOLD: float = 0.6

    def update(self, bot: "PiG_Bot") -> None:
        """Read current StrategyPrediction, decide if reaction should change.

        Routes detection output to the appropriate handler. Always runs,
        even when under attack (unlike the old early_threat_sensor gate).

        Architecture (per bayesian_belief_layer_plan.md):
          1. BN model (primary) — route when P(top category) ≥ threshold
          2. Auto-TRUE guards — fallback when model is unavailable
          3. Rule-based detect_cheese() — fallback when belief is disabled

        The BN model is the primary path. When it produces a prediction below
        the confidence threshold, that's a valid "not enough evidence yet"
        signal — we wait, not fall back to rules. The rule-based fallback only
        runs when belief is disabled or the model hasn't produced a prediction.
        """
        prediction = None
        use_belief = bot.config.get("Belief", {}).get("enable_strategy", True)

        # Primary path: Strategy Belief (BN model + auto-TRUE guards)
        # The BN model always produces a prediction when loaded. Auto-TRUE
        # guards produce deterministic P=1.0 predictions when detect_cheese
        # fires. Both are consumed via last_prediction.
        if use_belief and bot.belief_state.strategy is not None:
            belief_pred = bot.belief_state.strategy.last_prediction
            if belief_pred is not None and self._meets_threshold(belief_pred):
                prediction = belief_pred

        # Fallback: rule-based detection (only when belief is disabled)
        # Per the architecture, rules are the last resort when the model is
        # unavailable — not when the model says "low confidence".
        elif not use_belief and detect_cheese(bot):
            prediction = self._prediction_from_rules(bot)

        # Route the prediction
        if prediction is not None:
            self._route_prediction(bot, prediction)

    def _meets_threshold(self, prediction) -> bool:
        """Check if a prediction's top category meets the confidence threshold.

        Applies uniformly to all non-MACRO categories: P(cheese) ≥ 0.6,
        P(all_in) ≥ 0.6, P(timing_attack) ≥ 0.6. MACRO is never routed as a
        reaction (it's the default state, handled in _route_prediction).
        """
        top_prob = max(prediction.probs.values())
        return top_prob >= self.REACTION_CONFIDENCE_THRESHOLD

    def execute(self, bot: "PiG_Bot") -> None:
        """Run the active reaction's handler. Called from bot.py on_step."""
        if not self.is_active:
            return

        # Check transition/deactivation every frame
        self._check_deactivation(bot)

        if not self.is_active:
            return  # Deactivated during check

        handler = self._get_handler()
        if handler is not None:
            handler(bot)

    def deactivate(self, bot: "PiG_Bot") -> None:
        """Centralized cleanup for any reaction. Resets all state."""
        # Return defending workers to gathering
        defending_workers = bot.mediator.get_units_from_role(
            role=UnitRole.DEFENDING,
            unit_type=UnitTypeId.PROBE,
        )
        for worker in defending_workers:
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

        # Complete cheese reaction build if active
        if (bot.build_order_runner.chosen_opening == "Cheese_Reaction_Build"
                and not bot.build_order_runner.build_completed):
            bot.build_order_runner.set_build_completed()

        # Reset state
        self._active_category = None
        self._active_strategy = None
        self._reaction_start_time = -1.0
        self._transitioned = False
        self._last_threat_near_base_time = -1.0
        bot._under_attack = False

    # --- Internal routing ---

    def _route_prediction(self, bot: "PiG_Bot", prediction) -> None:
        """Route a StrategyPrediction to the appropriate handler."""
        category = prediction.label
        level2 = prediction.level2

        # Only activate for non-macro categories
        if category == StrategyCategory.MACRO:
            if self.is_active:
                self._check_deactivation(bot)
            return

        # Determine handler
        handler = self._resolve_handler(category, level2)

        # If same reaction already active, no change needed
        if (self._active_category == category
                and self._active_strategy == level2
                and handler is not None):
            return

        # Category changed — full transition with cleanup
        if self._active_category is not None and self._active_category != category:
            self.deactivate(bot)

        # Activate new reaction
        self._active_category = category
        self._active_strategy = level2
        self._reaction_start_time = bot.time
        self._transitioned = False
        self._last_threat_near_base_time = -1.0

        # Apply category config: build order switch
        config = self.category_config
        if config.build is not None:
            remove_completed = bot.structures(UnitTypeId.CYBERNETICSCORE).exists
            bot.build_order_runner.switch_opening(config.build, remove_completed=remove_completed)

            # Cancel fast-expanding Nexus if category says to
            if config.cancel_nexus:
                pending_townhalls = cy_structure_pending_ares(bot, UnitTypeId.NEXUS)
                if pending_townhalls == 1 and bot.time < 2 * 60:
                    for pt in bot.townhalls.not_ready:
                        bot.mediator.cancel_structure(structure=pt)

        # Set under_attack flag for threat detection
        bot._under_attack = True

    def _resolve_handler(self, category: StrategyCategory, level2: str) -> Optional[Callable]:
        """Look up handler by (category, level2), fall back to category default."""
        category_handlers = self._handlers.get(category, {})
        handler = category_handlers.get(level2)
        if handler is not None:
            return handler
        return self._category_defaults.get(category)

    def _get_handler(self) -> Optional[Callable]:
        """Get the handler for the currently active reaction."""
        if self._active_category is None or self._active_strategy is None:
            return None
        return self._resolve_handler(self._active_category, self._active_strategy)

    def _check_deactivation(self, bot: "PiG_Bot") -> None:
        """Check if the active reaction should deactivate or transition."""
        if not self.is_active:
            return

        # Check strategy-specific deactivation (e.g., no more enemy workers near base)
        if self._active_strategy in self._deactivation_checks:
            should_deactivate = self._deactivation_checks[self._active_strategy](bot)
            if should_deactivate:
                self.deactivate(bot)
                return

        # Category-default: cheese transitions to standard army (not full deactivation)
        # Transition = keep reaction active but switch army comp; deactivation = end reaction entirely
        if self._active_category == StrategyCategory.CHEESE and not self._transitioned:
            # Track threat presence for the sustained-clear window
            if _enemy_combat_near_bases(bot):
                self._last_threat_near_base_time = bot.time

            # Clear when any of:
            # 1. Game reached mid-game (6:00 timer — unconditional safety net)
            # 2. Sustained threat-free window (no enemy combat units near
            #    our bases for CHEESE_THREAT_CLEAR_GRACE seconds) — means
            #    the cheese attack is broken and we can safely transition
            # 3. Army has commenced attacking (we pushed out — defensive
            #    phase is over)
            # The economy check was removed: it created a circular
            # dependency where cheese mode kept workers low, which kept
            # economy_state "reduced", which blocked the transition.
            threat_clear = (
                self._last_threat_near_base_time >= 0.0
                and (bot.time - self._last_threat_near_base_time)
                    >= CHEESE_THREAT_CLEAR_GRACE
            )
            attack_commenced = getattr(bot, '_commenced_attack', False)
            if bot.game_state >= 1 or threat_clear or attack_commenced:
                self._transitioned = True

    def _prediction_from_rules(self, bot: "PiG_Bot"):
        """Build a synthetic StrategyPrediction from rule-based detection labels.

        Used as fallback when Strategy Belief is disabled.
        """
        from bot.belief.strategy_belief import StrategyPrediction

        # Worker rush (all races)
        if getattr(bot, '_worker_rush_detected', False):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="worker_rush",
                source="rules:worker_rush",
                game_time=bot.time,
            )

        # Cannon rush
        if getattr(bot, '_protoss_strategy_label', 'none') == 'cannon_rush':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="cannon_rush",
                source="rules:cannon_rush",
                game_time=bot.time,
            )

        # Generic cheese (any other detection)
        cheese_label = getattr(bot, '_cheese_label', 'none')
        if cheese_label != 'none':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.8,
                    StrategyCategory.ALL_IN: 0.2,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2=cheese_label,
                source="rules:cheese",
                game_time=bot.time,
            )

        # Zerg all-in
        zerg_label = getattr(bot, '_zerg_allin_label', 'none')
        if zerg_label != 'none':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.1,
                    StrategyCategory.ALL_IN: 0.7,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2=zerg_label,
                source="rules:zerg_allin",
                game_time=bot.time,
            )

        # Terran strategy
        terran_label = getattr(bot, '_terran_strategy_label', 'none')
        if terran_label != 'none':
            if terran_label in ('proxy_rax', 'bunker_rush'):
                return StrategyPrediction(
                    probs={
                        StrategyCategory.CHEESE: 0.8,
                        StrategyCategory.ALL_IN: 0.2,
                        StrategyCategory.TIMING_ATTACK: 0.0,
                        StrategyCategory.MACRO: 0.0,
                    },
                    label=StrategyCategory.CHEESE,
                    level2=terran_label,
                    source="rules:terran_cheese",
                    game_time=bot.time,
                )
            if terran_label in ('all_in', 'marine_rush', 'one_base_all_in', 'two_base_all_in'):
                return StrategyPrediction(
                    probs={
                        StrategyCategory.CHEESE: 0.2,
                        StrategyCategory.ALL_IN: 0.6,
                        StrategyCategory.TIMING_ATTACK: 0.2,
                        StrategyCategory.MACRO: 0.0,
                    },
                    label=StrategyCategory.ALL_IN,
                    level2=terran_label,
                    source="rules:terran_allin",
                    game_time=bot.time,
                )
            if terran_label in ('bio_timing', 'tank_timing', 'widow_mine_drop'):
                return StrategyPrediction(
                    probs={
                        StrategyCategory.CHEESE: 0.0,
                        StrategyCategory.ALL_IN: 0.15,
                        StrategyCategory.TIMING_ATTACK: 0.75,
                        StrategyCategory.MACRO: 0.1,
                    },
                    label=StrategyCategory.TIMING_ATTACK,
                    level2=terran_label,
                    source="rules:terran_timing",
                    game_time=bot.time,
                )

        # Protoss strategy
        protoss_label = getattr(bot, '_protoss_strategy_label', 'none')
        if protoss_label != 'none':
            if protoss_label in ('four_gate', 'six_gate', 'all_in', 'two_base_colossus', 'two_base_all_in', 'one_base_all_in'):
                return StrategyPrediction(
                    probs={
                        StrategyCategory.CHEESE: 0.1,
                        StrategyCategory.ALL_IN: 0.7,
                        StrategyCategory.TIMING_ATTACK: 0.2,
                        StrategyCategory.MACRO: 0.0,
                    },
                    label=StrategyCategory.ALL_IN,
                    level2=protoss_label,
                    source="rules:protoss_allin",
                    game_time=bot.time,
                )
            if protoss_label == 'stargate_timing':
                return StrategyPrediction(
                    probs={
                        StrategyCategory.CHEESE: 0.0,
                        StrategyCategory.ALL_IN: 0.15,
                        StrategyCategory.TIMING_ATTACK: 0.75,
                        StrategyCategory.MACRO: 0.1,
                    },
                    label=StrategyCategory.TIMING_ATTACK,
                    level2=protoss_label,
                    source="rules:protoss_timing",
                    game_time=bot.time,
                )

        # No detection
        return None


# ===== HELPER FUNCTIONS =====


def _should_deactivate_worker_rush(bot: "PiG_Bot") -> bool:
    """No enemy workers near our base → worker rush is over."""
    defense_point = (bot.natural_expansion
                     if bot.structures.closer_than(8, bot.natural_expansion)
                     else bot.start_location)
    enemy_units = bot.mediator.get_units_in_range(
        start_points=[defense_point],
        distances=25,
        query_tree=UnitTreeQueryType.AllEnemy,
    )[0]
    enemy_workers = enemy_units.filter(lambda u: u.type_id in WORKER_TYPES)
    return not bool(enemy_workers)


def _should_deactivate_cannon_rush(bot: "PiG_Bot") -> bool:
    """No enemy probes/cannons/pylons near our base → cannon rush is over."""
    enemy_units = bot.mediator.get_units_in_range(
        start_points=[bot.start_location],
        distances=14,
        query_tree=UnitTreeQueryType.AllEnemy,
    )[0]
    enemy_probes = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PROBE)
    enemy_cannons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PHOTONCANNON)
    enemy_pylons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PYLON)
    return not bool(enemy_probes or enemy_cannons or enemy_pylons)


def _enemy_combat_near_bases(bot: "PiG_Bot") -> bool:
    """Any enemy combat unit in the ARES near-bases dict → threat is present.

    Uses the same source as threat_detection() so the transition is
    consistent with the _under_attack flag.  Workers and non-combat
    units are excluded by EnemyToBaseManager's filtering.
    """
    ground = bot.mediator.get_ground_enemy_near_bases
    flying = bot.mediator.get_flying_enemy_near_bases
    return any(ground.values()) or any(flying.values())


# ===== HANDLER FUNCTIONS =====
# Pure per-frame logic. No flag management — the ReactionManager owns all state.


def defend_cannon_rush(bot):
    """Defend against cannon rush by pulling workers and targeting threats.

    Pure per-frame logic — no flag management. The ReactionManager handles
    activation, deactivation, and cleanup. This function only does the
    micro: pull workers, prioritize targets, auto-return to mining.

    Deactivation is handled by _should_deactivate_cannon_rush() which
    checks for absence of enemy probes/cannons/pylons near our base.
    """
    # Get enemy units in base area
    enemy_units: Units = bot.mediator.get_units_in_range(
        start_points=[bot.start_location],
        distances=14,
        query_tree=UnitTreeQueryType.AllEnemy,
    )[0]

    enemy_probes = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PROBE)
    enemy_cannons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PHOTONCANNON)
    enemy_pylons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PYLON)

    # Calculate how many workers to pull (cannon-specific formula)
    workers_needed = min(24, len(enemy_cannons) + (len(enemy_probes) // 2) + 8)

    # Get current defending workers
    defending_workers = bot.mediator.get_units_from_role(
        role=UnitRole.DEFENDING,
        unit_type=UnitTypeId.PROBE
    )

    # Get workers that should be mining (not already defending)
    available_workers = bot.workers.filter(
        lambda w: w.tag not in defending_workers.tags
    )

    # Assign more workers if needed
    while len(defending_workers) < workers_needed and available_workers:
        worker = available_workers.closest_to(bot.start_location)
        if not worker:
            break
        bot.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
        defending_workers.append(worker)
        available_workers.remove(worker)

    # Per-worker control with priority chain
    for worker in defending_workers:
        # 1. Handle resource return
        if worker.is_carrying_resource and bot.townhalls:
            worker.return_resource()
            continue

        # 2. Cannon-specific prioritization
        # Prioritize cannons that are nearly complete or complete
        urgent_targets = enemy_cannons.filter(
            lambda c: c.build_progress > 0.5 or c.is_ready
        )

        if urgent_targets:
            target = cy_closest_to(worker.position, urgent_targets)
            worker.attack(target)
            continue

        if enemy_probes:
            target = cy_closest_to(worker.position, enemy_probes)
            # Only attack if in range and ready (smarter targeting)
            if cy_attack_ready(bot, worker, target):
                worker.attack(target)
            else:
                worker.move(target.position)
            continue

        if enemy_cannons:  # Cannons < 50% complete
            target = cy_closest_to(worker.position, enemy_cannons)
            worker.attack(target)
            continue

        if enemy_pylons:
            target = cy_closest_to(worker.position, enemy_pylons)
            worker.attack(target)
            continue

        # 3. Automatic fallback to mining
        if bot.mineral_field:
            mf = cy_closest_to(worker.position, bot.mineral_field)
            worker.gather(mf)
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

def defend_worker_rush(bot):
    """Defend against worker rush by pulling workers and kiting.

    Pure per-frame logic — no flag management. The ReactionManager handles
    activation, deactivation, and cleanup. This function only does the
    micro: pull workers, kite enemies, auto-return to mining.

    Deactivation is handled by _should_deactivate_worker_rush() which
    checks for absence of enemy workers near our base.
    """
    # Get all enemy units in our base and filter for workers
    defense_point = bot.natural_expansion if bot.structures.closer_than(8, bot.natural_expansion) else bot.start_location

    enemy_units = bot.mediator.get_units_in_range(
        start_points=[defense_point],
        distances=25,  # Larger radius to catch workers coming in
        query_tree=UnitTreeQueryType.AllEnemy,
    )[0]
    enemy_workers = enemy_units.filter(lambda u: u.type_id in WORKER_TYPES)

    # Get current defending workers
    defending_workers = bot.mediator.get_units_from_role(
        role=UnitRole.DEFENDING,
        unit_type=UnitTypeId.PROBE
    )

    # Calculate how many workers to pull (worker rush specific: 1.5x enemy workers)
    workers_needed = min(16, max(4, int(len(enemy_workers) * 1.5)))

    # Get workers that should be mining (not already defending)
    available_workers = bot.workers.filter(
        lambda w: w.tag not in defending_workers.tags
    )

    # Assign more workers if needed
    while len(defending_workers) < workers_needed and available_workers:
        worker = available_workers.closest_to(bot.start_location)
        if not worker:
            break
        bot.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
        defending_workers.append(worker)
        available_workers.remove(worker)

    # Per-worker control with priority chain
    for worker in defending_workers:
        # 1. Handle resource return
        if worker.is_carrying_resource and bot.townhalls:
            worker.return_resource()
            continue

        # 2. Worker rush specific: use WorkerKiteBack for micro
        if enemy_workers:
            target = cy_closest_to(worker.position, enemy_workers)
            bot.register_behavior(WorkerKiteBack(unit=worker, target=target))
            continue

        # 3. Automatic fallback to mining
        if bot.mineral_field:
            mf = cy_closest_to(worker.position, bot.mineral_field)
            worker.gather(mf)
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)


def cheese_reaction(bot):
    """Generic cheese defense handler.

    The build order switch and Nexus cancel are handled by ReactionManager
    via CategoryConfig. This function is the category default for CHEESE
    when no specific handler (worker_rush, cannon_rush) matches.

    Currently a no-op — per-frame micro for generic cheese (12-pool,
    speedling, proxy rax, etc.) is handled by the build order switch
    and CHEESE_DEFENSE_ARMY composition. Future: add specific micro
    for proxy defense, wall-off logic, etc.
    """
    pass


# ===== THREAT ASSESSMENT FUNCTIONS =====
# Moved from combat.py for better separation of concerns

def assess_threat_severity(bot, enemy_units: Units, threat_location: Point2) -> dict:
    """
    Enhanced threat assessment that determines both severity and appropriate response.
    Returns dict with threat_level, required_units, and response_type.
    """
    if not enemy_units:
        return {"threat_level": 0, "required_units": 0, "response_type": "none"}
    
    # Calculate base threat value
    threat_value = 0
    unit_count = enemy_units.amount
    
    # Categorize threats by type
    harassment_units = enemy_units.filter(lambda u: u.type_id in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG
    })
    
    combat_units = enemy_units.filter(lambda u: u.type_id not in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG,
        UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE
    })
    
    # Weight different unit types
    for unit in enemy_units:
        if unit.type_id in UNIT_DATA:
            base_weight = UNIT_DATA[unit.type_id]['army_value']
            # Scale by health percentage
            health_factor = (unit.health + unit.shield) / (unit.health_max + unit.shield_max)
            threat_value += base_weight * health_factor
    
    # Distance factor - closer threats are more urgent
    closest_base = min(bot.townhalls, key=lambda th: cy_distance_to_squared(th.position, threat_location))
    distance_to_base = cy_distance_to(threat_location, closest_base.position)
    distance_factor = max(0.5, 2.0 - (distance_to_base / 20.0))
    threat_value *= distance_factor
    
    # Determine response type and required units
    if harassment_units.amount >= unit_count * 0.8 and unit_count <= 6:
        # Mostly harassment units
        response_type = "harassment_response"
        required_units = min(4, max(2, unit_count))
        threat_level = min(3, threat_value)
    elif combat_units.amount > 3 or threat_value > 15:
        # Significant combat threat
        response_type = "combat_response" 
        required_units = max(6, int(threat_value * 0.8))
        threat_level = min(10, threat_value)
    else:
        # Mixed or small threat
        response_type = "patrol_response"
        required_units = min(3, max(1, unit_count // 2))
        threat_level = min(5, threat_value)
    
    return {
        "threat_level": int(threat_level),
        "required_units": required_units,
        "response_type": response_type,
        "harassment_ratio": harassment_units.amount / max(1, unit_count),
        "unit_count": unit_count,
        "threat_value": threat_value
    }


def assess_threat(bot, enemy_units: Units, own_forces: Units, return_details: bool = False):
    """
    Enhanced threat assessment that can return simple score or detailed analysis.
    
    Args:
        return_details: If True, returns dict with detailed analysis. If False, returns int score.
    """
    if not enemy_units:
        return {"threat_level": 0, "required_units": 0, "response_type": "none"} if return_details else 0
    
    # Calculate base threat value
    threat_value = 0
    unit_count = enemy_units.amount
    
    # Check for damage-dealing low threats (e.g., lings attacking probes/buildings)
    damage_bonus = _assess_damage_threat(bot, enemy_units)
    
    # Categorize threats by type  
    harassment_units = enemy_units.filter(lambda u: u.type_id in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG
    })
    
    # Weight different unit types
    for unit in enemy_units:
        if unit.type_id in UNIT_DATA:
            base_weight = UNIT_DATA[unit.type_id]['army_value']
            # Scale by health percentage
            health_factor = (unit.health + unit.shield) / max(1, unit.health_max + unit.shield_max)
            threat_value += base_weight * health_factor
    
    # Density check: adjust threat level based on enemy clustering
    center = cy_center(enemy_units)
    cluster_count = 0
    for unit in enemy_units:
        if cy_distance_to(unit.position, center) <= 5.0:
            cluster_count += 1
    
    # If fewer than 3 enemy units are clustered, scale down the threat level
    if cluster_count < 3:
        threat_value *= 0.5
    
    # Apply damage bonus for units actively damaging our assets
    threat_value += damage_bonus
    
    # Simple integer return for backward compatibility
    if not return_details:
        return max(round(threat_value), 0)
    
    # Enhanced details for new system
    # (This replaces assess_threat_severity functionality)
    if len(bot.townhalls) > 0:
        closest_base = min(bot.townhalls, key=lambda th: cy_distance_to_squared(th.position, center))
        distance_to_base = cy_distance_to(center, closest_base.position)
        distance_factor = max(0.5, 2.0 - (distance_to_base / 20.0))
        threat_value *= distance_factor
    
    # Apply damage bonus again after distance factor (for detailed analysis)
    threat_value += damage_bonus
    
    # Determine response type and required units
    # Damage-dealing threats get elevated response even if low unit count
    if damage_bonus > 2.0:  # Critical damage being dealt
        response_type = "damage_response"
        required_units = min(3, max(1, unit_count + 1))  # +1 extra for damage threats
        threat_level = min(6, max(3, threat_value))  # Minimum level 3 for damage threats
    elif harassment_units.amount >= unit_count * 0.8 and unit_count <= 6:
        response_type = "harassment_response"
        required_units = min(4, max(2, unit_count))
        threat_level = min(3, threat_value)
    elif unit_count > 3 and threat_value > 10:
        response_type = "combat_response" 
        required_units = max(6, int(threat_value * 0.8))
        threat_level = min(10, threat_value)
    else:
        response_type = "patrol_response"
        required_units = min(3, max(1, unit_count // 2))
        threat_level = min(5, threat_value)
    
    return {
        "threat_level": int(threat_level),
        "required_units": required_units,
        "response_type": response_type,
        "harassment_ratio": harassment_units.amount / max(1, unit_count),
        "unit_count": unit_count,
        "threat_value": threat_value
    }


def _assess_damage_threat(bot, enemy_units: Units) -> float:
    """
    Assess if low-threat units are actively doing damage to our assets.
    Returns bonus threat value for units near damaged/low-health friendly units.
    """
    damage_bonus = 0.0
    
    # Get our damaged units (buildings + workers)
    damaged_buildings = bot.structures.filter(lambda s: s.health_percentage < 1.0)
    damaged_workers = bot.workers.filter(lambda w: w.health_percentage < 1.0)
    
    # Get very low health units that need immediate attention
    critical_buildings = bot.structures.filter(lambda s: s.health_percentage < 0.5)
    critical_workers = bot.workers.filter(lambda w: w.health_percentage < 0.3)
    
    # Check if enemy units are near damaged assets
    for enemy in enemy_units:
        # Higher bonus for units near critical assets
        if critical_buildings:
            closest_critical = critical_buildings.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_critical.position) <= 3.0:
                damage_bonus += 3.0  # High priority for units attacking critical buildings
        
        if critical_workers:
            closest_critical_worker = critical_workers.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_critical_worker.position) <= 2.0:
                damage_bonus += 2.0  # Medium priority for units attacking low-health workers
        
        # Medium bonus for units near damaged assets
        if damaged_buildings:
            closest_damaged = damaged_buildings.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_damaged.position) <= 3.0:
                damage_bonus += 1.5
        
        if damaged_workers:
            closest_damaged_worker = damaged_workers.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_damaged_worker.position) <= 2.0:
                damage_bonus += 1.0
    
    return damage_bonus


def allocate_defensive_forces(bot, threat_info: dict, threat_location: Point2, enemy_units: Units) -> Units:
    """
    Smart unit allocation based on threat value and air/ground requirements.
    Uses army_value matching and capability filtering like the examples.
    """
    available_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    
    if not available_units:
        return Units([], bot)
    
    # Dynamic threat composition check (following example patterns)
    has_air = any(unit.is_flying for unit in enemy_units)
    has_ground = any(not unit.is_flying for unit in enemy_units)
    
    # Filter for units that can actually attack (exclude disruptors, etc.)
    combat_capable = available_units.filter(lambda u: u.can_attack)
    
    # Filter capable units (exactly like examples)
    if has_air and has_ground:
        # Mixed threat - prioritize units that can attack both, fallback to air-capable
        capable_units = combat_capable.filter(lambda u: u.can_attack_both)
        if not capable_units:
            capable_units = combat_capable.filter(lambda u: u.can_attack_air)
    elif has_air:
        capable_units = combat_capable.filter(lambda u: u.can_attack_air)
    else:
        capable_units = combat_capable.filter(lambda u: u.can_attack_ground)
    
    if not capable_units:
        # Fallback to any combat capable units if no specific match found
        capable_units = combat_capable
    
    response_type = threat_info["response_type"]
    threat_value = threat_info.get("threat_value", 0)
    
    # For damage-dealing threats, send closest capable units immediately
    if response_type == "damage_response":
        closest_units = capable_units.sorted(lambda u: cy_distance_to_squared(u.position, threat_location))
        required_count = min(threat_info["required_units"], len(closest_units))
        return closest_units[:required_count]
    
    # Value-based allocation for proportional response
    selected_units = []
    current_value = 0.0
    target_value = max(threat_value, 1.0)  # Match threat value exactly
    
    # Sort by distance (closest units respond, regardless of value)
    sorted_units = capable_units.sorted(lambda u: cy_distance_to_squared(u.position, threat_location))
    
    # Unit allocation logic
    
    # Select units until threat is adequately countered
    for unit in sorted_units:
        unit_value = UNIT_DATA.get(unit.type_id, {}).get('army_value', 1.0)
        
        # Add this unit
        selected_units.append(unit)
        current_value += unit_value
        
        # Track current allocation value
        
        # Stop when we have enough value OR enough units for safety
        if current_value >= target_value or len(selected_units) >= 3:
            break
    
    # Ensure minimum response for non-trivial threats
    if not selected_units and capable_units:
        selected_units = [capable_units.closest_to(threat_location)]
    
    return Units(selected_units, bot)


def threat_detection(bot, main_army: Units) -> None:
    """
    Enhanced threat detection with smart force allocation.
    Detects threats near bases and responds with appropriate force levels
    instead of always redirecting the entire main army.
    """
    # Import here to avoid circular dependency
    from bot.combat import control_main_army
    
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases

    # Combine ground + air threats
    all_threats = {}
    for key, value in ground_near.items():
        all_threats[key] = value.copy()
    for key, value in flying_near.items():
        all_threats.setdefault(key, set()).update(value)
    
    # Aggregate total threat value from all bases for global _under_attack flag
    total_threat_value = 0
    if all_threats:
        all_enemy_tags = set()
        for enemy_tags in all_threats.values():
            all_enemy_tags.update(enemy_tags)
        all_threatening_units = bot.enemy_units.tags_in(all_enemy_tags)
        
        if all_threatening_units:
            threat_info = assess_threat(bot, all_threatening_units, main_army, return_details=True)
            assert isinstance(threat_info, dict), "assess_threat with return_details=True should return dict"
            total_threat_value = threat_info.get("threat_value", 0)
    
    # Calculate threat ratio: (threat near bases) / (total known enemy army)
    # Filters: workers, scouts (overlords/observers), and non-combat units
    # When composition belief is enabled, use probability-weighted army value instead of
    # binary age cutoff — stale units contribute proportionally to their P(exists)
    use_belief = bot.config.get("Belief", {}).get("enable_composition", False)
    if use_belief:
        weighted_army = bot.belief_state.composition.get_weighted_army(
            exclude_workers=True, exclude_ignored=True, min_confidence=0.01,
        )
        known_army_value = sum(
            UNIT_DATA.get(wu.unit.type_id, {}).get('army_value', 1.0) * wu.confidence
            for wu in weighted_army
            if not wu.unit.is_structure
        )
    else:
        known_enemy = bot.mediator.get_cached_enemy_army
        known_army_value = sum(
            UNIT_DATA.get(u.type_id, {}).get('army_value', 1.0) 
            for u in known_enemy 
            if u.type_id not in WORKER_TYPES and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            and u.age < MEMORY_EXPIRY_TIME
        ) if known_enemy else 0
    threat_ratio = total_threat_value / max(known_army_value, 1.0)
    
    # Strategy-aware thresholds: known cheesers trigger defensive posture earlier
    # and hold it longer. Multipliers lower both trigger and clear thresholds.
    use_strategy = (
        bot.config.get("Belief", {}).get("enable_strategy", False)
        and bot.belief_state.strategy is not None
        and bot.belief_state.strategy.last_prediction is not None
    )
    if use_strategy:
        prediction = bot.belief_state.strategy.last_prediction
        dominant = max(prediction.probs, key=prediction.probs.get)
        trigger_mult = STRATEGY_THREAT_MULTIPLIER.get(dominant, 1.0)
        clear_mult = STRATEGY_THREAT_CLEAR_MULTIPLIER.get(dominant, 1.0)
        effective_trigger = UNDER_ATTACK_VALUE_THRESHOLD * trigger_mult
        effective_clear = UNDER_ATTACK_CLEAR_VALUE * clear_mult
    else:
        effective_trigger = UNDER_ATTACK_VALUE_THRESHOLD
        effective_clear = UNDER_ATTACK_CLEAR_VALUE

    # Update _under_attack flag with hysteresis (higher threshold to set, lower to clear)
    # Runs every frame even when no threats detected to allow flag to clear properly
    if bot._under_attack:
        if total_threat_value < effective_clear:
            bot._under_attack = False
    else:
        if total_threat_value >= effective_trigger:
            bot._under_attack = True
        elif threat_ratio >= UNDER_ATTACK_RATIO_THRESHOLD:
            bot._under_attack = True

    # Detect cloaked/burrowed enemies near bases that need detection
    # These may not appear in ground_near/flying_near if partially visible
    # but still pose a threat our defenders can't fight without an observer
    cloaked_threat_positions: list[Point2] = []
    if bot.townhalls:
        th_positions = [th.position for th in bot.townhalls]
        enemies_near_bases = bot.mediator.get_units_in_range(
            start_points=th_positions,
            distances=18,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )
        for result in enemies_near_bases:
            for enemy in result:
                if not (enemy.is_cloaked or enemy.is_burrowed):
                    continue
                if enemy.is_revealed:
                    continue
                cloaked_threat_positions.append(enemy.position)
    bot._cloaked_threat_positions = cloaked_threat_positions

    # Per-base defender allocation
    if all_threats:
        main_army_should_respond = False
        
        for base_location, enemy_tags in all_threats.items():
            enemy_units = bot.enemy_units.tags_in(enemy_tags)
            if not enemy_units:
                continue
                
            threat_position, _ = cy_find_units_center_mass(enemy_units, 10.0)
            threat_position = Point2(threat_position)
            
            # Assess threat level for this specific base location
            threat_info = assess_threat(bot, enemy_units, main_army, return_details=True)
            assert isinstance(threat_info, dict), "assess_threat with return_details=True should return dict"
            base_threat_value = threat_info.get("threat_value", 0)
            
            # Allocate defenders proportionally to threat level with capped response
            if base_threat_value > 0:
                existing_defenders = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
                
                # Cap defenders at 150% of threat value (minimum 10) to avoid over-committing
                existing_defender_value = sum(UNIT_DATA.get(u.type_id, {}).get('army_value', 1.0) for u in existing_defenders)
                max_defender_value = max(total_threat_value * 1.5, 10.0)
                
                defender_cap_reached = existing_defender_value >= max_defender_value
                
                # Assign new defenders only if below cap (unit control handled in combat.py)
                if not defender_cap_reached:
                    new_defenders = allocate_defensive_forces(bot, threat_info, threat_position, enemy_units)
                    if new_defenders:
                        for unit in new_defenders:
                            bot.mediator.assign_role(tag=unit.tag, role=UnitRole.BASE_DEFENDER)
                
                # Store threat position for defensive targeting in combat systems
                bot._defender_threat_position = threat_position
        
        # Flag main army involvement for coordination with attack logic
        bot._main_army_defending = main_army_should_respond
