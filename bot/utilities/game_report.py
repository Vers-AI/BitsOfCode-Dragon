"""
Game Report Utility
Purpose: Handles presentation and formatting of reports AND telemetry events.
Key Decisions: print_* functions retained for dev console readability; TELEM emissions
              added alongside for structured data collection (see telemetry_plan.md).
Limitations: Transition events only emit on value change; periodic snapshots sample by env.
"""

from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole, WORKER_TYPES, TIE_OR_BETTER
from sc2.data import Race
from cython_extensions import cy_distance_to
from bot.constants import StrategyCategory, STRATEGY_LABELS
from bot.managers.macro import get_economy_state
from bot.utilities.telemetry import (
    log_event, log_transition, log_match, log_event_no_sample,
)


def _bool_to_int(value: bool, seen: bool) -> int:
    """Encode a position boolean for telemetry: 1=yes, 0=no (seen but not near), -1=unknown."""
    if value:
        return 1
    return 0 if seen else -1


def _cannon_near_base_int(bot) -> int:
    """Encode cannon-near-base as int. No persistent attr exists, so derive at game end.

    Mirrors strategy_belief.py: checks _cannon_rush_active (reaction manager flag)
    and falls back to geometric proximity of visible cannons.
    """
    if getattr(bot, "_cannon_rush_active", False):
        return 1
    cannons = [s for s in bot.enemy_structures
               if s.type_id == UnitTypeId.PHOTONCANNON]
    if not cannons:
        return -1
    our_nat = bot.mediator.get_own_nat
    near = any(
        cy_distance_to(c.position, bot.start_location) < 25.0
        or cy_distance_to(c.position, our_nat) < 25.0
        for c in cannons
    )
    return 1 if near else 0


def _get_cheese_type(bot) -> str:
    """Derive strategy label from detection flags (first match wins).

    Covers all labels that strategy_detect.py can set, so the fallback
    path in get_replay_tags_to_send() has full coverage even when the
    belief system is disabled. Priority: specific labels > ARES mediator
    booleans > generic labels.
    """
    # Zerg: ling rush labels
    if (bot.enemy_race in {Race.Zerg, Race.Random}
            and hasattr(bot, '_cheese_detected') and bot._cheese_detected
            and hasattr(bot, '_cheese_label') and bot._cheese_label in {'12_pool', 'speedling'}):
        return bot._cheese_label
    # Zerg: all-in labels
    if (bot.enemy_race in {Race.Zerg, Race.Random}
            and hasattr(bot, '_zerg_allin_label')
            and bot._zerg_allin_label not in {'none', ''}):
        return bot._zerg_allin_label
    # Zerg: timing labels
    if (bot.enemy_race in {Race.Zerg, Race.Random}
            and hasattr(bot, '_zerg_timing_label')
            and bot._zerg_timing_label not in {'none', ''}):
        return bot._zerg_timing_label
    # Terran: strategy labels
    if (bot.enemy_race == Race.Terran
            and hasattr(bot, '_terran_strategy_label')
            and bot._terran_strategy_label not in {'none', ''}):
        return bot._terran_strategy_label
    # Protoss: strategy labels
    if (bot.enemy_race == Race.Protoss
            and hasattr(bot, '_protoss_strategy_label')
            and bot._protoss_strategy_label not in {'none', ''}):
        return bot._protoss_strategy_label
    # ARES mediator booleans (race-agnostic, highest priority overrides)
    if bot.reaction_manager.active_reaction_name == "worker_rush":
        return "worker_rush"
    if bot.reaction_manager.active_reaction_name == "cannon_rush":
        return "cannon_rush"
    if bot.mediator.get_enemy_marine_rush:
        return "marine_rush"
    if bot.mediator.get_enemy_marauder_rush:
        return "marauder_rush"
    if bot.mediator.get_is_proxy_zealot:
        return "proxy_zealot"
    if bot.mediator.get_enemy_four_gate:
        return "four_gate"
    if bot.mediator.get_enemy_roach_rushed:
        return "roach_rush"
    if bot.mediator.get_enemy_ravager_rush:
        return "ravager_rush"
    return "none"


def _get_fight_result(bot):
    """Get can_win_fight EngagementResult, returning None on error."""
    try:
        own_combat = [u for u in bot.own_army if u.type_id not in WORKER_TYPES]
        enemy_combat = [u for u in bot.enemy_army if u.type_id not in WORKER_TYPES]
        return bot.mediator.can_win_fight(
            own_units=own_combat,
            enemy_units=enemy_combat,
            timing_adjust=True,
            good_positioning=True,
            workers_do_no_damage=True
        )
    except Exception:
        return None


def _get_defender_composition(bot) -> dict[str, int]:
    """Get defender unit type → count as JSON-serializable dict."""
    try:
        defending = bot.mediator.get_units_from_role(role=UnitRole.DEFENDING)
        if not defending:
            return {}
        counts: dict[str, int] = {}
        for unit in defending:
            name = unit.type_id.name
            counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception:
        return {}


def _get_scouted_enemy_units(bot) -> dict[str, int]:
    """Get scouted enemy unit type → count (bot's perception, not reality)."""
    if not bot.enemy_units:
        return {}
    counts: dict[str, int] = {}
    for unit in bot.enemy_units:
        name = unit.type_id.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _get_scouted_enemy_structures(bot) -> dict[str, int]:
    """Get scouted enemy structure type → count (bot's perception, not reality)."""
    if not bot.enemy_structures:
        return {}
    counts: dict[str, int] = {}
    for structure in bot.enemy_structures:
        name = structure.type_id.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _get_active_intel_sources(bot) -> list[str]:
    """Determine which scout types are currently providing vision/intel."""
    sources: list[str] = []

    try:
        # Observer (any role: primary, army, patrol, detection, hunting)
        observer_count = (
            bot.units(UnitTypeId.OBSERVER).amount
            + bot.units(UnitTypeId.OBSERVERSIEGEMODE).amount
        )
        if observer_count > 0:
            sources.append("observer")

        # Build runner worker scout
        br_scouts = bot.mediator.get_units_from_role(
            role=UnitRole.BUILD_RUNNER_SCOUT, unit_type=bot.worker_type
        )
        if br_scouts:
            sources.append("worker_scout_br")

        # Manual worker scout (SCOUTING role + worker type)
        scouts = bot.mediator.get_units_from_role(role=UnitRole.SCOUTING)
        worker_scouts = scouts.filter(lambda u: u.type_id in WORKER_TYPES) if scouts else []
        if worker_scouts:
            sources.append("worker_scout")

        # Hallucinated Phoenix scout
        hallu_phoenix = bot.units(UnitTypeId.PHOENIX).filter(lambda u: u.is_hallucination)
        if hallu_phoenix:
            sources.append("hallu_phoenix")
    except Exception:
        pass

    return sources


def print_startup_report(bot) -> None:
    """Print one-time startup report and initialize telemetry context."""
    print("\n" + "="*60)
    print("  GAME STARTUP REPORT")
    print("="*60)
    print(f"  Map: {bot.game_info.map_name}")
    print(f"  Enemy Race: {bot.enemy_race.name}")
    print(f"  Build Chosen: {bot.build_order_runner.chosen_opening}")
    print(f"  Natural Expansion: {bot.natural_expansion}")
    print(f"  Enemy Natural: {bot.mediator.get_enemy_nat}")
    print(f"  Rush Distance Tier: {bot.rush_distance_tier}")
    rush_time_str = f"{bot._rush_time_seconds:.1f}s" if bot._rush_time_seconds > 0 else "unknown"
    print(f"  Rush Time: {rush_time_str}")

    # Opponent prior (if known from cross-game profiles)
    opponent = getattr(bot._belief_updater, "_opponent", None)
    if opponent is not None:
        opponent_id = getattr(bot, 'opponent_id', None)
        enemy_race = bot.enemy_race.name if hasattr(bot, 'enemy_race') else "Unknown"
        summary = opponent.describe_prior(opponent_id, enemy_race)
        if summary is not None:
            print(f"  Opponent Prior: {summary}")
        print(f"  Opponent Profiles Loaded: {opponent.profile_count}")

    # Map prior (if known from training data)
    map_prior = getattr(bot._belief_updater, "_map_prior", None)
    if map_prior is not None:
        map_summary = map_prior.describe_prior(bot.game_info.map_name)
        if map_summary is not None:
            print(f"  Map Prior: {map_summary}")
        print(f"  Map Priors Loaded: {map_prior.map_count}")

    print("="*60 + "\n")


def print_periodic_intel_report(bot, iteration: int) -> None:
    """Print periodic intelligence report every 30 game seconds + emit telemetry events."""
    if bot.time < 30:
        return

    if not hasattr(bot, '_last_intel_report_time'):
        bot._last_intel_report_time = 0.0

    next_report_time = bot._last_intel_report_time + 30.0
    if bot.time < next_report_time:
        return

    bot._last_intel_report_time = (int(bot.time) // 30) * 30

    report_time = bot._last_intel_report_time
    game_time_minutes = int(report_time // 60)
    game_time_seconds = int(report_time % 60)

    # === Telemetry: Transition events (emit only on change) ===

    log_transition(
        subsystem="combat", action="state_change",
        reason="attack_commenced",
        key="commenced_attack", value=bot._commenced_attack,
        _ts=bot.time,
    )

    log_transition(
        subsystem="combat", action="state_change",
        reason="under_attack",
        key="under_attack", value=bot._under_attack,
        _ts=bot.time,
    )

    _PHASE_TRANSITIONS = {0: "game_start", 1: "early_to_mid", 2: "mid_to_late"}
    phase_reason = _PHASE_TRANSITIONS.get(bot.game_state, "unknown")
    log_transition(
        subsystem="reactions", action="phase_transition",
        reason=phase_reason,
        key="game_phase", value=bot.game_state,
        _ts=bot.time,
    )

    economy_state = get_economy_state(bot)
    log_transition(
        subsystem="economy", action="state_transition",
        reason="economy_state_change",
        key="economy_state", value=economy_state,
        _ts=bot.time,
    )

    # === Telemetry: Combat snapshot ===

    fight_result = _get_fight_result(bot)
    event_fields = {"_ts": bot.time}

    if fight_result is not None:
        event_fields["can_win_fight"] = fight_result in TIE_OR_BETTER

    if hasattr(bot, 'current_attack_target') and bot.current_attack_target:
        event_fields["attack_target"] = str(bot.current_attack_target)

    try:
        attacking = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        defending = bot.mediator.get_units_from_role(role=UnitRole.DEFENDING)
        base_defenders = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)

        event_fields["role_attacking"] = len(attacking)
        event_fields["role_defending"] = len(defending)
        event_fields["role_base_defender"] = len(base_defenders)
        event_fields["defender_composition"] = _get_defender_composition(bot)

        from bot.constants import ATTACKING_SQUAD_RADIUS, DEFENDER_SQUAD_RADIUS
        attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
        defending_squads = bot.mediator.get_squads(role=UnitRole.DEFENDING, squad_radius=ATTACKING_SQUAD_RADIUS)
        base_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)

        event_fields["squads_atk"] = len(attacking_squads)
        event_fields["squads_def"] = len(defending_squads)
        event_fields["squads_base"] = len(base_squads)
    except Exception:
        pass

    log_event(subsystem="combat", action="periodic", reason="timer", **event_fields)

    # === Telemetry: Intel snapshot ===

    enemy_units = _get_scouted_enemy_units(bot)
    enemy_structures = _get_scouted_enemy_structures(bot)
    if enemy_units or enemy_structures:
        intel_fields: dict = {"_ts": bot.time}
        intel_sources = _get_active_intel_sources(bot)
        if intel_sources:
            intel_fields["intel_source"] = intel_sources
        if enemy_units:
            intel_fields["scouted_enemy_units"] = enemy_units
        if enemy_structures:
            intel_fields["scouted_enemy_structures"] = enemy_structures
        intel_fields["visible_enemy_count"] = len(bot.enemy_units) if bot.enemy_units else 0
        log_event(subsystem="intel", action="scout_update", reason="periodic", **intel_fields)

    # === Telemetry: Composition belief snapshot (when enabled) ===
    if bot.config.get("Belief", {}).get("enable_composition", False):
        comp = bot.belief_state.composition
        belief_fields: dict = {"_ts": bot.time}
        belief_fields["belief_freshness"] = round(comp.freshness, 3)
        belief_fields["belief_weighted_count"] = round(comp.total_weighted_count, 1)
        weighted = comp.get_weighted_army(
            exclude_workers=True, exclude_structures=True, exclude_ignored=True,
            min_confidence=0.05,
        )
        belief_fields["belief_visible_units"] = sum(1 for wu in weighted if wu.confidence >= 0.9)
        belief_fields["belief_memory_units"] = len(weighted) - belief_fields["belief_visible_units"]
        belief_fields["belief_expected_types"] = list(comp.get_expected_unit_types().keys())
        log_event(subsystem="belief", action="periodic", reason="composition", **belief_fields)

    # === Telemetry: Strategy belief snapshot (when enabled) ===
    if (bot.config.get("Belief", {}).get("enable_strategy", True)
            and bot.belief_state.strategy is not None):
        pred = bot.belief_state.strategy.last_prediction
        if pred is not None:
            strat_fields: dict = {"_ts": bot.time}
            strat_fields["strategy_label"] = pred.label.value
            strat_fields["strategy_level2"] = pred.level2
            strat_fields["strategy_source"] = pred.source
            strat_fields["p_cheese"] = round(pred.p_cheese, 3)
            strat_fields["p_all_in"] = round(pred.p_all_in, 3)
            strat_fields["p_timing"] = round(pred.p_timing, 3)
            strat_fields["p_macro"] = round(pred.p_macro, 3)
            strat_fields["strategy_confidence"] = round(pred.confidence, 3)
            log_event(subsystem="belief", action="periodic", reason="strategy", **strat_fields)
            # Console output for live validation
            print(f"  Strategy: {pred.label.value}({pred.source}) "
                  f"C:{pred.p_cheese:.0%} A:{pred.p_all_in:.0%} "
                  f"T:{pred.p_timing:.0%} M:{pred.p_macro:.0%} "
                  f"[{pred.level2}]")

    # === Telemetry: Scout VOI staleness snapshot (when enabled) ===
    if bot.config.get("Belief", {}).get("enable_scout_voi", False):
        location_last_seen = getattr(bot, "_location_last_seen", {})
        if location_last_seen:
            voi_fields: dict = {"_ts": bot.time}
            staleness_parts = []
            game_time = bot.time
            for key in sorted(location_last_seen.keys()):
                voi_fields[f"last_seen_{key}"] = round(location_last_seen[key], 1)
                staleness = game_time - location_last_seen[key]
                staleness_parts.append(f"{key}:{staleness:.0f}s")
            log_event(subsystem="belief", action="periodic", reason="scout_voi", **voi_fields)
            print(f"  VOI: {' '.join(staleness_parts[:5])}")

    # === Telemetry: Rush detection transitions (Zerg/Random only) ===
    if bot.enemy_race in {Race.Zerg, Race.Random}:
        _emit_cheese_detection_transitions(bot)

    # === Telemetry: Worker rush transition (any race) ===
    log_transition(
        subsystem="cheese_detect", action="state_change",
        reason="worker_rush",
        key="worker_rush_active", value=bot.reaction_manager.active_reaction_name == "worker_rush",
        _ts=bot.time,
    )

    # === Console report (unchanged) ===

    print("\n" + "="*60)
    print(f"  INTEL REPORT {game_time_minutes}:{game_time_seconds:02d}")
    print("="*60)

    print("\n  COMBAT STATUS:")
    print(f"    Attack Commenced: {bot._commenced_attack}")
    print(f"    Under Attack: {bot._under_attack}")
    print(f"    Cheese Response: {bot.reaction_manager.is_cheese_response}")
    print(f"    Game State: {bot.game_state} ({'Early' if bot.game_state == 0 else 'Mid' if bot.game_state == 1 else 'Late'})")

    if fight_result is not None:
        print(f"    Can Win Fight: {fight_result}")
    else:
        print("    Can Win Fight: Error")

    print("\n  ARMY COMPOSITION:")
    _print_army_composition(bot)

    print("\n  UNIT ROLES:")
    _print_unit_roles(bot)

    print("\n  ENEMY INTELLIGENCE:")
    _print_enemy_intel(bot)

    print("\n  ECONOMY:")
    _print_economy_state(bot)

    if hasattr(bot, 'current_attack_target') and bot.current_attack_target:
        print("\n  TARGETING:")
        print(f"    Current Target: {bot.current_attack_target}")

    if bot.enemy_race in {Race.Zerg, Race.Random}:
        _print_cheese_detection_status(bot)

    # Strategy belief in console report
    if (bot.config.get("Belief", {}).get("enable_strategy", True)
            and bot.belief_state.strategy is not None):
        pred = bot.belief_state.strategy.last_prediction
        if pred is not None:
            print("\n  STRATEGY BELIEF:")
            print(f"    {pred.label.value}({pred.source}) "
                  f"C:{pred.p_cheese:.0%} A:{pred.p_all_in:.0%} "
                  f"T:{pred.p_timing:.0%} M:{pred.p_macro:.0%} "
                  f"[{pred.level2}]")
            if pred.evidence:
                ev = pred.evidence
                print(f"    Evidence: race={ev.get('enemy_race','?')} "
                      f"pool={ev.get('pool_bin','?')} bases={ev.get('bases_bin','?')} "
                      f"rax={ev.get('rax_bin','?')} gw={ev.get('gateway_bin','?')} "
                      f"factory={ev.get('factory_bin','?')} duration={ev.get('duration_bin','?')}")

    print("="*60 + "\n")


def _emit_cheese_detection_transitions(bot) -> None:
    """Emit telemetry transition events for cheese detection state changes."""
    if not hasattr(bot, '_cheese_label'):
        return

    log_transition(
        subsystem="cheese_detect", action="classification",
        reason="label_changed",
        key="cheese_label", value=bot._cheese_label,
        _ts=bot.time,
    )

    if hasattr(bot, '_cheese_detected'):
        log_transition(
            subsystem="cheese_detect", action="classification",
            reason="cheese_detected",
            key="cheese_detected", value=bot._cheese_detected,
            _ts=bot.time,
        )

    if hasattr(bot, '_auto_true_fired') and bot._auto_true_fired:
        log_event_no_sample(
            subsystem="cheese_detect", action="classification",
            reason="auto_true_fired",
            rush_label=getattr(bot, '_cheese_label', 'none'),
            score_12p=getattr(bot, '_score_12p', 0),
            score_speed=getattr(bot, '_score_speed', 0),
            rush_source=getattr(bot, '_cheese_source', None),
            _ts=bot.time,
        )

    if hasattr(bot, '_ml_probs') and bot._ml_probs:
        log_event_no_sample(
            subsystem="cheese_detect", action="ml_update",
            reason="model_evaluated",
            ml_probs=getattr(bot, '_ml_probs', {}),
            ml_confidence=getattr(bot, '_ml_confidence', 0.0),
            _ts=bot.time,
        )


def _build_commitment_lookup() -> dict[str, str]:
    """Build a complete label→commitment mapping from the canonical STRATEGY_LABELS.

    Covers all 4 categories. Extras not in STRATEGY_LABELS come from
    strategy_detect.py and _get_cheese_type() — these are appended below.
    """
    lookup: dict[str, str] = {}
    for category, race_dict in STRATEGY_LABELS.items():
        cat_name = category.value
        for _race, labels in race_dict.items():
            for label in labels:
                lookup[label] = cat_name

    # Labels produced by strategy_detect.py that aren't in STRATEGY_LABELS
    extras: dict[str, str] = {
        "speedling": "cheese",
        "worker_rush": "cheese",
        "marine_rush": "all_in",
        "marauder_rush": "all_in",
        "marauder_push": "all_in",
        "proxy_zealot": "cheese",
        "proxy_gate": "cheese",
        "roach_rush": "all_in",
        "ravager_rush": "all_in",
        "ravager_push": "all_in",
        "one_base_all_in": "all_in",
        "two_base_all_in": "all_in",
        "all_in": "all_in",
        "six_gate": "all_in",
        "stargate_timing": "timing_attack",
        "bio_timing": "timing_attack",
        "tank_timing": "timing_attack",
        "widow_mine_drop": "timing_attack",
        "roach_timing": "timing_attack",
        "ling_bane_timing": "timing_attack",
    }
    lookup.update(extras)
    return lookup


_COMMITMENT_LOOKUP = _build_commitment_lookup()
"""Maps any known strategy label to its commitment category name."""


def get_replay_tags_to_send(bot) -> list[str]:
    """
    Collect replay tags that should be sent this iteration.
    Returns list of tags to send via chat_send().

    Emits a Commitment tag and a Strategy tag from the belief system's
    StrategyPrediction. Each tag is sent only once per game. For macro
    games, only the Commitment tag is emitted (no specific build to tag).

    Falls back to legacy _get_cheese_type() when the belief system is
    disabled or has not yet produced a prediction.
    """
    if not hasattr(bot, '_replay_tags_sent'):
        bot._replay_tags_sent = set()

    tags = []

    pred = None
    if (hasattr(bot, 'belief_state')
            and bot.belief_state is not None
            and bot.belief_state.strategy is not None):
        pred = bot.belief_state.strategy.last_prediction

    if pred is not None:
        # Commitment tag (always sent, even macro)
        cat_key = f"Commitment_{pred.label.value}"
        if cat_key not in bot._replay_tags_sent:
            tags.append(cat_key)
            bot._replay_tags_sent.add(cat_key)

        # Strategy tag (skip for macro — no specific build worth filtering on)
        if (pred.level2
                and pred.level2 not in ("none", "unknown")
                and pred.label != StrategyCategory.MACRO):
            strat_key = f"Strategy_{pred.level2}"
            if strat_key not in bot._replay_tags_sent:
                tags.append(strat_key)
                bot._replay_tags_sent.add(strat_key)
    else:
        # Fallback: legacy cheese detection when belief system is off
        cheese_type = _get_cheese_type(bot)
        if cheese_type != "none":
            commitment = _COMMITMENT_LOOKUP.get(cheese_type, "cheese")
            cat_key = f"Commitment_{commitment}"
            if cat_key not in bot._replay_tags_sent:
                tags.append(cat_key)
                bot._replay_tags_sent.add(cat_key)
            strat_key = f"Strategy_{cheese_type}"
            if strat_key not in bot._replay_tags_sent:
                tags.append(strat_key)
                bot._replay_tags_sent.add(strat_key)

    return tags


def _print_army_composition(bot) -> None:
    """Print breakdown of army unit types and totals."""
    if not bot.own_army:
        print("    No army units")
        return

    unit_counts = {}
    for unit in bot.own_army:
        type_name = unit.type_id.name
        unit_counts[type_name] = unit_counts.get(type_name, 0) + 1

    for unit_type, count in sorted(unit_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {unit_type}: {count}")

    print(f"    Total Army: {len(bot.own_army)} units")


def _print_unit_roles(bot) -> None:
    """Print unit role assignments and squad counts."""
    try:
        attacking = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        defending = bot.mediator.get_units_from_role(role=UnitRole.DEFENDING)
        base_defenders = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)

        print(f"    ATTACKING: {len(attacking)}")
        print(f"    DEFENDING: {len(defending)}")

        if defending:
            defender_counts = {}
            for unit in defending:
                type_name = unit.type_id.name
                defender_counts[type_name] = defender_counts.get(type_name, 0) + 1

            defender_list = ", ".join([f"{count}x{unit_type}" for unit_type, count in sorted(defender_counts.items())])
            print(f"      └─ Defenders: {defender_list}")

        print(f"    BASE_DEFENDER: {len(base_defenders)}")

        from bot.constants import ATTACKING_SQUAD_RADIUS, DEFENDER_SQUAD_RADIUS
        attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
        defending_squads = bot.mediator.get_squads(role=UnitRole.DEFENDING, squad_radius=ATTACKING_SQUAD_RADIUS)
        base_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)

        print(f"    Squads: ATK:{len(attacking_squads)} DEF:{len(defending_squads)} BASE:{len(base_squads)}")
    except Exception as e:
        print(f"    Error getting roles: {e}")


def _print_enemy_intel(bot) -> None:
    """Print scouted enemy units and structures."""
    if not bot.enemy_units and not bot.enemy_structures:
        print("    No enemy scouted")
        return

    if bot.enemy_units:
        unit_counts = {}
        for unit in bot.enemy_units:
            type_name = unit.type_id.name
            unit_counts[type_name] = unit_counts.get(type_name, 0) + 1

        print("    Enemy Units:")
        for unit_type, count in sorted(unit_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {unit_type}: {count}")

    if bot.enemy_structures:
        structure_counts = {}
        for structure in bot.enemy_structures:
            type_name = structure.type_id.name
            structure_counts[type_name] = structure_counts.get(type_name, 0) + 1

        print("    Enemy Structures:")
        for struct_type, count in sorted(structure_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {struct_type}: {count}")


def _print_economy_state(bot) -> None:
    """Print economy state (bases, workers, resources)."""
    gatherers = bot.mediator.get_units_from_role(role=UnitRole.GATHERING)
    economy_state = get_economy_state(bot)

    print(f"    State: {economy_state}")
    print(f"    Bases: {len(bot.townhalls)}")
    print(f"    Workers: {len(bot.workers)} (Gathering: {len(gatherers)})")
    print(f"    Supply: {bot.supply_used}/{bot.supply_cap}")
    print(f"    Minerals: {bot.minerals}")
    print(f"    Vespene: {bot.vespene}")
    print(f"    Income: {bot.state.score.collection_rate_minerals}/min minerals, {bot.state.score.collection_rate_vespene}/min gas")


def _print_cheese_detection_status(bot) -> None:
    """Print cheese detection intel (Zerg only - early game)."""
    if not hasattr(bot, '_cheese_label') or bot.time > 240.0:
        return

    print("\n  RUSH DETECTION (vs Zerg):")

    score_12p = getattr(bot, '_score_12p', 0)
    score_speed = getattr(bot, '_score_speed', 0)
    rush_label = getattr(bot, '_cheese_label', 'none')
    is_rushed = getattr(bot, '_cheese_detected', False)
    auto_true = getattr(bot, '_auto_true_fired', False)
    rush_source = getattr(bot, '_cheese_source', None)

    ml_probs = getattr(bot, '_ml_probs', None)
    ml_confidence = getattr(bot, '_ml_confidence', None)

    source_str = f" [{rush_source}]" if rush_source else ""
    print(f"    Label: {rush_label}{source_str} (12p={score_12p}, speed={score_speed})")

    if ml_probs:
        prob_str = ", ".join(f"{k}={v*100:.0f}%" for k, v in sorted(ml_probs.items()))
        print(f"    ML Probs: {prob_str}")

    print(f"    Rush Detected: {is_rushed} (auto-TRUE={auto_true})")

    last_scout_time = getattr(bot, '_last_nat_scout_time', None)
    nat_present_last = getattr(bot, '_nat_present_on_last_scout', None)
    nat_started = getattr(bot, '_enemy_nat_started_at', None)
    nat_str = f"{nat_started:.1f}s" if nat_started else "absent"
    scout_str = f"last@{last_scout_time:.0f}s" if last_scout_time else "not scouted"
    present_str = "yes" if nat_present_last else ("no" if nat_present_last is False else "?")
    print(f"    Enemy Natural: {nat_str} ({scout_str}, present={present_str})")

    pool_state = getattr(bot, '_pool_seen_state', 'none')
    pool_time = getattr(bot, '_pool_seen_time', None)
    pool_time_str = f"{pool_time:.1f}s" if pool_time else "unknown"
    print(f"    Pool: {pool_state} (start≈{pool_time_str})")

    speed_started = getattr(bot, '_speed_research_started', False)
    speed_time = getattr(bot, '_speed_research_time', None)
    speed_str = f"{speed_time:.1f}s" if speed_time else "not seen"
    if speed_started:
        print(f"    Speed Research: started at {speed_str}")


def _get_rush_timing(attr: str, bot, default=-1):
    """Get rush detection timing value, returning default for missing."""
    val = getattr(bot, attr, None)
    return val if val is not None else default


def emit_match_record(bot, game_result, game_time: float,
                      idle_worker_time: float, idle_production_time: float) -> None:
    """Emit single match record telemetry event (called from on_end).
    
    Combines performance data, cheese classification, and rush detection
    timing features into one record per game.
    """
    pm = bot.performance_monitor
    cheese_type = _get_cheese_type(bot)

    result_str = str(game_result)
    if result_str == "Result.Victory":
        result = "win"
    elif result_str == "Result.Defeat":
        result = "loss"
    elif result_str == "Result.Tie":
        result = "tie"
    else:
        # Result.Undecided or unknown — likely a crash/disconnect
        result = "undecided"

    # Flush an in-progress attack so the accumulator reflects total time
    # even if the game ended mid-engagement (on_end may not pass through combat).
    _current_start = getattr(bot, '_current_attack_start', 0.0)
    if _current_start > 0:
        bot._total_attack_time = getattr(bot, '_total_attack_time', 0.0) + (game_time - _current_start)
        bot._current_attack_start = 0.0

    match_fields = {
        "result": result,
        "length": round(game_time, 1),
        "cheese_type": cheese_type,
        "commenced_attack": getattr(bot, '_commenced_attack', False),
        "attack_initiation_count": getattr(bot, '_attack_initiation_count', 0),
        "total_attack_time": round(getattr(bot, '_total_attack_time', 0.0), 1),
        "used_cheese_response": bot.reaction_manager.is_cheese_response,
        "sq": round(pm.get_current_sq(), 1),
        "mineral_sq": round(pm.get_mineral_sq(), 1),
        "gas_sq": round(pm.get_gas_sq(), 1),
        "efficiency_rating": pm.get_efficiency_rating(),
        "avg_unspent_minerals": round(pm.avg_unspent_minerals, 1),
        "avg_unspent_vespene": round(pm.avg_unspent_vespene, 1),
        "avg_income_minerals": round(pm.avg_income_minerals, 1),
        "avg_income_vespene": round(pm.avg_income_vespene, 1),
        "idle_worker_time": round(idle_worker_time, 1),
        "idle_production_time": round(idle_production_time, 1),
    }

    # Worker rush detection timestamp
    reaction_start = bot.reaction_manager.reaction_start_time
    if reaction_start >= 0 and bot.reaction_manager.active_reaction_name == "worker_rush":
        match_fields["worker_rush_detected_at"] = round(reaction_start, 1)

    # Timing features — sent for ALL races (not just Zerg).
    # enemy_timings.py tracks these for every matchup; the API needs them
    # for BN training across all races, not just Zerg cheese detection.
    match_fields.update({
        "pool_start": _get_rush_timing('_pool_seen_time', bot),
        "nat_start": _get_rush_timing('_enemy_nat_started_at', bot),
        "last_nat_scout_time": _get_rush_timing('_last_nat_scout_time', bot),
        "nat_present_on_last_scout": (1 if getattr(bot, '_nat_present_on_last_scout', None)
                                      else (0 if getattr(bot, '_nat_present_on_last_scout', None) is False
                                            else -1)),
        "gas_time": _get_rush_timing('_extractor_seen_time', bot),
        "ling_seen": _get_rush_timing('_first_ling_seen_time', bot),
        "ling_contact": _get_rush_timing('_first_ling_contact_nat_time', bot),
        "rush_distance_seconds": round(getattr(bot, '_rush_time_seconds', 0.0), 1),
        # Schema v2 BN features (Phase 5 Steps 1+2) — position + timing.
        # Position booleans: 1=yes, 0=no (seen but not near), -1=unknown (never seen).
        # These feed the API's match-level-full endpoint so train_strategy_belief.py
        # can use real position data instead of hardcoding "unknown".
        "rax_start": _get_rush_timing('_barracks_seen_time', bot),
        "gw_start": _get_rush_timing('_gateway_seen_time', bot),
        "rax_near_base": _bool_to_int(getattr(bot, '_barracks_near_our_base', False),
                                      seen=bool(getattr(bot, '_barracks_count', 0))),
        "gw_near_base": _bool_to_int(getattr(bot, '_gateway_near_our_base', False),
                                     seen=bool(getattr(bot, '_gateway_count', 0)
                                               + getattr(bot, '_warpgate_count', 0))),
        "cannon_near_base": _cannon_near_base_int(bot),
        "bunker_near_base": _bool_to_int(getattr(bot, '_bunker_near_base', False),
                                         seen=any(s.type_id == UnitTypeId.BUNKER
                                                  for s in bot.enemy_structures)),
    })

    # Zerg-specific cheese detection fields (only set for Zerg/Random)
    if hasattr(bot, '_cheese_label'):
        match_fields.update({
            "cheese_label": getattr(bot, '_cheese_label', 'none'),
            "queen_time": _get_rush_timing('_queen_started_time', bot),
            "speed_start": _get_rush_timing('_speed_research_time', bot),
            "ling_has_speed": 1 if getattr(bot, '_ling_has_speed', False) else 0,
            "gas_workers": getattr(bot, '_gas_workers_count', 0),
            "score_12p": getattr(bot, '_score_12p', 0),
            "score_speed": getattr(bot, '_score_speed', 0),
            "auto_true_fired": getattr(bot, '_auto_true_fired', False),
        })

    # Strategy belief fields (when enabled)
    if (bot.config.get("Belief", {}).get("enable_strategy", True)
            and bot.belief_state.strategy is not None):
        pred = bot.belief_state.strategy.last_prediction
        if pred is not None:
            match_fields.update({
                "strategy_label": pred.label.value,
                "strategy_level2": pred.level2,
                "strategy_source": pred.source,
                "strategy_p_cheese": round(pred.p_cheese, 3),
                "strategy_p_all_in": round(pred.p_all_in, 3),
                "strategy_p_timing": round(pred.p_timing, 3),
                "strategy_p_macro": round(pred.p_macro, 3),
            })
            # Opponent prior applied (Phase 4): record which prior shifted the prediction
            if pred.source == "BN+OPP":
                match_fields["opponent_prior_applied"] = True

    # Scout VOI staleness snapshot (when enabled)
    if bot.config.get("Belief", {}).get("enable_scout_voi", False):
        location_last_seen = getattr(bot, "_location_last_seen", {})
        if location_last_seen:
            match_fields["scout_voi_enabled"] = True
            for key in sorted(location_last_seen.keys()):
                match_fields[f"last_seen_{key}_final"] = round(location_last_seen[key], 1)

    log_match(**match_fields)


def print_end_game_report(
    performance_monitor,
    game_result,
    game_time: float,
    idle_worker_time: float,
    idle_production_time: float
) -> None:
    """
    Print streamlined end-game report using PerformanceMonitor data.

    Args:
        performance_monitor: PerformanceMonitor instance with collected data
        game_result: Result enum (Victory/Defeat/Tie)
        game_time: Game time in seconds
        idle_worker_time: Total idle worker time in seconds
        idle_production_time: Total idle production time in seconds
    """
    sq = performance_monitor.get_current_sq()
    rating = performance_monitor.get_efficiency_rating()

    combined_income = performance_monitor.avg_income_minerals + performance_monitor.avg_income_vespene
    combined_unspent = performance_monitor.avg_unspent_minerals + performance_monitor.avg_unspent_vespene

    print("\n" + "="*50)
    print(f"  RESULT: {game_result}")
    print(f"  Game Time: {game_time / 60:.1f} min")
    print("="*50)
    mineral_sq = performance_monitor.get_mineral_sq()
    gas_sq = performance_monitor.get_gas_sq()
    print(f"  SPENDING QUOTIENT:  {sq:.1f}  [{rating}]")
    print(f"    Mineral SQ: {mineral_sq:.1f}  |  Gas SQ: {gas_sq:.1f}")
    print("-"*50)
    print("  RESOURCES:")
    print(f"    Combined Income:  {combined_income:.0f}/min")
    print(f"    Combined Unspent: {combined_unspent:.0f}")
    print(f"      └─ Minerals: {performance_monitor.avg_unspent_minerals:.0f}m")
    print(f"      └─ Gas:      {performance_monitor.avg_unspent_vespene:.0f}g")
    print("-"*50)
    print("  EFFICIENCY:")
    print(f"    Idle Workers:     {idle_worker_time:.1f}s")
    print(f"    Idle Production:  {idle_production_time:.1f}s")
    print("="*50 + "\n")