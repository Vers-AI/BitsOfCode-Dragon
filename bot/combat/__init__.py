"""Combat management package for StarCraft II bot.

Handles all combat-related functionality including unit control, targeting, and engagement logic.
"""

from bot.constants import (
    ATTACKING_SQUAD_RADIUS,
    DEFENDER_SQUAD_RADIUS,
    COMMON_UNIT_IGNORE_TYPES,
    DISRUPTOR_IGNORE_TYPES,
    MELEE_RANGE_THRESHOLD,
    STAY_AGGRESSIVE_DURATION,
    TARGET_LOCK_DISTANCE,
)

from bot.combat.combat import (
    control_main_army,
    control_defenders,
    gatekeeper_control,
    warp_prism_follower,
    handle_attack_toggles,
    attack_target,
    fallback_target,
    control_base_defenders,
    manage_defensive_unit_roles,
    try_mass_recall,
)

from bot.combat.unit_micro import (
    micro_ranged_unit,
    micro_melee_unit,
    micro_disruptor,
    micro_high_templar,
    merge_high_templars,
    micro_sentry,
    micro_stalker,
    micro_air_unit,
)

from bot.combat.target_scoring import (
    score_target,
    select_target,
)

from bot.combat.formation import (
    execute_fan_out,
    clear_formation_state,
)

from bot.combat.force_field import (
    compute_ff_split,
    compute_ff_main_ramp_block,
    compute_ff_choke_block,
    FFResult,
)

from bot.combat.group_snipe import (
    effective_value,
    damage_per_volley,
    stalkers_needed_to_kill,
    find_snipe_candidates,
    find_best_snipe_target,
    try_commit_snipe,
    execute_snipe_a,
    execute_snipe_b,
    execute_focus,
)

from bot.combat.group_chase import (
    try_commit_chase,
    execute_chase,
)

__all__ = [
    # Constants
    "ATTACKING_SQUAD_RADIUS",
    "DEFENDER_SQUAD_RADIUS",
    "COMMON_UNIT_IGNORE_TYPES",
    "DISRUPTOR_IGNORE_TYPES",
    "MELEE_RANGE_THRESHOLD",
    "STAY_AGGRESSIVE_DURATION",
    "TARGET_LOCK_DISTANCE",
    # Combat control functions
    "control_main_army",
    "gatekeeper_control",
    "warp_prism_follower",
    "handle_attack_toggles",
    "attack_target",
    "fallback_target",
    "control_defenders",
    "control_base_defenders",
    "manage_defensive_unit_roles",
    "try_mass_recall",
    # Unit micro functions
    "micro_ranged_unit",
    "micro_melee_unit",
    "micro_disruptor",
    "micro_high_templar",
    "merge_high_templars",
    "micro_sentry",
    "micro_stalker",
    "micro_air_unit",
    # Target scoring
    "score_target",
    "select_target",
    # Formation
    "execute_fan_out",
    "clear_formation_state",
    # Force Field
    "compute_ff_split",
    "compute_ff_main_ramp_block",
    "compute_ff_choke_block",
    "FFResult",
    # Group snipe
    "effective_value",
    "damage_per_volley",
    "stalkers_needed_to_kill",
    "find_snipe_candidates",
    "find_best_snipe_target",
    "try_commit_snipe",
    "execute_snipe_a",
    "execute_snipe_b",
    "execute_focus",
    # Group chase
    "try_commit_chase",
    "execute_chase",
]
