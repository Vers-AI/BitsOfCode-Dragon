"""System-wide constants and configuration values.

Purpose: Centralize all magic numbers, unit filters, and tunable parameters across the entire bot
Key Decisions: Immutable constants to prevent accidental modification, organized by subsystem
Limitations: None - pure data definitions

Organization:
- Build profiles (army comp, upgrades, economy targets per build)
- Combat constants (squad radii, detection ranges, timings)
- Unit filtering sets (ignore lists, priority targets)
"""

from dataclasses import dataclass, field
from typing import Callable, Union

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from cython_extensions import cy_structure_pending_ares, cy_unit_pending

# Lazy import to avoid circular dependency — get_economy_state is only called inside lambdas
# that execute at runtime, so the import resolves correctly by then.
def _get_economy_state(bot):
    """Runtime proxy for get_economy_state to avoid circular imports at module load time."""
    from bot.utilities.performance_monitor import get_economy_state
    return get_economy_state(bot)

# ===== SQUAD CONFIGURATION =====
ATTACKING_SQUAD_RADIUS = 9.0
"""Squad radius for ATTACKING role units - looser formation for army mobility"""

DEFENDER_SQUAD_RADIUS = 6.0
"""Squad radius for BASE_DEFENDER role units - tighter formation for base defense"""

# ===== UNIT FILTERING =====
COMMON_UNIT_IGNORE_TYPES: set[UnitTypeId] = {
    UnitTypeId.EGG,
    UnitTypeId.LARVA,
    UnitTypeId.CREEPTUMORBURROWED,
    UnitTypeId.CREEPTUMORQUEEN,
    UnitTypeId.CREEPTUMOR,
    UnitTypeId.MULE,
    UnitTypeId.OVERLORD,
    UnitTypeId.OVERSEER,
    UnitTypeId.OVERSEERSIEGEMODE,
    UnitTypeId.OBSERVER,
    UnitTypeId.OBSERVERSIEGEMODE,
    UnitTypeId.LOCUSTMP,
    UnitTypeId.LOCUSTMPFLYING,
    UnitTypeId.ADEPTPHASESHIFT,
    UnitTypeId.CHANGELING,
    UnitTypeId.CHANGELINGMARINE,
    UnitTypeId.CHANGELINGMARINESHIELD,
    UnitTypeId.CHANGELINGZEALOT,
    UnitTypeId.CHANGELINGZERGLING,
    UnitTypeId.CHANGELINGZERGLINGWINGS,
}
"""Units to ignore in combat targeting and threat calculations - non-threatening scouts, supply, temporary units"""

DISRUPTOR_IGNORE_TYPES: set[UnitTypeId] = COMMON_UNIT_IGNORE_TYPES | {
    UnitTypeId.SCV,
    UnitTypeId.DRONE,
    UnitTypeId.PROBE,
    UnitTypeId.BROODLING,
}
"""Units to ignore for Disruptor nova targeting - includes workers (too low value)"""

# ===== COMBAT PARAMETERS =====
MELEE_RANGE_THRESHOLD = 3.0
"""Range threshold to classify units as melee vs ranged"""

MELEE_THREAT_BUFFER = 1.0
"""Extra buffer beyond melee attack range for threat detection (tiles)"""

STAY_AGGRESSIVE_DURATION = 20.0
"""Minimum time (seconds) to stay committed to an attack before reconsidering"""

TARGET_LOCK_DISTANCE = 25.0
"""Distance threshold for switching attack targets - prevents oscillation"""

# ===== FORMATION & COHESION =====
# Dynamic thresholds: base + sqrt(unit_count) * scale
# Small army (5) → ahead ~1.4, spread ~4.6  |  Large army (30) → ahead ~2.1, spread ~6.8
FORMATION_AHEAD_BASE = 1.0
"""Minimum ahead threshold before unit is considered 'streaming ahead'"""

FORMATION_AHEAD_SCALE = 0.2
"""How much ahead threshold grows with sqrt(n)"""

FORMATION_SPREAD_BASE = 3.0
"""Minimum spread threshold before unit is considered 'too spread out'"""

FORMATION_SPREAD_SCALE = 0.7
"""How much spread threshold grows with sqrt(n)"""

FORMATION_UNIT_MULTIPLIER = 1.2
"""How far ranged units reposition behind melee"""

FORMATION_RETREAT_ANGLE = 0.3
"""Diagonal spread for ranged units"""

# ===== ENGAGEMENT GATE =====
ENGAGEMENT_ARMY_VALUE_THRESHOLD = 4.0
"""Minimum total enemy army_value before switching from formation to full micro. Prevents lone scouts from fragmenting the army."""

ACTIVE_ENGAGE_RANGE_BUFFER = 0.5
"""Extra buffer beyond enemy weapon range + radii for active engagement fallback"""

ACTIVE_ENGAGE_ANGLE = 0.3
"""~17° tolerance for is_facing check in active engagement fallback"""

# ===== COMBAT SIMULATOR THRESHOLDS =====
SQUAD_NEARBY_FRIENDLY_RANGE_SQ = 255.0
"""Squared distance (16^2) to find nearby friendly units for squad-level combat simulation"""

# ===== DISTANCE THRESHOLDS =====
UNSAFE_GROUND_CHECK_RADIUS = 8.0
"""Radius to search for safe spots when unit is on unsafe ground"""

DISRUPTOR_SQUAD_FOLLOW_DISTANCE = 5.0
"""Maximum distance disruptor should be from squad center"""

DISRUPTOR_SQUAD_TARGET_DISTANCE = 4.0
"""Target distance for disruptor to maintain from squad"""

DISRUPTOR_FORWARD_OFFSET = 8.0
"""Distance Disruptor positions ahead of army toward target"""

HT_SQUAD_FOLLOW_DISTANCE = 5.0
"""Maximum distance HT should be from squad center (mirrors disruptor)"""

HT_SQUAD_TARGET_DISTANCE = 4.0
"""Target distance for HT to maintain from squad"""

HT_STORM_ENERGY_COST = 75
"""Energy cost of Psi Storm ability"""

HT_STORM_MIN_TARGETS = 4
"""Minimum enemies clumped to justify casting Psi Storm"""

HT_FEEDBACK_ENERGY_COST = 50
"""Energy cost of Feedback ability"""

HT_MERGE_ENERGY_THRESHOLD = 50
"""Only merge HTs to archon when both have less than this energy"""

HT_MERGE_COUNT_THRESHOLD = 3
"""Minimum HT count before considering merges (non-PvP).

When we have this many or more HTs, low-energy HTs become eligible
for merging into Archons. Below this count, HTs are preserved even
if low on energy so the SpawnController doesn't over-produce replacements.

This works with the army composition system: the SpawnController only
builds HTs when current proportion < target proportion. By only merging
when we're well-stocked (>= this threshold), the remaining HTs stay at
or above the target proportion after merging, preventing an infinite
build-merge-build loop.
"""

HT_MERGE_COUNT_THRESHOLD_PVP = 1
"""Minimum HT count before considering merges in PvP.

Lowest meaningful value: as soon as 2 HTs exist (len > 1), trigger 2
fires and merges them. In PvP, HTs are primarily Archon-in-waiting (IAC
composition), so we want immediate conversion. The Archon-percentage
switch to PVP_ARMY_1 (at 30% Archons) handles the loop brake.
"""

HT_FEEDBACK_RANGE = 10.0
"""Cast range of Feedback ability"""

HT_FEEDBACK_MIN_ENEMY_ENERGY = 50
"""Minimum enemy energy to be worth Feedbacking (damage = enemy energy)"""

FEEDBACK_TARGET_TYPES: set[UnitTypeId] = {
    UnitTypeId.GHOST,
    UnitTypeId.VIPER,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.INFESTOR,
    UnitTypeId.RAVEN,
}
"""Unit types worth casting Feedback on — high-value casters only"""

SENTRY_SQUAD_FOLLOW_DISTANCE = 5.0
"""Maximum distance sentry should be from squad center"""

SENTRY_SQUAD_TARGET_DISTANCE = 4.0
"""Target distance for sentry to maintain from squad"""

GUARDIAN_SHIELD_ENERGY_COST = 75
"""Energy cost of Guardian Shield"""

GUARDIAN_SHIELD_RADIUS = 4.5
"""Radius of Guardian Shield effect"""

GUARDIAN_SHIELD_OVERLAP_DISTANCE = 8.0
"""If another shielded sentry is within this range, skip casting (avoid overlap).
Roughly 2x radius — so shields cover different areas instead of stacking."""

HALLUCINATION_ENERGY_COST = 75
"""Energy cost of Hallucination ability"""

HALLUCINATION_SCOUT_MIN_OBSERVERS = 3
"""Only hallucinate a scout if we have fewer than this many observers.
If we have 3+ observers, they provide enough scouting coverage."""

HALLUCINATION_SCOUT_COOLDOWN = 30.0
"""Minimum seconds between hallucination scout casts (prevents energy waste)"""

STRUCTURE_ATTACK_RANGE = 12.0
"""Range to detect nearby enemy structures"""

PROXIMITY_STICKY_DISTANCE_SQ = 450.0
"""Squared distance (21.2^2) for proximity stickiness to structures"""

MAP_CROSSING_DISTANCE_SQ = 49.0
"""Squared distance (7^2) threshold for safe map crossing with PathUnitToTarget"""

MAP_CROSSING_SUCCESS_DISTANCE = 6.5
"""Success distance for PathUnitToTarget during map crossing"""

# ===== ENGAGEMENT RANGES =====
UNIT_ENEMY_DETECTION_RANGE = 15.0
"""Range to detect enemies around each unit's position (per-unit detection)"""

GATEKEEPER_DETECTION_RANGE = 6.0
"""Range to detect enemies around gatekeeping position"""

GATEKEEPER_MOVE_DISTANCE = 3.0
"""Distance gatekeeper moves when no enemies present"""

# ===== WARP PRISM PARAMETERS =====
WARP_PRISM_FOLLOW_DISTANCE = 15.0
"""Distance threshold for warp prism to morph to phasing mode"""

WARP_PRISM_FOLLOW_OFFSET = 3.0
"""Distance offset behind army center for warp prism positioning"""

WARP_PRISM_UNIT_CHECK_RANGE = 6.5
"""Range to check for units warping in near prism"""

WARP_PRISM_DANGER_DISTANCE = 10.0
"""Danger distance parameter for warp prism pathfinding"""

WARP_PRISM_SAFETY_LIMIT = 1.5
"""Max grid weight to consider position safe for warp-ins (1.0 = no enemy influence)"""

WARP_PRISM_MIN_ENEMY_DISTANCE = 12.0
"""Minimum distance from enemy army center to enter phase mode"""

WARP_PRISM_MATRIX_RADIUS = 3.75
"""Psionic matrix radius for warp prism phasing mode"""

WARP_PRISM_POSITION_SEARCH_RANGE = 8.0
"""Range to search for valid warp-in positions around current position"""

WARP_PRISM_POSITION_SEARCH_STEP = 2.0
"""Step size when searching for valid warp-in positions"""

# ===== MASS RECALL (NEXUS EMERGENCY RECALL) =====
MASS_RECALL_RADIUS = 6.5
"""Radius of Nexus Mass Recall effect — units within this radius are teleported to the Nexus"""

MASS_RECALL_COOLDOWN = 130.0
"""Global cooldown (seconds) for Nexus Mass Recall across all Nexuses"""

MASS_RECALL_ENERGY_COST = 50
"""Energy cost for Nexus Mass Recall ability"""

MASS_RECALL_MIN_OWN_SUPPLY = 10
"""Minimum own army supply to consider mass recall (don't recall trivial forces)"""

MASS_RECALL_RETREAT_SEARCH_RADIUS = 15.0
"""Radius to search for safe retreat spot on ARES avoidance grid.
If no safe spot closer to base exists within this radius, retreat is considered blocked."""

# ===== UNDER ATTACK DETECTION =====
# OBSERVATION: These thresholds determine when _under_attack flag triggers
# Adjust based on gameplay feedback
UNDER_ATTACK_VALUE_THRESHOLD = 15.0
"""Minimum threat value to trigger under_attack (roughly 3 stalkers worth)"""

UNDER_ATTACK_RATIO_THRESHOLD = 0.4
"""Minimum ratio of enemy army near bases to trigger under_attack (40% of known army)"""

UNDER_ATTACK_CLEAR_VALUE = 5.0
"""Threat value below which under_attack clears (hysteresis to prevent flickering)"""

# ===== EARLY GAME DEFENSE =====
EARLY_GAME_TIME_LIMIT = 600.0
"""Time limit (seconds) for early game defensive positioning (10 minutes)"""

EARLY_GAME_SAFE_GROUND_CHECK_BASES = 1
"""Maximum number of bases to still check for safe ground positioning"""

# ===== RUSH DETECTION =====
RUSH_SPEED = 3.94
"""Rush movement speed (used for rush distance calculations)"""

RUSH_DISTANCE_CALIBRATION = 0.833
"""Calibration factor to match official map rush distances (36s official / 43.2s calculated = 0.833)"""

# ===== INTEL QUALITY & FRESHNESS =====
MEMORY_EXPIRY_TIME = 30.0
"""Time (seconds) after which ARES UnitMemoryManager expires ghost units - used to filter stale cache entries"""

VISIBLE_AGE_THRESHOLD = 3.0
"""Age threshold (seconds) to consider a unit 'visible' vs 'memory' (forgiving for active combat)"""

STALENESS_WINDOW = 40.0
"""Time window (seconds) for intel freshness decay - extended to reduce staleness rate"""

FRESH_INTEL_THRESHOLD = 0.7
"""Freshness score above which intel is considered 'fresh' - use normal combat sim thresholds"""

STALE_INTEL_THRESHOLD = 0.05
"""Freshness score below which intel is 'very stale' - don't initiate attacks, need scouting.

At 0.2, a single observer glimpse (~0.3-0.4 freshness) decayed below threshold within ~10s,
locking out attacks entirely for most of the game. 0.05 reserves the hard block for genuinely
blind scenarios (no intel for 38+ seconds)."""

URGENCY_BUILD_RATE = 0.02
"""Rate per frame that intel urgency builds when stale (~0.4/sec at 22fps)"""

URGENCY_DECAY_RATE = 0.005
"""Rate per frame that intel urgency decays when fresh (slower than build)"""

# ===== BELIEF LAYER =====
# Per-unit-type half-lives (seconds) for composition belief decay.
# P(unit still exists | age, type) = 0.5^(age / half_life).
# Workers survive longer (mining, not in fights); fragile units die fast; structures persist.
UNIT_HALF_LIFE: dict[UnitTypeId, float] = {
    # Workers — long half-life, they stay alive mining unless harassed
    UnitTypeId.SCV: 60.0,
    UnitTypeId.DRONE: 60.0,
    UnitTypeId.PROBE: 60.0,
    UnitTypeId.MULE: 45.0,
    # Terran combat
    UnitTypeId.MARINE: 20.0,
    UnitTypeId.MARAUDER: 22.0,
    UnitTypeId.REAPER: 15.0,
    UnitTypeId.GHOST: 20.0,
    UnitTypeId.HELLION: 18.0,
    UnitTypeId.HELLIONTANK: 18.0,
    UnitTypeId.CYCLONE: 20.0,
    UnitTypeId.SIEGETANK: 25.0,
    UnitTypeId.SIEGETANKSIEGED: 25.0,
    UnitTypeId.THOR: 30.0,
    UnitTypeId.VIKINGFIGHTER: 18.0,
    UnitTypeId.VIKINGASSAULT: 18.0,
    UnitTypeId.MEDIVAC: 30.0,
    UnitTypeId.RAVEN: 25.0,
    UnitTypeId.BANSHEE: 15.0,
    UnitTypeId.BATTLECRUISER: 35.0,
    UnitTypeId.LIBERATOR: 18.0,
    UnitTypeId.LIBERATORAG: 18.0,
    # Protoss combat
    UnitTypeId.ZEALOT: 20.0,
    UnitTypeId.STALKER: 20.0,
    UnitTypeId.SENTRY: 22.0,
    UnitTypeId.ADEPT: 20.0,
    UnitTypeId.HIGHTEMPLAR: 15.0,
    UnitTypeId.DARKTEMPLAR: 15.0,
    UnitTypeId.ARCHON: 25.0,
    UnitTypeId.IMMORTAL: 25.0,
    UnitTypeId.COLOSSUS: 22.0,
    UnitTypeId.WARPPRISM: 30.0,
    UnitTypeId.PHOENIX: 15.0,
    UnitTypeId.VOIDRAY: 20.0,
    UnitTypeId.ORACLE: 15.0,
    UnitTypeId.CARRIER: 30.0,
    UnitTypeId.TEMPEST: 30.0,
    UnitTypeId.MOTHERSHIP: 35.0,
    UnitTypeId.DISRUPTOR: 15.0,
    # Zerg combat
    UnitTypeId.ZERGLING: 15.0,
    UnitTypeId.ZERGLINGBURROWED: 15.0,
    UnitTypeId.ROACH: 20.0,
    UnitTypeId.ROACHBURROWED: 20.0,
    UnitTypeId.RAVAGER: 20.0,
    UnitTypeId.HYDRALISK: 20.0,
    UnitTypeId.HYDRALISKBURROWED: 20.0,
    UnitTypeId.MUTALISK: 15.0,
    UnitTypeId.CORRUPTOR: 22.0,
    UnitTypeId.BROODLORD: 25.0,
    UnitTypeId.INFESTOR: 20.0,
    UnitTypeId.INFESTORBURROWED: 20.0,
    UnitTypeId.SWARMHOSTBURROWEDMP: 22.0,
    UnitTypeId.SWARMHOSTMP: 22.0,
    UnitTypeId.VIPER: 20.0,
    UnitTypeId.ULTRALISK: 25.0,
    UnitTypeId.QUEEN: 22.0,
    UnitTypeId.BANELING: 12.0,
    UnitTypeId.BANELINGBURROWED: 12.0,
    # Key structures (visible as units in combat)
    UnitTypeId.BUNKER: 90.0,
    UnitTypeId.MISSILETURRET: 60.0,
    UnitTypeId.PHOTONCANNON: 60.0,
    UnitTypeId.SPINECRAWLER: 90.0,
    UnitTypeId.SPORECRAWLER: 60.0,
}

DEFAULT_HALF_LIFE = 20.0
"""Default half-life for unit types not in UNIT_HALF_LIFE — standard combat unit assumption"""

STRUCTURE_SEEN_UNIT_PRIOR: dict[UnitTypeId, dict[UnitTypeId, float]] = {
    # Terran structures → likely units
    UnitTypeId.BARRACKS: {UnitTypeId.MARINE: 0.6, UnitTypeId.MARAUDER: 0.3, UnitTypeId.REAPER: 0.1},
    UnitTypeId.FACTORY: {UnitTypeId.SIEGETANK: 0.4, UnitTypeId.HELLION: 0.3, UnitTypeId.CYCLONE: 0.2, UnitTypeId.THOR: 0.1},
    UnitTypeId.STARPORT: {UnitTypeId.MEDIVAC: 0.3, UnitTypeId.VIKINGFIGHTER: 0.3, UnitTypeId.LIBERATOR: 0.2, UnitTypeId.BANSHEE: 0.1, UnitTypeId.RAVEN: 0.1},
    # Protoss structures → likely units
    UnitTypeId.GATEWAY: {UnitTypeId.ZEALOT: 0.5, UnitTypeId.STALKER: 0.3, UnitTypeId.SENTRY: 0.1, UnitTypeId.ADEPT: 0.1},
    UnitTypeId.ROBOTICSFACILITY: {UnitTypeId.IMMORTAL: 0.4, UnitTypeId.WARPPRISM: 0.2, UnitTypeId.COLOSSUS: 0.2, UnitTypeId.DISRUPTOR: 0.2},
    UnitTypeId.STARGATE: {UnitTypeId.VOIDRAY: 0.3, UnitTypeId.PHOENIX: 0.2, UnitTypeId.ORACLE: 0.2, UnitTypeId.CARRIER: 0.15, UnitTypeId.TEMPEST: 0.15},
    # Zerg structures → likely units
    UnitTypeId.HATCHERY: {UnitTypeId.QUEEN: 0.5, UnitTypeId.ZERGLING: 0.3, UnitTypeId.DRONE: 0.2},
    UnitTypeId.SPAWNINGPOOL: {UnitTypeId.ZERGLING: 0.6, UnitTypeId.QUEEN: 0.3, UnitTypeId.ROACH: 0.1},
    UnitTypeId.ROACHWARREN: {UnitTypeId.ROACH: 0.7, UnitTypeId.RAVAGER: 0.3},
    UnitTypeId.HYDRALISKDEN: {UnitTypeId.HYDRALISK: 0.7, UnitTypeId.LURKERMP: 0.3},
    UnitTypeId.SPIRE: {UnitTypeId.MUTALISK: 0.5, UnitTypeId.CORRUPTOR: 0.3, UnitTypeId.BROODLORD: 0.2},
}
"""Structure-based priors: P(unit_type produced | structure_type seen)"""

# ===== CHOKE/RAMP DETECTION =====
RAMP_CHOKE_RADIUS = 2.5
"""Radius around ramp top/bottom for choke grid marking (actual ramp ~2-3 tiles wide)"""

MAP_CHOKE_RADIUS = 3.5
"""Radius around map_analyzer choke points for grid marking (actual choke ~3-5 tiles wide)"""

CHOKE_GRID_WEIGHT = 10.0
"""Weight value to mark choke zones in grid (>1.0 indicates choke area)"""

CHOKE_SAMPLE_POINTS = 8
"""Number of points to sample along the line between squad and enemy to detect chokes"""

CHOKE_MELEE_DPS_THRESHOLD = 0.35
"""Minimum enemy_melee_dps / total_enemy_dps ratio for choke to be considered favorable.
Below this, the enemy is mostly ranged and choke doesn't help enough to suppress engagement."""

CHOKE_MELEE_RANGE = 3.0
"""Ground range threshold to classify enemy units as melee for choke DPS calculation.
Matches MELEE_RANGE_THRESHOLD but kept separate for choke-specific tuning."""

CHOKE_MAX_WIDTH = 10.0
"""Maximum choke width (tiles) to include in the choke-width map. Wider chokes don't limit
engagement surface area enough to matter. SC2 ramps are ~3-5 tiles, natural walls ~6-8.
Note: This is a pre-filter only. At engagement time, the choke width is compared against
the effective army widths — a 10-tile choke won't suppress engagement if both armies fit."""

CHOKE_MIN_ARMY_WIDTH = 2.0
"""Minimum effective army width returned by effective_army_width(). Prevents tiny squads
(1-2 units) from trivially passing through any choke and triggering the policy."""

CHOKE_RETREAT_DIST = 7.0
"""Distance (tiles) behind the choke tile to set the retreat target.
Anchored to the specific choke tile (not squad position) so the bot retreats
to a stable fixed point rather than continuing through multiple choke points.
Ramp retreat uses 4.0 per-unit; group retreat uses a smaller distance since
the anchor is the choke edge, not the current squad position."""

# ===== CONCAVE FORMATION =====
CONCAVE_TRIGGER_RANGE = 25.0
"""Distance to enemy center at which squads begin fan-out spread"""

CONCAVE_MIN_RANGED_UNITS = 4
"""Minimum ground ranged units in a squad to bother with concave formation"""

CONCAVE_FAN_WIDTH_PER_UNIT = 1.0
"""Lateral spread per ranged unit (total fan width = n * this, capped)"""

CONCAVE_MAX_FAN_WIDTH = 15.0
"""Maximum total lateral spread width regardless of unit count"""

CONCAVE_SPREAD_FRAMES = 70
"""Max frames (~3s at 22fps) to spend spreading before forcing engagement"""

CONCAVE_WEAPON_RANGE_ABORT = 8.0
"""If enemy closer than this, skip/abort formation and fight immediately"""

CONCAVE_RESET_RANGE = 35.0
"""Distance to enemy above which formation state resets (ready for next engagement)"""

# ===== STALKER BLINK =====
STALKER_BLINK_HEALTH_THRESHOLD = 0.4
"""Health ratio (health+shields / max) below which Stalker considers blinking back.
At 0.4 a Stalker (160hp+80sh=240 total) triggers at ~96 combined HP.
Low enough to be in real danger, high enough to still have time to blink."""

STALKER_BLINK_RANGE = 8.0
"""Blink ability range in tiles (standard SC2 value)."""

STALKER_LOCKON_BREAK_DISTANCE = 15.0
"""Cyclone Lock-on max tether range. Blinking beyond this distance breaks the lock."""

# ===== DETECTION CANNON SYSTEM =====
DETECTION_CANNON_RANGE = 10.0
"""Range from nexus to search for pylons/cannons near mineral line.
Covers the full mineral line area behind the nexus."""

PYLON_POWER_RANGE = 6.5
"""Pylon power radius. Cannon must be within this range of a powered Pylon."""

MINERAL_CLEARANCE = 1.5
"""Minimum distance from mineral field center for detection cannon placements.
Prevents structures from blocking probe pathing between nexus and mineral line."""

STALKER_WIDOWMINE_DODGE_RADIUS = 10.0
"""Max distance from a WIDOWMINEBURROWED to consider dodge blink.
Widow Mine Sentinel Missile range is ~5 tiles; 10 gives margin for
detecting the mine's can_be_attacked transition and facing direction."""

STALKER_FUNGAL_DODGE_RADIUS = 14.0
"""Max distance from an INFESTOR to consider fungal dodge blink.
Fungal Growth has 9 cast range + 2.25 radius = 11.25 max reach;
plus stalkers on far side of impact zone = 13.5 tiles.
14 gives margin for detection and blink reaction time."""

FUNGAL_GROWTH_ENERGY_COST = 75
"""Energy cost of Fungal Growth ability — used to detect casts via energy drop."""

FUNGAL_GROWTH_IMPACT_RADIUS = 2.25
"""Radius of Fungal Growth impact zone (2.25 tiles, per LotV balance patch)."""

# ===== FORCE FIELD SPLIT =====
FF_ENERGY_COST = 50
"""Energy cost per Force Field cast"""

FF_RADIUS = 1.7
"""Radius of a single Force Field (tiles). Diameter = 3.0 tiles."""

FF_CAST_RANGE = 9.0
"""Sentry cast range for Force Field"""

FF_OVERLAP = 0.5
"""Overlap between adjacent Force Fields to prevent gaps.
0.5 tiles ensures no unit can slip through the chain."""

FF_SPLIT_MIN_ENEMIES = 8
"""Minimum total ground combat enemies to attempt a force field split"""

FF_RAMP_BLOCK_RADIUS = 5.0
"""Max distance from enemy center to ramp top/bottom center to trigger a ramp block.
A single FF at the ramp center when the enemy is crossing through it."""

FF_RAMP_BLOCK_MIN_VALUE = 6.0
"""Minimum enemy army_value near a ramp to justify a ramp block FF.
Roughly 2 stalkers or 6 zerglings worth — below this, 50 energy isn't worth spending.
Uses the same UNIT_DATA army_value as ENGAGEMENT_ARMY_VALUE_THRESHOLD."""

# ===== BLINK SNIPE / CHASE =====
SNIPE_MIN_HEALTH = 0.75
"""Minimum shield_health_percentage for a stalker to participate in a snipe.
Don't send already-hurt stalkers into a dive."""

SNIPE_MIN_TARGET_VALUE = 14.0
"""Minimum effective value (army_value * TYPE_VALUE_SCALE + TACTICAL_BONUS) for a target
to qualify for blink sniping. Roughly: High Templar ~29, Siege Tank sieged ~21, Medivac ~14."""

SNIPE_ISOLATION_MIN = 12.0
"""Minimum isolation score (effective_value - tactical_grid_excess * 0.3) for a snipe.
High value + low tactical = isolated → snipe. High value + high tactical = skip."""

SNIPE_OVERKILL_BUFFER_MOBILE = 1
"""Extra stalkers beyond kill math for mobile targets (can dodge/move between shots)."""

SNIPE_OVERKILL_BUFFER_STATIC = 0
"""Extra stalkers beyond kill math for static/immobile targets (sieged, burrowed)."""

SNIPE_EXIT_FRAMES = 30
"""Frames to hold VOLLEY state before retreating (~1.3s at 22.4fps, one stalker weapon cycle)."""

SNIPE_COMMIT_COOLDOWN = 180
"""Minimum frames between snipe commits per squad (~8s). Prevents re-committing
before previous snipe group's blink is off cooldown."""

CHASE_MIN_VALUE = 10.0
"""Minimum target value to initiate a chase. Lower bar than snipe since we're already winning."""

CHASE_TACTICAL_MAX = 210
"""Maximum tactical_grid value at target position for chase (200=neutral, >200=enemy-heavy).
210 means target is only lightly supported."""

CHASE_TIMEOUT_FRAMES = 180
"""Maximum chase duration in frames (~8s). One full blink cooldown cycle."""

CHASE_BLINK_GAP_THRESHOLD = 3.0
"""Distance beyond stalker range at which chase/focus Stalkers blink to close the gap.
If target is further than stalker_range + threshold, blink toward target."""

TACTICAL_ESCAPE_MAX = 260.0
"""Maximum tactical_grid value along escape corridor for Snipe-B (blink-in, walk-out).
Above this, the escape lane is too enemy-dominated to walk through."""

RETREAT_DETECTION_FRAMES = 5
"""Consecutive frames of increasing enemy-squad distance to confirm retreat."""

SNIPE_APPROACH_RANGE_BUFFER = 1.5
"""Buffer added to stalker range when deciding Snipe-A (walk-in) eligibility.
If target.ground_range <= stalker_range + buffer, stalker can walk into range safely."""

FOCUS_MIN_STALKERS = 2
"""Minimum Stalkers required to commit a focus-fire. Below this, not worth the commitment."""

# ===== RESOURCE-AWARE PRODUCTION =====
RESOURCE_PRESSURE_MAX_NUDGE = 0.15
"""Maximum proportion shift per unit type from resource pressure (±15%). Larger than counter-table nudge since resource starvation is more urgent."""

RESOURCE_IMBALANCE_RATIO = 2.0
"""Mineral:gas (or gas:mineral) ratio threshold to trigger resource-pressure nudging."""

FREEFLOW_INCOME_RATIO_THRESHOLD = 3.0
"""Income ratio (minerals/gas or gas/minerals) threshold for income-aware freeflow."""

FREEFLOW_BANK_THRESHOLD = 800
"""Resource bank threshold for triggering freeflow when spending is inefficient."""


# ===== BUILD PROFILES =====
# Each build gets a BuildProfile that bundles all build-specific macro settings.
# Adding a new build = add a BuildProfile instance + one dict entry. No macro.py changes needed.

@dataclass
class BuildProfile:
    """Bundles all build-specific macro settings for plug-and-play build selection.

    Fields accept int for static values or Callable[[bot], int] for dynamic ones.
    The _resolve() helper handles both transparently.
    """
    army_composition_0: dict[UnitTypeId, dict]
    army_composition_1: dict[UnitTypeId, dict]
    archon_switch_threshold: float
    upgrade_order: list[UpgradeId]
    conditional_upgrades: list[tuple[UpgradeId, Callable]]
    gas_target: Union[int, Callable]
    worker_cap: Union[int, Callable]
    observer_target: Union[int, Callable]
    warp_prism_target: Union[int, Callable]
    gateway_thresholds: list[tuple[int, int]]
    forge_count: Union[int, Callable]
    chrono_priority: list[UnitTypeId]
    conditional_structures: list[tuple[UnitTypeId, Callable]]
    army_composition_2: dict[UnitTypeId, dict] = field(default_factory=dict)  # Post-Archon composition (used when archon % >= threshold)
    detection_cannons: Union[bool, Callable] = False
    """Whether to build Pylon+PhotonCannon behind mineral lines at each base when
detection is needed. Set True for builds without natural detection (e.g. 2021
Stalker build). Uses _needs_detection_cannons predicate when callable."""
    economy_switch_threshold: Union[str, None] = None
    """Economy state that triggers a one-way switch from army_composition_0 to
army_composition_1. Valid values: 'moderate', 'full', or None (no switch).
Once triggered, the switch never reverts even if economy drops."""
    archon_switch_gas_requirement: Union[int, Callable] = 0
    """Minimum gas geysers required for the Archon switch to trigger.
Prevents the Archon composition switch until gas infrastructure can sustain
HT production. Default 0 = no gas requirement (always allow switch).
Set to 6 for gas-gated builds like the 2021 Stalker build."""


# --- 2023 PvT Standard (Robo-Centric) ---
# Wraps all current hardcoded values — zero regression.
PVT_STANDARD_2023_PROFILE = BuildProfile(
    army_composition_0={},  # Set at runtime from macro.py STANDARD_ARMY_0
    army_composition_1={},  # Set at runtime from macro.py STANDARD_ARMY_1
    archon_switch_threshold=0.15,
    upgrade_order=[
        UpgradeId.WARPGATERESEARCH,
        UpgradeId.EXTENDEDTHERMALLANCE,
        UpgradeId.CHARGE,
        UpgradeId.BLINKTECH,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
    ],
    conditional_upgrades=[],  # Populated after import in macro.py (avoids circular deps)
    gas_target=lambda bot: len(bot.townhalls) * 2,
    worker_cap=lambda bot: 90 if bot.game_state >= 1 else 66,
    observer_target=3,
    warp_prism_target=lambda bot: 1 if _needs_warp_prism(bot) else 0,
    gateway_thresholds=[(1, 3), (3, 5), (5, 8)],
    forge_count=lambda bot: 2 if len(bot.townhalls.ready) >= 4 else (1 if len(bot.townhalls.ready) >= 2 else 0),
    chrono_priority=[
        UnitTypeId.ROBOTICSBAY,
        UnitTypeId.FORGE,
        UnitTypeId.TWILIGHTCOUNCIL,
        UnitTypeId.CYBERNETICSCORE,
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.GATEWAY,
        UnitTypeId.NEXUS,
    ],
    conditional_structures=[],  # No reactive structures needed — Robo is in core path
)

# --- Detection predicate for 2021 Stalker build ---
# The 2021 build has no Robo in its core path, so it can't build Observers.
# Two separate predicates:
#   _needs_robo_for_detection: triggers Robo construction (stops once Robo exists)
#   _needs_observer: triggers Observer training (stays True as long as detection is needed)
_CLOAKED_THREAT_UNITS = {
    UnitTypeId.BANSHEE, UnitTypeId.WRAITH, UnitTypeId.DARKTEMPLAR,
    UnitTypeId.LURKERMP, UnitTypeId.LURKERMPBURROWED,
    UnitTypeId.WIDOWMINE, UnitTypeId.WIDOWMINEBURROWED,
}
_CLOAKED_THREAT_STRUCTURES = {
    UnitTypeId.ARMORY, UnitTypeId.STARPORTTECHLAB,
    UnitTypeId.FACTORYTECHLAB,
    UnitTypeId.DARKSHRINE, UnitTypeId.LURKERDENMP,
}


def _needs_robo_for_detection(bot) -> bool:
    """Return True if the 2021 Stalker build should build a Robo for detection.
    Stops triggering once a Robo exists or is pending (no duplicate orders).
    """
    if bot.structures(UnitTypeId.ROBOTICSFACILITY).amount > 0:
        return False
    if cy_structure_pending_ares(bot, UnitTypeId.ROBOTICSFACILITY) > 0:
        return False

    for unit in bot.enemy_units:
        if unit.type_id in _CLOAKED_THREAT_UNITS:
            return True
        # Any cloaked/burrowed unit we can see (even partially) means detection is needed
        if unit.is_cloaked or unit.is_burrowed:
            return True

    for structure in bot.enemy_structures:
        if structure.type_id in _CLOAKED_THREAT_STRUCTURES:
            return True

    return False


def _needs_observer(bot) -> bool:
    """Return True if the 2021 Stalker build should train an Observer.
    Stays True as long as detection-requiring threats exist, even after the Robo is built.
    This ensures the Observer actually gets trained after the Robo completes.
    """
    for unit in bot.enemy_units:
        if unit.type_id in _CLOAKED_THREAT_UNITS:
            return True
        if unit.is_cloaked or unit.is_burrowed:
            return True

    for structure in bot.enemy_structures:
        if structure.type_id in _CLOAKED_THREAT_STRUCTURES:
            return True

    return False


def _needs_detection_cannons(bot) -> bool:
    """Return True if cloaked/burrowed threats are detected.
    This is the trigger condition — once True, the detection cannon system activates
    for ALL bases simultaneously and continues until every base is complete, even
    if the threat moves out of vision or dies. New expansions also get protection.
    """
    for unit in bot.enemy_units:
        if unit.type_id in _CLOAKED_THREAT_UNITS:
            return True
        if unit.is_cloaked or unit.is_burrowed:
            return True
    for structure in bot.enemy_structures:
        if structure.type_id in _CLOAKED_THREAT_STRUCTURES:
            return True
    return False


def _needs_warp_prism(bot) -> bool:
    """Return True if the active build should build a Warp Prism.
    Requires completed Robo, TemplarArchive, and RoboticBay, plus supply >= 60.
    Only triggers when no Warp Prism exists or is pending (limits to 1 total).
    """
    total_prisms = (bot.units(UnitTypeId.WARPPRISM).amount +
                    bot.units(UnitTypeId.WARPPRISMPHASING).amount +
                    cy_unit_pending(bot, UnitTypeId.WARPPRISM))
    return (bot.structures(UnitTypeId.ROBOTICSFACILITY).ready
            and bot.structures(UnitTypeId.TEMPLARARCHIVE).ready
            and bot.structures(UnitTypeId.ROBOTICSBAY).ready
            and bot.supply_used >= 60
            and total_prisms == 0)


# --- 2021 PvT Stalker-Centric ---
# Twilight + Blink opener, Stalker/Zealot army, 2 gas, aggressive gateway scaling.
# Economy-gated: switches to HT/Sentry composition at moderate economy.
# Upgrades gated by economy: Weapons/Armor 2 at moderate+, 3 at full.
PVT_STALKER_2021_PROFILE = BuildProfile(
    army_composition_0={},  # Set at runtime from macro.py PVT_STALKER_2021_ARMY
    army_composition_1={},  # Set at runtime from macro.py PVT_STALKER_2021_ARMY_MODERATE
    army_composition_2={},  # Set at runtime from macro.py PVT_STALKER_2021_ARMY_FULL
    archon_switch_threshold=0.15,  # Switch to post-Archon comp when Archons reach 15%
    upgrade_order=[
        UpgradeId.WARPGATERESEARCH,
        UpgradeId.BLINKTECH,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
        UpgradeId.PSISTORMTECH,
        UpgradeId.CHARGE,
    ],
    conditional_upgrades=[
        # Moderate+ economy: Weapons 2 and Armor 2
        (UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2, lambda bot: _get_economy_state(bot) in ("moderate", "full")),
        (UpgradeId.PROTOSSGROUNDARMORSLEVEL2, lambda bot: _get_economy_state(bot) in ("moderate", "full")),
        # Full economy only: Weapons 3 and Armor 3
        (UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3, lambda bot: _get_economy_state(bot) == "full"),
        (UpgradeId.PROTOSSGROUNDARMORSLEVEL3, lambda bot: _get_economy_state(bot) == "full"),
    ],
    gas_target=lambda bot: len(bot.townhalls) * 2,  # Scales with bases (same as 2023)
    worker_cap=70,
    observer_target=lambda bot: 1 if _needs_observer(bot) else 0,
    warp_prism_target=lambda bot: 1 if _get_economy_state(bot) in ("moderate", "full") else 0,
    gateway_thresholds=[(1, 3), (3, 8), (4, 12)],
    forge_count=1,
    chrono_priority=[
        UnitTypeId.TWILIGHTCOUNCIL,
        UnitTypeId.TEMPLARARCHIVE,  # Psionic Storm research
        UnitTypeId.FORGE,
        UnitTypeId.CYBERNETICSCORE,
        UnitTypeId.GATEWAY,
        UnitTypeId.NEXUS,
    ],
    conditional_structures=[(UnitTypeId.ROBOTICSFACILITY, _needs_robo_for_detection)],  # Reactive Robo for detection
    detection_cannons=True,  # Build Pylon+Cannon behind mineral lines when detection is needed
    economy_switch_threshold="moderate",  # One-way switch to HT/Sentry composition at moderate economy
    archon_switch_gas_requirement=6,  # Need 6 gas geysers before Archon switch (sustains HT production)
)

# Lookup dict: build name (from protoss_builds.yml) → BuildProfile
BUILD_PROFILES: dict[str, BuildProfile] = {
    "B2GM_PVT_Standard_Build": PVT_STANDARD_2023_PROFILE,
    "B2GM_PVT_Stalker_Centric_2021": PVT_STALKER_2021_PROFILE,
}

# --- PvZ Standard (Robo-Centric) ---
# Same robo path as PvT 2023 — Robo → Observer → Immortal → RoboBay.
# Uses STANDARD_ARMY_0/1 (same as PvT). Same economy targets.
PVZ_STANDARD_PROFILE = BuildProfile(
    army_composition_0={},  # Set at runtime from macro.py STANDARD_ARMY_0
    army_composition_1={},  # Set at runtime from macro.py STANDARD_ARMY_1
    archon_switch_threshold=0.15,
    upgrade_order=[
        UpgradeId.WARPGATERESEARCH,
        UpgradeId.EXTENDEDTHERMALLANCE,
        UpgradeId.CHARGE,
        UpgradeId.BLINKTECH,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
    ],
    conditional_upgrades=[],  # Populated after import in macro.py
    gas_target=lambda bot: len(bot.townhalls) * 2,
    worker_cap=lambda bot: 90 if bot.game_state >= 1 else 66,
    observer_target=3,
    warp_prism_target=0,  # No Warp Prism in PvZ standard
    gateway_thresholds=[(1, 3), (3, 5), (5, 8)],
    forge_count=lambda bot: 2 if len(bot.townhalls.ready) >= 4 else (1 if len(bot.townhalls.ready) >= 2 else 0),
    chrono_priority=[
        UnitTypeId.ROBOTICSBAY,
        UnitTypeId.FORGE,
        UnitTypeId.TWILIGHTCOUNCIL,
        UnitTypeId.CYBERNETICSCORE,
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.GATEWAY,
        UnitTypeId.NEXUS,
    ],
    conditional_structures=[],  # No reactive structures needed — Robo is in core path
)

# --- PvP 2-Gate Expand ---
# Blink-first upgrade order, PVP_ARMY_0/1 compositions, higher archon threshold.
PVP_2GATE_PROFILE = BuildProfile(
    army_composition_0={},  # Set at runtime from macro.py PVP_ARMY_0
    army_composition_1={},  # Set at runtime from macro.py PVP_ARMY_1
    archon_switch_threshold=0.30,
    upgrade_order=[
        UpgradeId.WARPGATERESEARCH,
        UpgradeId.BLINKTECH,
        UpgradeId.EXTENDEDTHERMALLANCE,
        UpgradeId.CHARGE,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
        UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
    ],
    conditional_upgrades=[],  # Populated after import in macro.py
    gas_target=lambda bot: len(bot.townhalls) * 2,
    worker_cap=lambda bot: 90 if bot.game_state >= 1 else 66,
    observer_target=2,
    warp_prism_target=0,  # No Warp Prism in PvP standard
    gateway_thresholds=[(1, 3), (3, 5), (5, 8)],
    forge_count=lambda bot: 2 if len(bot.townhalls.ready) >= 4 else (1 if len(bot.townhalls.ready) >= 2 else 0),
    chrono_priority=[
        UnitTypeId.TWILIGHTCOUNCIL,
        UnitTypeId.ROBOTICSBAY,
        UnitTypeId.FORGE,
        UnitTypeId.CYBERNETICSCORE,
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.GATEWAY,
        UnitTypeId.NEXUS,
    ],
    conditional_structures=[],  # No reactive structures needed — Robo is in core path
)

# Add all profiles to the lookup dict
BUILD_PROFILES.update({
    "B2GM_PVZ_Standard_Build": PVZ_STANDARD_PROFILE,
    "B2GM_PVP_2-Gate_Expand": PVP_2GATE_PROFILE,
})


def _resolve(value, bot):
    """Resolve a BuildProfile field that may be int or Callable[[bot], int]."""
    return value(bot) if callable(value) else value


def get_active_profile(bot) -> BuildProfile:
    """Look up the BuildProfile for the currently active build order.

    Falls back to PVT_STANDARD_2023_PROFILE for unknown build names,
    ensuring no regression for builds not yet in the profile dict.
    """
    name = bot.build_order_runner.chosen_opening
    return BUILD_PROFILES.get(name, PVT_STANDARD_2023_PROFILE)
