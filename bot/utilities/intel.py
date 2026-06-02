"""
Utility functions for gathering intelligence about the enemy.

Purpose: Track enemy intel quality and cannon rush detection
Key Decisions: Use unit.age for staleness tracking; filter expired ghosts (age >= 30s) from UnitCacheManager
Limitations: UnitCacheManager never removes units, so we must filter by MEMORY_EXPIRY_TIME
"""
from typing import TYPE_CHECKING

import numpy as np
from map_analyzer import MapData
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from cython_extensions import cy_distance_to

from ares.consts import WORKER_TYPES

from bot.constants import (
    RAMP_CHOKE_RADIUS,
    MAP_CHOKE_RADIUS,
    CHOKE_GRID_WEIGHT,
    CHOKE_MAX_WIDTH,
    MEMORY_EXPIRY_TIME,
    VISIBLE_AGE_THRESHOLD,
    STALENESS_WINDOW,
    FRESH_INTEL_THRESHOLD,
    URGENCY_BUILD_RATE,
    URGENCY_DECAY_RATE,
)

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


def create_choke_grid(bot: "PiG_Bot") -> np.ndarray:
    """
    Create a grid marking choke/ramp areas for O(1) formation skip detection.
    
    Call this once in on_start and store as bot.choke_grid.
    Cells with value > 1.0 indicate "near choke, skip formation logic".
    
    Args:
        bot: Bot instance with access to map_data and game_info
        
    Returns:
        np.ndarray: Grid where values > 1.0 mark choke/ramp zones
    """
    map_data: MapData = bot.mediator.get_map_data_object
    grid = map_data.get_pyastar_grid()
    
    # Mark ramps (top and bottom centers)
    for ramp in bot.game_info.map_ramps:
        grid = map_data.add_cost(
            ramp.top_center, RAMP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )
        grid = map_data.add_cost(
            ramp.bottom_center, RAMP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )
    
    # Mark map_analyzer choke points
    for choke in map_data.map_chokes:
        grid = map_data.add_cost(
            choke.center, MAP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )
    
    return grid


def create_narrow_choke_points(bot: "PiG_Bot") -> dict[Point2, float]:
    """
    Build a dict mapping choke tile → choke width for chokes ≤ CHOKE_MAX_WIDTH.
    
    Wide-open "chokes" that map_analyzer classifies don't actually bottleneck armies,
    so we pre-filter at game start and only keep tiles from real narrow passages.
    
    Width measurement per choke type:
      - RawChoke: md_pl_choke.min_length (exact C-extension measurement)
      - MDRamp / VisionBlockerArea: side_a ↔ side_b distance
      - Fallback: Polygon.width (approximate, 0.5-1.5x real)
    
    Returns dict so is_choke_between() can look up the width of the matched choke
    and compare it against the effective army widths at engagement time.
    
    Call once in on_start, store as bot.narrow_choke_points.
    """
    map_data: MapData = bot.mediator.get_map_data_object
    choke_width_map: dict[Point2, float] = {}
    
    for choke in map_data.map_chokes:
        # Measure choke width using the best available metric
        width = None
        if hasattr(choke, "md_pl_choke") and choke.md_pl_choke is not None:
            # RawChoke: exact min distance between sides from C extension
            width = choke.md_pl_choke.min_length
        elif choke.side_a is not None and choke.side_b is not None:
            # MDRamp / VisionBlockerArea: distance between computed sides
            sa = Point2(choke.side_a) if not isinstance(choke.side_a, Point2) else choke.side_a
            sb = Point2(choke.side_b) if not isinstance(choke.side_b, Point2) else choke.side_b
            width = cy_distance_to(sa, sb)
        
        if width is None:
            # Fallback: approximate Polygon.width
            width = choke.width
        
        if width <= CHOKE_MAX_WIDTH:
            for point in choke.points:
                choke_width_map[point] = width
    return choke_width_map


def is_near_choke(choke_grid: np.ndarray, position: Point2) -> bool:
    """
    O(1) check if position is near a choke/ramp using precomputed grid.
    
    Args:
        choke_grid: Precomputed choke grid from create_choke_grid()
        position: Position to check (typically army_center)
        
    Returns:
        True if near choke (skip formation), False otherwise
    """
    pos = position.rounded
    return choke_grid[pos[0], pos[1]] > 1.0


def get_enemy_cannon_rushed(bot, detection_radius: float = 25.0) -> bool:
    """Check if the enemy is cannon rushing.
    
    Args:
        bot: The bot instance
        detection_radius: Distance in game units to check around each base (default: 25.0)
        
    Returns:
        bool: True if cannon rush is detected, False otherwise
    """
    # Only check against Protoss
    if bot.enemy_race != Race.Protoss:
        return False
        
    # Only check in early game (before 3 minutes)
    if bot.time > 180.0:  # 3 minutes
        return False
    
    try:
        # Get main base and natural expansion positions
        main_base = bot.start_location
        natural_expansion = bot.mediator.get_own_nat  # This gets the natural expansion position
        
        # Look for enemy pylons and cannons near our bases
        enemy_structures = bot.enemy_structures.filter(
            lambda s: s.type_id in {UnitID.PYLON, UnitID.PHOTONCANNON} and 
                     (cy_distance_to(s.position, main_base) < detection_radius or 
                      cy_distance_to(s.position, natural_expansion) < detection_radius)
        )
        
        # Count pylons and cannons
        pylon_count = sum(1 for s in enemy_structures if s.type_id == UnitID.PYLON)
        cannon_count = sum(1 for s in enemy_structures if s.type_id == UnitID.PHOTONCANNON)
        
        # If we see cannons or multiple pylons near our bases, it's likely a cannon rush
        if cannon_count > 0 or pylon_count > 1:
            return True
            
        return False
        
    except Exception as e:
        print(f"Error in cannon rush detection: {e}")
        return False


def get_enemy_intel_quality(bot: "PiG_Bot") -> dict:
    """
    Analyze the freshness of our enemy army intel.
    
    Uses unit.age from python-sc2 to determine how trustworthy our
    combat simulation data is. Filters out expired ghost units that
    UnitCacheManager retains but UnitMemoryManager has already expired.
    
    Returns:
        dict with keys:
            - has_intel: bool - have we ever seen enemy army?
            - avg_age: float - average age of active enemy unit data (seconds)
            - visible_count: int - units we can currently see (age < 3s)
            - memory_count: int - units from recent memory (3s-30s)
            - expired_count: int - ghost units filtered out (age >= 30s)
            - freshness: float - 0-1 score (1 = all fresh, 0 = all stale)
    """
    # Get all cached enemy army units (UnitCacheManager never removes units,
    # so this includes expired ghosts with age >> 30s)
    all_cached = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES
    ]
    
    # No units in cache at all
    if not all_cached:
        if bot._enemy_army_ever_seen:
            # We've seen the enemy before but cache is empty — intel is stale,
            # not fresh. Returning 1.0 here would prevent urgency from building
            # and block scout dispatch.
            return {
                "has_intel": True,
                "avg_age": float('inf'),
                "visible_count": 0,
                "memory_count": 0,
                "expired_count": 0,
                "freshness": 0.0,
            }
        return {
            "has_intel": False,  # Never seen enemy army
            "avg_age": float('inf'),
            "visible_count": 0,
            "memory_count": 0,
            "expired_count": 0,
            "freshness": 0.0,
        }
    
    # Filter to active units only: exclude expired ghosts (age >= MEMORY_EXPIRY_TIME)
    # UnitMemoryManager expires at 30s but UnitCacheManager retains them indefinitely.
    # Without this filter, expired ghosts drag the freshness average toward zero.
    active_units = [u for u in all_cached if u.age < MEMORY_EXPIRY_TIME]
    
    expired_count = len(all_cached) - len(active_units)
    
    if not active_units:
        # All cached units expired - we lost track of the enemy army
        return {
            "has_intel": True,
            "avg_age": max(u.age for u in all_cached),
            "visible_count": 0,
            "memory_count": 0,
            "expired_count": expired_count,
            "freshness": 0.0,
        }
    
    # Split active units into visible (recent) and memory (older but not expired)
    visible = [u for u in active_units if u.age < VISIBLE_AGE_THRESHOLD]
    memory = [u for u in active_units if u.age >= VISIBLE_AGE_THRESHOLD]
    
    avg_age = sum(u.age for u in active_units) / len(active_units)
    
    # Freshness: 1.0 = all visible now, decays linearly over STALENESS_WINDOW seconds
    # Only active (non-expired) units contribute, preventing ghost pollution
    freshness = sum(
        max(0.0, 1.0 - u.age / STALENESS_WINDOW) for u in active_units
    ) / len(active_units)
    
    return {
        "has_intel": True,
        "avg_age": avg_age,
        "visible_count": len(visible),
        "memory_count": len(memory),
        "expired_count": expired_count,
        "freshness": freshness,
    }


def update_enemy_intel_tracking(bot: "PiG_Bot") -> None:
    """
    Update enemy intel tracking flags and intel urgency. Call this every frame.
    
    Sets:
        - bot._enemy_army_ever_seen: sticky flag, True once we've seen enemy army
        - bot._last_enemy_army_visible_time: last time we had direct vision of enemy army
        - bot._intel_urgency: 0-1 score, builds when stale, decays when fresh
    """
    # Get enemy army excluding workers
    enemy_army = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES
    ]
    
    # Check if any enemy army is currently visible (age-based check;
    # is_memory is unreliable for UnitCacheManager units since snapshots
    # were taken when visible, so is_memory is always False for them)
    visible_army = [u for u in enemy_army if u.age < VISIBLE_AGE_THRESHOLD]
    
    if visible_army:
        # We currently see enemy army
        bot._enemy_army_ever_seen = True
        bot._last_enemy_army_visible_time = bot.time
        # Populate last-seen timestamps for composition belief decay
        for unit in visible_army:
            bot._enemy_unit_last_seen[unit.tag] = bot.time
    elif enemy_army:
        # We have memory of enemy army but can't see them now
        bot._enemy_army_ever_seen = True
    
    # Update intel urgency (gradual build/decay to avoid oscillation)
    # Always track urgency — if we've never seen the enemy army, that's the
    # most urgent scenario (we need to find them). Previously gated on
    # _enemy_army_ever_seen, which prevented urgency from ever building
    # when the enemy army was never spotted (e.g., scout only saw workers).
    # When composition belief is enabled, use its probability-weighted
    # freshness instead of the binary age-threshold freshness — one source
    # of truth for both combat decisions and scout dispatch.
    use_belief = bot.config.get("Belief", {}).get("enable_composition", False)
    if use_belief:
        freshness = bot.belief_state.composition.freshness
    else:
        intel = get_enemy_intel_quality(bot)
        freshness = intel["freshness"]
    
    # Build urgency when intel is not fresh (below FRESH_INTEL_THRESHOLD).
    # Previously used STALE_INTEL_THRESHOLD (0.2) which only triggered at
    # "BLIND" level, causing urgency to decay during "STALE" (0.2–0.7)
    # and preventing scout dispatch.
    if freshness < FRESH_INTEL_THRESHOLD:
        # Intel is stale or blind - build urgency
        bot._intel_urgency = min(1.0, bot._intel_urgency + URGENCY_BUILD_RATE)
    else:
        # Intel is fresh - slowly decay urgency
        bot._intel_urgency = max(0.0, bot._intel_urgency - URGENCY_DECAY_RATE)
        # Reset worker scout flag so we can scout again next stale period
        bot._worker_scout_sent_this_stale_period = False
