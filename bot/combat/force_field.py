"""Force Field placement for Sentry micro.

Purpose: Compute whether and where to place Force Fields:
         - Main ramp block: single FF at a main-base ramp when enemy is crossing
         - Choke block: FF chain at a refined map choke, perpendicular to passage
          - Army split: chain of FFs through the enemy front line (shifted toward
            our army) to trap front-line units while blocking reinforcements
Key Decisions: Priority chain is main ramp block > choke block > army split. A single
               FF at a real bottleneck (main ramp or narrow choke) is more impactful
               than a generic split through the enemy center. The main ramp block is
               specialized for the two main-base ramps only; all other ramps are
               handled by the general choke-block path (they're in map_chokes as
               MDRamp instances). Choke block uses raycast-refined exact width +
               orientation from choke_grid.py. Pooled energy across all sentries,
               greedy assignment.
Limitations: Army split assumes roughly elliptical enemy formation; L-shaped/
              scattered armies may get suboptimal split lines. The front-line
              shift assumes the enemy is advancing toward our army (split axis
              derived from own_center - enemy_center). Choke block only fires
              when is_choke_between() has already detected a choke.
"""

import math
from typing import List, NamedTuple, Optional

import numpy as np
from sc2.game_info import Ramp as Sc2Ramp
from sc2.position import Point2
from sc2.unit import Unit

from cython_extensions import cy_distance_to, cy_find_units_center_mass
from cython_extensions.general_utils import cy_in_pathing_grid_ma

from ares.consts import WORKER_TYPES
from ares.dicts.unit_data import UNIT_DATA

from bot.constants import (
    COMMON_UNIT_IGNORE_TYPES,
    FF_CAST_RANGE,
    FF_CHOKE_BLOCK_MAX_WIDTH,
    FF_CHOKE_BLOCK_MIN_VALUE,
    FF_CHOKE_BLOCK_RADIUS,
    FF_ENERGY_COST,
    FF_MAIN_RAMP_BLOCK_MIN_VALUE,
    FF_MAIN_RAMP_BLOCK_RADIUS,
    FF_OVERLAP,
    FF_RADIUS,
    FF_SPLIT_FRONT_FRACTION,
    FF_SPLIT_MIN_ENEMIES,
)
from bot.utilities.choke_grid import RefinedChoke


class FFResult(NamedTuple):
    """Result from any compute_ff_* function with assignments and debug info."""
    assignments: list[tuple[Unit, Point2]]
    enemy_center: Point2


def _normalize(v: Point2) -> Point2:
    """Return unit-length direction vector, or (1, 0) if zero-length."""
    length = math.sqrt(v.x ** 2 + v.y ** 2)
    if length < 1e-6:
        return Point2((1.0, 0.0))
    return Point2((v.x / length, v.y / length))


def _ramp_center(ramp: Sc2Ramp) -> Point2:
    """Compute the centroid of ALL ramp tiles — the true center of the ramp.

    Unlike top_center/bottom_center (centroids of only upper/lower subsets),
    this accounts for both left-right and top-bottom asymmetry. On asymmetric
    ramps, the upper/lower centroids are pulled toward the wider side.
    """
    pts = ramp.points
    n = len(pts)
    return Point2((
        sum(p.x for p in pts) / n,
        sum(p.y for p in pts) / n,
    ))


def _filter_ground_combat(enemies: List[Unit]) -> list[Unit]:
    """Filter to ground combat enemies (no flying, no workers, no ignores, no structures)."""
    return [
        e for e in enemies
        if not e.is_flying
        and e.type_id not in COMMON_UNIT_IGNORE_TYPES
        and e.type_id not in WORKER_TYPES
        and not e.is_structure
    ]


def _greedy_assign(
    ff_positions: list[Point2], sentries: List[Unit],
) -> Optional[list[tuple[Unit, Point2]]]:
    """Greedy assignment: each FF to closest sentry with energy + in cast range.

    Returns the assignment list, or None if any FF can't be assigned (whole
    placement fails — no partial chains). Pools energy across all sentries.
    """
    assignments: list[tuple[Unit, Point2]] = []
    sentry_energy: dict[int, float] = {s.tag: s.energy for s in sentries}

    for pos in ff_positions:
        best_sentry: Optional[Unit] = None
        best_dist = float("inf")
        for s in sentries:
            if sentry_energy[s.tag] < FF_ENERGY_COST:
                continue
            dist = cy_distance_to(s.position, pos)
            if dist > FF_CAST_RANGE:
                continue
            if dist < best_dist:
                best_dist = dist
                best_sentry = s

        if best_sentry is None:
            return None

        assignments.append((best_sentry, pos))
        sentry_energy[best_sentry.tag] -= FF_ENERGY_COST

    return assignments


def compute_ff_main_ramp_block(
    enemies: List[Unit],
    sentries: List[Unit],
    own_ramp: Optional[Sc2Ramp],
    enemy_ramp: Optional[Sc2Ramp],
    active_ffs: Optional[List[Point2]] = None,
) -> Optional[FFResult]:
    """Check if the enemy center is crossing a main ramp and place a single FF to block it.

    Specialized fast path for the two main-base ramps only (our ramp and the
    enemy's ramp). Non-main ramps (naturals, thirds, mid-map) are handled by
    the general choke-block path via create_narrow_choke_points() →
    is_choke_between() → compute_ff_choke_block().

    A single FF at the ramp center is sufficient — main ramps are narrow
    enough (~3-5 tiles) that one FF seals the bottleneck.

    Places the FF at the centroid of ALL ramp tiles — this is the true center
    of the ramp regardless of shape, accounting for both left-right and
    top-bottom asymmetry.

    Checks active Force Fields to avoid stacking a second FF on top of an
    existing one at the same ramp position.

    Args:
        enemies: All nearby enemy units (pre-filtered for combat relevance)
        sentries: All friendly sentries (energy pooled across all of them)
        own_ramp: Our main Ramp object (from bot.main_base_ramp), or None
        enemy_ramp: Enemy main Ramp object (from bot.mediator.get_enemy_ramp), or None
        active_ffs: List of active FF center positions from bot.mediator.get_forcefield_positions,
            or None. Used to avoid stacking duplicate FFs.

    Returns:
        FFResult with a single FF assignment at the ramp center, or None
        if no main ramp block opportunity is detected.
    """
    ground_enemies = _filter_ground_combat(enemies)

    # Army value gate: is this force worth spending 50 energy on?
    enemy_value = sum(
        UNIT_DATA.get(e.type_id, {}).get("army_value", 0.0)
        for e in ground_enemies
    )
    if enemy_value < FF_MAIN_RAMP_BLOCK_MIN_VALUE:
        return None

    # Enemy center of mass
    enemy_center_arr, _ = cy_find_units_center_mass(ground_enemies, 20.0)
    enemy_center = Point2(enemy_center_arr)

    # Check if enemy center is near our ramp or their ramp.
    # Use the centroid of ALL ramp tiles (ramp.points) — this is the true
    # center of the ramp regardless of shape, accounting for both left-right
    # and top-bottom asymmetry. top_center/bottom_center are centroids of
    # only the upper/lower subsets, which are offset on asymmetric ramps.
    target_pos: Optional[Point2] = None
    best_dist = FF_MAIN_RAMP_BLOCK_RADIUS

    if own_ramp is not None:
        ramp_center = _ramp_center(own_ramp)
        dist = cy_distance_to(enemy_center, ramp_center)
        if dist < best_dist:
            best_dist = dist
            target_pos = ramp_center

    if enemy_ramp is not None:
        ramp_center = _ramp_center(enemy_ramp)
        dist = cy_distance_to(enemy_center, ramp_center)
        if dist < best_dist:
            best_dist = dist
            target_pos = ramp_center

    if target_pos is None:
        return None

    # Don't stack FFs: if there's already an active FF near the target, skip
    if active_ffs:
        for ff_pos in active_ffs:
            if cy_distance_to(ff_pos, target_pos) < FF_RADIUS:
                return None

    # Find closest sentry with enough energy and in cast range
    for s in sentries:
        if s.energy >= FF_ENERGY_COST and cy_distance_to(s.position, target_pos) <= FF_CAST_RANGE:
            return FFResult(
                assignments=[(s, target_pos)],
                enemy_center=enemy_center,
            )

    return None


def compute_ff_choke_block(
    enemies: List[Unit],
    sentries: List[Unit],
    refined: RefinedChoke,
    active_ffs: Optional[List[Point2]] = None,
) -> Optional[FFResult]:
    """Place an FF chain at a refined choke to block the enemy crossing through it.

    Uses the exact narrowest point + perpendicular orientation from
    refine_choke_with_raycast(). The FF chain is centered on the refined
    choke center and spans the measured width, perpendicular to the passage.

    Gates:
      - Enemy army value >= FF_CHOKE_BLOCK_MIN_VALUE (worth the energy)
      - Enemy center within FF_CHOKE_BLOCK_RADIUS of the choke center (actively crossing)
      - Choke width <= FF_CHOKE_BLOCK_MAX_WIDTH (wider chokes need too many FFs)
      - No active FF already near the choke center (anti-stack)

    Args:
        enemies: All nearby enemy units (pre-filtered for combat relevance)
        sentries: All friendly sentries (energy pooled across all of them)
        refined: RefinedChoke from get_or_refine_choke() — exact center, width, orientation
        active_ffs: List of active FF center positions, or None. Anti-stack check.

    Returns:
        FFResult with FF chain assignments at the choke, or None if no block
        opportunity is detected.
    """
    ground_enemies = _filter_ground_combat(enemies)

    # Army value gate
    enemy_value = sum(
        UNIT_DATA.get(e.type_id, {}).get("army_value", 0.0)
        for e in ground_enemies
    )
    if enemy_value < FF_CHOKE_BLOCK_MIN_VALUE:
        return None

    # Enemy center of mass
    enemy_center_arr, _ = cy_find_units_center_mass(ground_enemies, 20.0)
    enemy_center = Point2(enemy_center_arr)

    # Enemy must be actively crossing the choke, not just nearby
    if cy_distance_to(enemy_center, refined.center) > FF_CHOKE_BLOCK_RADIUS:
        return None

    # Width gate: don't block chokes too wide to seal
    if refined.width > FF_CHOKE_BLOCK_MAX_WIDTH:
        return None

    # Anti-stack: skip if an active FF is already at the choke center
    if active_ffs:
        for ff_pos in active_ffs:
            if cy_distance_to(ff_pos, refined.center) < FF_RADIUS:
                return None

    # FF chain along the perpendicular, centered on the refined choke center
    ff_spacing = 2 * FF_RADIUS - FF_OVERLAP
    num_ffs = max(1, math.ceil(refined.width / ff_spacing))
    # Single FF for narrow chokes; chain for wider ones
    ff_positions: list[Point2] = []
    for i in range(num_ffs):
        # Offset from center: spread FFs symmetrically
        t = (i - (num_ffs - 1) / 2.0) * ff_spacing
        pos = Point2((
            refined.center.x + refined.perp.x * t,
            refined.center.y + refined.perp.y * t,
        ))
        ff_positions.append(pos)

    if not ff_positions:
        return None

    # Pooled energy check
    total_energy = sum(s.energy for s in sentries)
    if total_energy < len(ff_positions) * FF_ENERGY_COST:
        return None

    assignments = _greedy_assign(ff_positions, sentries)
    if assignments is None:
        return None

    return FFResult(assignments=assignments, enemy_center=enemy_center)


def compute_ff_split(
    enemies: List[Unit],
    sentries: List[Unit],
    own_center: Point2,
    ground_grid: np.ndarray,
) -> Optional[FFResult]:
    """Determine if a force field split is feasible and return cast assignments.

    Algorithm:
      1. Filter to ground combat enemies (no flying, no workers, no ignores)
      2. Compute enemy center of mass
      3. Determine split axis: direction from enemy center toward our army
      4. Project enemies onto perpendicular + split axes for chain extent
      5. Shift the FF line from enemy center toward our army by FF_SPLIT_FRONT_FRACTION
         of the enemy front-line extent, trapping front-line units on our side
      6. Walk along the perpendicular line placing FF positions
      7. Validate edge positions (beyond enemy cluster) are pathable
      8. Pool energy across all sentries; greedy assign each FF to closest
         sentry with enough remaining energy and within cast range

    The split line is shifted toward our army (not through enemy center) so
    the enemy front line is trapped against our army while reinforcements
    behind the FF can't reach the fight. Choke/ramp situations are handled
    by compute_ff_main_ramp_block() and compute_ff_choke_block() which place
    FFs at real bottlenecks, and are more effective than trying to split
    through a generic point.

    Args:
        enemies: All nearby enemy units (pre-filtered for combat relevance)
        sentries: All friendly sentries (energy pooled across all of them)
        own_center: Our army center of mass (determines split direction + shift)
        ground_grid: Ground grid for pathing validation (cy_in_pathing_grid_ma)

    Returns:
        FFResult with assignments + debug info, or None if split not feasible.
        Assignments is a list of (sentry, target_position) pairs. Each sentry
        may appear multiple times if it has enough energy for multiple FFs.
    """
    # 1. Filter to ground combat enemies only
    ground_enemies = _filter_ground_combat(enemies)
    if len(ground_enemies) < FF_SPLIT_MIN_ENEMIES:
        return None

    # 2. Enemy center of mass
    enemy_center_arr, _ = cy_find_units_center_mass(ground_enemies, 20.0)
    enemy_center = Point2(enemy_center_arr)

    # 3. Split axis: direction from enemy center toward our army
    #    This cuts the army perpendicular to the engagement axis,
    #    separating the front (engaging us) from the back (reinforcing)
    split_dir = _normalize(own_center - enemy_center)
    # Perpendicular direction: the line we place FFs along
    perp_dir = Point2((-split_dir.y, split_dir.x))

    # 4. Project enemies onto both axes for FF chain placement
    perp_projections: list[float] = []
    split_projections: list[float] = []
    for e in ground_enemies:
        offset = e.position - enemy_center
        proj_perp = offset.x * perp_dir.x + offset.y * perp_dir.y
        perp_projections.append(proj_perp)
        proj_split = offset.x * split_dir.x + offset.y * split_dir.y
        split_projections.append(proj_split)

    # Shift the FF line toward our army by a fraction of the enemy's
    # front-line extent. This traps the front line on our side while
    # blocking backline reinforcements, instead of cutting through center.
    front_extent = max(split_projections) if split_projections else 0.0
    line_shift = split_dir * (front_extent * FF_SPLIT_FRONT_FRACTION)
    line_origin = Point2((
        enemy_center.x + line_shift.x,
        enemy_center.y + line_shift.y,
    ))

    # 6. Calculate FF chain along the perpendicular line
    #    Extend slightly beyond the enemy cluster to ensure full coverage
    cluster_min_perp = min(perp_projections)
    cluster_max_perp = max(perp_projections)
    min_perp = cluster_min_perp - FF_RADIUS
    max_perp = cluster_max_perp + FF_RADIUS
    span = max_perp - min_perp

    # Each FF covers 2*FF_RADIUS diameter, with FF_OVERLAP overlap
    ff_spacing = 2 * FF_RADIUS - FF_OVERLAP
    num_ffs = max(1, math.ceil(span / ff_spacing))

    # 7. Generate FF positions along the split line, shifted toward our army
    #    Positions within the enemy cluster span are pathable by definition
    #    (enemies are standing there). Only validate positions that extend
    #    beyond the cluster, where the line may cross cliffs/water.
    ff_positions: list[Point2] = []
    for i in range(num_ffs):
        t = min_perp + ff_spacing * (i + 0.5)
        pos = Point2((
            line_origin.x + perp_dir.x * t,
            line_origin.y + perp_dir.y * t,
        ))
        # Only pathing-check positions outside the enemy cluster span.
        # Inside the cluster, enemies are standing there — it's pathable.
        is_within_cluster = cluster_min_perp <= t <= cluster_max_perp
        if not is_within_cluster and not cy_in_pathing_grid_ma(ground_grid, pos):
            # Position extends beyond the cluster onto potentially unpathable
            # terrain (cliffs, water, walls). Try nudging along the split axis.
            found_valid = False
            for offset_dist in [0.5, -0.5, 1.0, -1.0]:
                alt_pos = Point2((
                    pos.x + split_dir.x * offset_dist,
                    pos.y + split_dir.y * offset_dist,
                ))
                if cy_in_pathing_grid_ma(ground_grid, alt_pos):
                    pos = alt_pos
                    found_valid = True
                    break
            if not found_valid:
                # Can't place this edge FF — skip it rather than waste energy
                continue
        ff_positions.append(pos)

    if not ff_positions:
        return FFResult(assignments=[], enemy_center=enemy_center)

    # 8. Check total pooled energy across all sentries
    total_energy = sum(s.energy for s in sentries)
    if total_energy < len(ff_positions) * FF_ENERGY_COST:
        return FFResult(assignments=[], enemy_center=enemy_center)

    # 9. Greedy assignment: each FF to closest sentry with energy + in range
    assignments = _greedy_assign(ff_positions, sentries)
    if assignments is None:
        return FFResult(assignments=[], enemy_center=enemy_center)

    return FFResult(
        assignments=assignments, enemy_center=enemy_center,
    )
