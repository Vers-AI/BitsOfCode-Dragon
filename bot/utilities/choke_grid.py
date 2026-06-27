"""Choke grid, narrow choke point computation, and raycast refinement — map geometry, not enemy intel.

Purpose: Precompute choke/ramp zones for O(1) formation skip detection,
          choke width lookup for engagement decisions, and exact narrowest-point
          + orientation refinement via perpendicular ray marching. Call the
          precompute functions once in on_start; call get_or_refine_choke()
          lazily during combat when a choke is detected.

Key Decisions: Wide-open "chokes" that map_analyzer classifies don't actually
              bottleneck armies, so we pre-filter at game start and only keep
              tiles from real narrow passages. Raycast refinement uses
              cy_in_pathing_grid_ma per-tile (the batched numpy_helper functions
              have unreliable semantics for pathability checks); refinement is
              rare (only when is_choke_between hits) and cached permanently per
              choke tile, so the per-tile cost is negligible.

Limitations: Width measurement varies by choke type (RawChoke exact, MDRamp
            approximate, Polygon.width fallback). Raycast refinement gives
            exact width + orientation but only for the tile detected by
            is_choke_between — it does not scan the full passage for the
            global narrowest point (parked in Backlog).
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from map_analyzer import MapData, VisionBlockerArea
from sc2.position import Point2
from cython_extensions import cy_distance_to

from bot.constants import (
    RAMP_CHOKE_RADIUS,
    MAP_CHOKE_RADIUS,
    CHOKE_GRID_WEIGHT,
    CHOKE_MAX_WIDTH,
    RAYCAST_MAX_WIDTH,
    RAYCAST_STEP_SIZE,
    RAYCAST_ANGLE_SEARCH,
)
from cython_extensions.general_utils import cy_in_pathing_grid_ma

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


@dataclass(frozen=True)
class RefinedChoke:
    """Exact narrowest-point + orientation for a choke, from perpendicular raycasting.

    Produced by refine_choke_with_raycast() and cached per choke tile in
    bot.refined_choke_points. Static terrain never changes, so the cache has no TTL.

    Attributes:
        center: Float-precision center of the narrowest cross-section.
        width: Exact walkable width across the passage (tiles).
        axis: Unit vector along the passage (squad→enemy direction).
        perp: Unit vector perpendicular to the passage — the FF block orientation.
    """
    center: Point2
    width: float
    axis: Point2
    perp: Point2


def create_choke_grid(bot: "PiG_Bot") -> np.ndarray:
    """Create a grid marking choke/ramp areas for O(1) formation skip detection.

    Call once in on_start and store as bot.choke_grid.
    Cells with value > 1.0 indicate "near choke, skip formation logic".
    """
    map_data: MapData = bot.mediator.get_map_data_object
    grid = map_data.get_pyastar_grid()

    for ramp in bot.game_info.map_ramps:
        grid = map_data.add_cost(
            ramp.top_center, RAMP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )
        grid = map_data.add_cost(
            ramp.bottom_center, RAMP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )

    for choke in map_data.map_chokes:
        grid = map_data.add_cost(
            choke.center, MAP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )

    return grid


def create_narrow_choke_points(bot: "PiG_Bot") -> dict[Point2, float]:
    """Build a dict mapping choke tile → choke width for chokes ≤ CHOKE_MAX_WIDTH.

    Iterates map_data.map_chokes, which includes RawChoke (terrain), MDRamp
    (ramps), and VisionBlockerArea (bushes/fog). Vision blockers are skipped
    — they look like narrow passages but don't physically block movement, so
    FF placed there seals nothing.

    Width measurement per choke type:
      - RawChoke: md_pl_choke.min_length (exact C-extension measurement)
      - MDRamp: side_a ↔ side_b distance (farthest walkable points perpendicular
        to the ramp direction — the true ramp width)
      - Other: side_a ↔ side_b distance, else Polygon.width (±50% unreliable)
    """
    map_data: MapData = bot.mediator.get_map_data_object
    choke_width_map: dict[Point2, float] = {}

    for choke in map_data.map_chokes:
        # Vision blockers (bushes/fog) look like narrow passages geometrically
        # but don't physically block movement — FF placed there seals nothing.
        if isinstance(choke, VisionBlockerArea):
            continue

        width = None
        if hasattr(choke, "md_pl_choke") and choke.md_pl_choke is not None:
            # RawChoke: exact narrowest width from the C extension
            width = choke.md_pl_choke.min_length
        elif getattr(choke, "is_ramp", False) and choke.side_a is not None and choke.side_b is not None:
            # MDRamp: side_a/side_b are the farthest walkable points perpendicular
            # to the ramp direction — the true ramp width
            sa = Point2(choke.side_a) if not isinstance(choke.side_a, Point2) else choke.side_a
            sb = Point2(choke.side_b) if not isinstance(choke.side_b, Point2) else choke.side_b
            width = cy_distance_to(sa, sb)
        elif hasattr(choke, "side_a") and choke.side_a is not None and hasattr(choke, "side_b") and choke.side_b is not None:
            sa = Point2(choke.side_a) if not isinstance(choke.side_a, Point2) else choke.side_a
            sb = Point2(choke.side_b) if not isinstance(choke.side_b, Point2) else choke.side_b
            width = cy_distance_to(sa, sb)

        # Polygon.width fallback is ±50% unreliable per its own docstring —
        # only use it if we have no side-based measurement at all
        if width is None:
            width = choke.width

        if width <= CHOKE_MAX_WIDTH:
            for point in choke.points:
                choke_width_map[point] = width

    return choke_width_map


def _march_pathable(
    grid: np.ndarray, start: Point2, direction: Point2, max_steps: int,
) -> int:
    """March from start along direction, return count of consecutive pathable steps.

    Uses cy_in_pathing_grid_ma per tile (weight >= 1.0 and != INFINITY = pathable).
    The start tile itself is NOT checked here — the caller (refine_choke_with_raycast)
    validates it separately. Step 1 is the first tile away from start; returns 0
    if that first step is already unpathable.

    Perf note: O(n) per ray with n = max_steps, but max_steps is small (≤15) and
    refinement only runs when is_choke_between detects a choke (rare) and is
    cached permanently — net cost is negligible.
    """
    count = 0
    for i in range(1, max_steps + 1):
        pos = Point2((
            start.x + direction.x * RAYCAST_STEP_SIZE * i,
            start.y + direction.y * RAYCAST_STEP_SIZE * i,
        ))
        if not cy_in_pathing_grid_ma(grid, pos):
            break
        count = i
    return count


def _normalize(v: Point2) -> Point2:
    """Return unit-length direction vector, or (1, 0) if zero-length."""
    length = math.sqrt(v.x ** 2 + v.y ** 2)
    if length < 1e-6:
        return Point2((1.0, 0.0))
    return Point2((v.x / length, v.y / length))


def _measure_choke_width(
    grid: np.ndarray, center: Point2, perp: Point2,
) -> tuple[float, Point2]:
    """Measure walkable width across the passage at center, perpendicular to perp.

    Marches both directions along perp from center, counts consecutive pathable
    steps. Returns (total_width, refined_center) where refined_center is the
    midpoint of the two boundary tiles.
    """
    half = int(RAYCAST_MAX_WIDTH)
    left = _march_pathable(grid, center, perp, half)
    right = _march_pathable(grid, center, Point2((-perp.x, -perp.y)), half)
    width = float(left + right + 1)  # +1 for the center tile itself
    # Midpoint: shift from center by (right - left)/2 along perp
    offset = (right - left) / 2.0
    refined = Point2((
        center.x + perp.x * offset,
        center.y + perp.y * offset,
    ))
    return width, refined


def refine_choke_with_raycast(
    grid: np.ndarray, choke_tile: Point2, passage_dir: Point2,
) -> RefinedChoke | None:
    """Refine an approximate choke tile into exact narrowest point + orientation.

    Casts perpendicular rays from the choke tile across the passage to measure
    the exact walkable width. Tries the squad→enemy axis ± RAYCAST_ANGLE_SEARCH
    to correct for angled chokes where the detected axis doesn't align with the
    true passage orientation.

    Args:
        grid: Terrain-only pathing grid (map_data.get_pyastar_grid()).
              Must not include building footprints — those create false chokes.
        choke_tile: Approximate choke position from is_choke_between() dict lookup.
        passage_dir: Direction along the passage (squad→enemy axis).

    Returns:
        RefinedChoke with exact center, width, and orientation, or None if the
        choke tile itself is unpathable (degenerate edge case).
    """
    if not cy_in_pathing_grid_ma(grid, choke_tile):
        return None

    axis = _normalize(passage_dir)
    # Candidate orientations: axis, axis + angle, axis - angle
    cos_a = math.cos(RAYCAST_ANGLE_SEARCH)
    sin_a = math.sin(RAYCAST_ANGLE_SEARCH)

    candidates: list[Point2] = [
        axis,
        Point2((axis.x * cos_a - axis.y * sin_a,
                axis.x * sin_a + axis.y * cos_a)),
        Point2((axis.x * cos_a + axis.y * sin_a,
               -axis.x * sin_a + axis.y * cos_a)),
    ]

    best: RefinedChoke | None = None
    for cand_axis in candidates:
        perp = Point2((-cand_axis.y, cand_axis.x))
        width, center = _measure_choke_width(grid, choke_tile, perp)
        if best is None or width < best.width:
            best = RefinedChoke(center=center, width=width, axis=cand_axis, perp=perp)

    return best


def get_or_refine_choke(
    bot: "PiG_Bot", choke_tile: Point2, passage_dir: Point2,
) -> RefinedChoke | None:
    """Return cached RefinedChoke for choke_tile, refining on first access.

    Static terrain never changes, so the cache (bot.refined_choke_points) has no
    TTL — each choke is refined at most once per game. Call during combat when
    is_choke_between() detects a choke; the refinement cost (≤45 cy_in_pathing_grid_ma
    calls for 3 angle candidates × 2 directions × 15 steps) is paid once, then
    O(1) dict lookups forever after.

    Args:
        bot: Bot instance with refined_choke_points dict and mediator for grid access.
        choke_tile: The choke tile from is_choke_between().
        passage_dir: Direction along the passage (squad→enemy).

    Returns:
        RefinedChoke or None if the tile is unpathable.
    """
    cache: dict[Point2, RefinedChoke | None] = bot.refined_choke_points
    if choke_tile in cache:
        return cache[choke_tile]

    grid = bot.mediator.get_map_data_object.get_pyastar_grid()
    refined = refine_choke_with_raycast(grid, choke_tile, passage_dir)
    cache[choke_tile] = refined
    return refined


def refine_all_chokes(bot: "PiG_Bot") -> None:
    """Pre-refine all narrow choke tiles at on_start and populate bot.refined_choke_points.

    Groups choke tiles by their parent choke (using side_a/side_b distance to
    identify unique chokes), picks one representative tile per choke, and
    raycasts it. The passage direction is estimated from side_a → side_b, which
    is the choke's narrowest span — the perpendicular of that is the passage axis.

    This makes refined chokes visible immediately when the squad approaches one,
    without needing enemies on the other side to trigger is_choke_between.

    Cost: ~N raycast calls where N = number of unique narrow chokes (typically 5-15).
    Each call is ≤45 cy_in_pathing_grid_ma checks. Total: <700 calls at on_start.
    """
    map_data: MapData = bot.mediator.get_map_data_object
    grid = map_data.get_pyastar_grid()
    cache: dict[Point2, RefinedChoke | None] = bot.refined_choke_points

    for choke in map_data.map_chokes:
        if isinstance(choke, VisionBlockerArea):
            continue
        if not choke.points:
            continue

        # Estimate passage direction from side_a → side_b if available
        # Only RawChoke and MDRamp have side_a/side_b (set in __init__);
        # base ChokeArea doesn't — guard with hasattr
        passage_dir: Point2 | None = None
        if hasattr(choke, "side_a") and choke.side_a is not None and hasattr(choke, "side_b") and choke.side_b is not None:
            sa = Point2(choke.side_a) if not isinstance(choke.side_a, Point2) else choke.side_a
            sb = Point2(choke.side_b) if not isinstance(choke.side_b, Point2) else choke.side_b
            # side_a → side_b is the narrowest span; perpendicular is the passage axis
            span = sb - sa
            passage_dir = Point2((-span.y, span.x))

        if passage_dir is None:
            # No side info — use the choke center as a fallback direction.
            # The angle search in refine_choke_with_raycast will try ±15° to correct.
            passage_dir = Point2((1.0, 0.0))

        # Use the choke center as the representative tile
        center_tile = Point2((int(choke.center.x), int(choke.center.y)))
        if center_tile in cache:
            continue
        refined = refine_choke_with_raycast(grid, center_tile, passage_dir)
        cache[center_tile] = refined


def _mark_buildings_unpathable(
    grid: np.ndarray, bot: "PiG_Bot", scan_center: Point2, scan_radius: float,
) -> np.ndarray:
    """Return a copy of the terrain grid with all obstacles marked unpathable.

    Includes:
    - Own structures (wall-offs, pylons, etc.)
    - Enemy structures (wall-offs, cannons, etc.)
    - Mineral patches (neutral, block pathing)
    - Vespene gas geysers (neutral, block pathing)

    Destructible rocks are already in the terrain pyastar grid (include_destructables=True),
    so they don't need to be re-marked here.
    """
    grid_copy = grid.copy()

    # Combine all path-blocking objects into a single iteration:
    # own structures, enemy structures, mineral patches, and gas geysers
    all_obstacles = list(bot.structures) + list(bot.enemy_structures)

    # Mineral patches and geysers are neutral units
    if hasattr(bot, "mineral_field"):
        all_obstacles += list(bot.mineral_field)
    if hasattr(bot, "vespene_geyser"):
        all_obstacles += list(bot.vespene_geyser)

    for structure in all_obstacles:
        # Skip structures outside the scan area
        if cy_distance_to(structure.position, scan_center) > scan_radius + 5.0:
            continue
        # footprint_radius is None for some structures (flying, rich geysers)
        fp = structure.footprint_radius
        if fp is None:
            # Mineral patches and geysers may not have footprint_radius;
            # use radius as a fallback (minerals ~0.5, geysers ~1.0)
            fp = structure.radius
            if fp is None or fp < 0.5:
                continue

        cx, cy = int(structure.position.x), int(structure.position.y)
        # Mark all tiles within the footprint radius as unpathable
        radius_int = int(math.ceil(fp))
        for dx in range(-radius_int, radius_int + 1):
            for dy in range(-radius_int, radius_int + 1):
                ix, iy = cx + dx, cy + dy
                # Check if this tile is within the circular footprint
                if dx * dx + dy * dy <= fp * fp:
                    if 0 <= ix < grid_copy.shape[0] and 0 <= iy < grid_copy.shape[1]:
                        grid_copy[ix, iy] = np.inf

    return grid_copy


def detect_dynamic_choke(
    bot: "PiG_Bot",
    squad_position: Point2,
    enemy_center: Point2 | None = None,
) -> RefinedChoke | None:
    """Detect a dynamic choke created by buildings/resources around the squad.

    Builds a local grid that combines terrain with all building footprints
    (own + enemy + minerals + geysers), then scans in multiple directions from
    the squad position to find narrow passages that don't exist on the static
    terrain grid alone.

    This catches enemy wall-offs, cannon-rush gaps, bunker fortification gaps,
    and even chokes created by our own buildings or mineral lines — anything
    map_analyzer is blind to since it only sees static terrain.

    Scan strategy:
    - If enemy_center is provided: scan along the squad→enemy axis (primary)
      plus 4 perpendicular/radial directions around the squad.
    - If enemy_center is None: scan 8 radial directions around the squad.

    A dynamic choke is reported when the dynamic grid (terrain + buildings) is
    narrower than the clean terrain grid at the same point — i.e., buildings
    created a passage that doesn't exist on raw terrain.

    Perf note: O(B + S * R) where B = buildings in scan radius (typically <30),
    S = scan sample points (~10 per direction × 5 directions = ~50), and
    R = raycast steps per sample (≤30 for 2 dirs × 15 steps). Total: ~1,500
    cy_in_pathing_grid_ma calls. The grid copy is a numpy array copy — cheap.
    Called every DYNAMIC_CHOKE_SCAN_INTERVAL frames per squad, not every frame.

    Args:
        bot: Bot instance with structures, enemy_structures, mineral_field,
             vespene_geyser, and mediator for grid access.
        squad_position: Current squad center.
        enemy_center: Enemy army center of mass, or None if no enemies nearby.

    Returns:
        RefinedChoke if a dynamic choke is found near the squad, or None.
    """
    from bot.constants import (
        DYNAMIC_CHOKE_SCAN_RADIUS,
        DYNAMIC_CHOKE_MIN_WIDTH,
        DYNAMIC_CHOKE_MAX_WIDTH,
    )

    # Quick exit: no buildings of any kind nearby means no dynamic choke
    has_nearby_obstacles = (
        bot.enemy_structures.closer_than(DYNAMIC_CHOKE_SCAN_RADIUS, squad_position)
        or bot.structures.closer_than(DYNAMIC_CHOKE_SCAN_RADIUS, squad_position)
        or (hasattr(bot, "mineral_field") and bot.mineral_field.closer_than(DYNAMIC_CHOKE_SCAN_RADIUS, squad_position))
        or (hasattr(bot, "vespene_geyser") and bot.vespene_geyser.closer_than(DYNAMIC_CHOKE_SCAN_RADIUS, squad_position))
    )
    if not has_nearby_obstacles:
        return None

    # Build grids: clean terrain vs terrain + all building footprints
    terrain_grid = bot.mediator.get_map_data_object.get_pyastar_grid()
    dynamic_grid = _mark_buildings_unpathable(
        terrain_grid, bot, squad_position, DYNAMIC_CHOKE_SCAN_RADIUS
    )

    # Build scan directions:
    # - If we have an enemy center, prioritize the squad→enemy axis
    # - Always add radial directions so we catch chokes from any angle
    scan_directions: list[Point2] = []
    if enemy_center is not None:
        enemy_dir = _normalize(enemy_center - squad_position)
        scan_directions.append(enemy_dir)
        # Also scan perpendicular to the engagement axis (flank chokes)
        scan_directions.append(Point2((-enemy_dir.y, enemy_dir.x)))
        scan_directions.append(Point2((enemy_dir.y, -enemy_dir.x)))

    # Add 4-8 radial directions covering all angles
    for i in range(8):
        angle = i * (math.pi / 4)  # 0, 45, 90, 135, 180, 225, 270, 315 degrees
        d = Point2((math.cos(angle), math.sin(angle)))
        # Skip directions already covered by enemy-axis scan
        if enemy_center is not None:
            is_dup = any(
                abs(d.x - sd.x) < 0.1 and abs(d.y - sd.y) < 0.1
                for sd in scan_directions
            )
            if is_dup:
                continue
        scan_directions.append(d)

    best: RefinedChoke | None = None

    for direction in scan_directions:
        perp = Point2((-direction.y, direction.x))

        # March along this direction from the squad, sampling at each tile
        max_march = int(DYNAMIC_CHOKE_SCAN_RADIUS)
        for step in range(1, max_march + 1):
            sample = Point2((
                squad_position.x + direction.x * RAYCAST_STEP_SIZE * step,
                squad_position.y + direction.y * RAYCAST_STEP_SIZE * step,
            ))
            sample_int = Point2((int(sample.x), int(sample.y)))

            # Skip if the sample tile is unpathable on terrain (inside a wall/cliff)
            if not cy_in_pathing_grid_ma(terrain_grid, sample_int):
                break  # Hit a wall — stop marching this direction

            # Skip if the sample tile is unpathable on the dynamic grid
            # (we're inside a building footprint — not a useful choke point)
            if not cy_in_pathing_grid_ma(dynamic_grid, sample_int):
                break

            # Measure width on both grids at this sample point
            terrain_width, _ = _measure_choke_width(terrain_grid, sample_int, perp)
            dynamic_width, dynamic_center = _measure_choke_width(dynamic_grid, sample_int, perp)

            # Dynamic choke: buildings narrowed the passage compared to terrain
            if dynamic_width < terrain_width and DYNAMIC_CHOKE_MIN_WIDTH <= dynamic_width <= DYNAMIC_CHOKE_MAX_WIDTH:
                if best is None or dynamic_width < best.width:
                    best = RefinedChoke(
                        center=dynamic_center,
                        width=dynamic_width,
                        axis=direction,
                        perp=perp,
                    )
                # Found a narrow point in this direction — stop marching further
                # (the choke is between squad and the obstacle, not behind it)
                break

    return best
