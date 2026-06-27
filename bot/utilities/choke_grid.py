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
from map_analyzer import MapData
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
        if choke.is_vision_blocker:
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
        elif choke.side_a is not None and choke.side_b is not None:
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
