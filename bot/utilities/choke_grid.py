"""Choke grid and narrow choke point computation — map geometry, not enemy intel.

Purpose: Precompute choke/ramp zones for O(1) formation skip detection and
         choke width lookup for engagement decisions. Call once in on_start.

Key Decisions: Wide-open "chokes" that map_analyzer classifies don't actually
              bottleneck armies, so we pre-filter at game start and only keep
              tiles from real narrow passages.

Limitations: Width measurement varies by choke type (RawChoke exact, MDRamp
            approximate, Polygon.width fallback).
"""

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
)

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


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

    Width measurement per choke type:
      - RawChoke: md_pl_choke.min_length (exact C-extension measurement)
      - MDRamp / VisionBlockerArea: side_a ↔ side_b distance
      - Fallback: Polygon.width (approximate, 0.5-1.5x real)
    """
    map_data: MapData = bot.mediator.get_map_data_object
    choke_width_map: dict[Point2, float] = {}

    for choke in map_data.map_chokes:
        width = None
        if hasattr(choke, "md_pl_choke") and choke.md_pl_choke is not None:
            width = choke.md_pl_choke.min_length
        elif choke.side_a is not None and choke.side_b is not None:
            sa = Point2(choke.side_a) if not isinstance(choke.side_a, Point2) else choke.side_a
            sb = Point2(choke.side_b) if not isinstance(choke.side_b, Point2) else choke.side_b
            width = cy_distance_to(sa, sb)

        if width is None:
            width = choke.width

        if width <= CHOKE_MAX_WIDTH:
            for point in choke.points:
                choke_width_map[point] = width

    return choke_width_map