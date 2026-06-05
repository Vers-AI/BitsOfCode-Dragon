"""Enemy intelligence package — observation, classification, and quality tracking.

Public API:
    - track_enemy_timings(bot)     — Record what we see and when (all races)
    - detect_cheese(bot)           — Classify enemy strategy (all races, all categories)
    - get_enemy_intel_quality(bot) — How fresh/reliable is our intel?
    - update_enemy_intel_tracking(bot) — Per-frame urgency updates
    - compute_rush_distance_tier(bot) — Rush distance for logging
"""

from bot.intel.enemy_timings import track_enemy_timings
from bot.intel.strategy_detect import (
    detect_cheese,
    compute_rush_distance_tier,
    get_ling_rush_signals,
    get_enemy_cannon_rushed,
    detect_worker_rush,
)
from bot.intel.intel_quality import (
    get_enemy_intel_quality,
    update_enemy_intel_tracking,
)

__all__ = [
    "track_enemy_timings",
    "detect_cheese",
    "compute_rush_distance_tier",
    "get_ling_rush_signals",
    "get_enemy_cannon_rushed",
    "detect_worker_rush",
    "get_enemy_intel_quality",
    "update_enemy_intel_tracking",
]