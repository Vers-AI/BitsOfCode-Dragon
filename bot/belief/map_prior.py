# bot/belief/map_prior.py
"""Map-based strategy prior — P(strategy | map_name).

Purpose: Provide a Dirichlet prior for strategy classification based on
         historical results on a specific map. If the bot gets rushed more
         on certain maps, the prior shifts probability toward cheese/all_in
         before the BN model even sees any in-game evidence.

Key Decisions: Same Dirichlet-multinomial pattern as OpponentBelief.
               Loaded from bot/models/map_priors.json at game start.
               Flat prior [1,1,1,1] when map unknown or file missing.

Limitations: Only as good as the training data. Maps with few games will
             have weak priors (close to flat). No runtime updates —
             priors are rebuilt by the training script from API data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from bot.constants import StrategyCategory

_SCHEMA_VERSION = 1

# Category order for serialization (must match OpponentBelief._CATEGORY_ORDER)
_CATEGORY_ORDER = [
    StrategyCategory.CHEESE,
    StrategyCategory.ALL_IN,
    StrategyCategory.TIMING_ATTACK,
    StrategyCategory.MACRO,
]

FLAT_PRIOR: dict[StrategyCategory, float] = {
    StrategyCategory.CHEESE: 1.0,
    StrategyCategory.ALL_IN: 1.0,
    StrategyCategory.TIMING_ATTACK: 1.0,
    StrategyCategory.MACRO: 1.0,
}

_PRIORS_FILE = Path("bot/models/map_priors.json")


class MapPrior:
    """Loads map-based strategy priors from a JSON file.

    File format (produced by train_strategy_belief.py):
        {
            "schema_version": 1,
            "categories": ["cheese", "all_in", "timing_attack", "macro"],
            "maps": {
                "Pylon AIE": [cheese_alpha, all_in_alpha, timing_alpha, macro_alpha],
                ...
            }
        }
    """

    def __init__(self):
        self._maps: dict[str, list[float]] = {}
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load map priors from bot/models/map_priors.json."""
        self._maps = {}
        if not _PRIORS_FILE.exists():
            self._loaded = True
            return

        try:
            with open(_PRIORS_FILE) as f:
                data = json.load(f)

            if data.get("schema_version") != _SCHEMA_VERSION:
                print(f"[MapPrior] Schema mismatch: expected {_SCHEMA_VERSION}, "
                      f"got {data.get('schema_version')}. Using flat priors.")
                self._loaded = True
                return

            file_categories = data.get("categories", [])
            expected = [cat.value for cat in _CATEGORY_ORDER]
            if file_categories != expected:
                print(f"[MapPrior] Category mismatch. Using flat priors.")
                self._loaded = True
                return

            maps = data.get("maps", {})
            for map_name, alphas in maps.items():
                if len(alphas) == 4:
                    self._maps[map_name] = [float(a) for a in alphas]

            print(f"[MapPrior] Loaded {len(self._maps)} map priors from {_PRIORS_FILE}")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"[MapPrior] Failed to load {_PRIORS_FILE}: {e}. Using flat priors.")

        self._loaded = True

    def get_prior(self, map_name: Optional[str]) -> dict[StrategyCategory, float]:
        """Return Dirichlet alpha params as a normalized prior for this map.

        Args:
            map_name: Current map name (e.g., "Pylon AIE"). None if unknown.

        Returns:
            Dict mapping StrategyCategory to prior weight. Flat [1,1,1,1]
            if map unknown or no data available.
        """
        if not map_name or not self._loaded:
            return dict(FLAT_PRIOR)

        if map_name not in self._maps:
            return dict(FLAT_PRIOR)

        alphas = self._maps[map_name]
        return {cat: alpha for cat, alpha in zip(_CATEGORY_ORDER, alphas)}