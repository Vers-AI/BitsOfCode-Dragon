"""Opponent Belief — Cross-game Dirichlet priors for known opponents.

Purpose: Remember opponent tendencies across games. Use Dirichlet alpha
         parameters to shift BN strategy predictions for known opponents.
         Unknown opponents get flat priors (same as today).

Key Decisions: Two data sources — runtime file (data/opponent_profiles.json)
               and baseline file (bot/models/opponent_priors.json). Runtime
               file takes precedence (more recent, ladder-accumulated).
               Flat prior [1,1,1,1] for unknown opponents.
               Race-specific priors keyed by (opponent_id, enemy_race).

Limitations: No per-frame I/O. One load at game start, one save at game end.
             Corrupt/missing files fall back to flat priors safely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bot.constants import StrategyCategory

if TYPE_CHECKING:
    from bot.bot import PiG_Bot

# File paths — runtime file takes precedence over baseline
_RUNTIME_PROFILES = Path("data/opponent_profiles.json")
_BASELINE_PRIORS = Path("bot/models/opponent_priors.json")

# Schema version for forward compatibility
_SCHEMA_VERSION = 1

# Maximum opponents to track (oldest-first eviction)
_MAX_OPPONENTS = 200

# Flat Dirichlet prior: uninformative starting point
FLAT_PRIOR: dict[StrategyCategory, float] = {
    StrategyCategory.CHEESE: 1.0,
    StrategyCategory.ALL_IN: 1.0,
    StrategyCategory.TIMING_ATTACK: 1.0,
    StrategyCategory.MACRO: 1.0,
}

# Category order for serialization (must be stable)
_CATEGORY_ORDER = [
    StrategyCategory.CHEESE,
    StrategyCategory.ALL_IN,
    StrategyCategory.TIMING_ATTACK,
    StrategyCategory.MACRO,
]


class OpponentBelief:
    """Tracks Dirichlet alpha parameters per (opponent_id, enemy_race).

    Usage:
        ob = OpponentBelief()
        ob.load()  # Call once at game start
        prior = ob.get_prior(opponent_id, enemy_race)
        # ... use prior to adjust BN output ...
        ob.update(opponent_id, enemy_race, predicted_category)  # Call once at game end
        ob.save()  # Call once at game end
    """

    def __init__(self):
        # Keyed by (opponent_id, enemy_race) → list of alpha values in _CATEGORY_ORDER
        self._profiles: dict[tuple[str, str], list[float]] = {}
        # Track insertion order for eviction
        self._insertion_order: list[tuple[str, str]] = []
        self._loaded = False

    def load(self) -> None:
        """Load opponent profiles from disk. Runtime file takes precedence.

        Tries data/opponent_profiles.json first (ladder-accumulated),
        then falls back to bot/models/opponent_priors.json (training baseline).
        Either file missing or corrupt → flat priors for all opponents.
        """
        self._profiles = {}
        self._insertion_order = []

        # Try runtime file first (ladder-accumulated, most recent)
        loaded = self._load_file(_RUNTIME_PROFILES)
        if not loaded:
            # Fall back to baseline (training-derived, baked into zip)
            loaded = self._load_file(_BASELINE_PRIORS)

        self._loaded = True

    def get_prior(
        self, opponent_id: Optional[str], enemy_race: str
    ) -> dict[StrategyCategory, float]:
        """Return Dirichlet alpha parameters as a normalized prior for this opponent.

        Args:
            opponent_id: Opponent identifier from ladder (None for local games).
            enemy_race: Enemy race name (e.g., "Zerg", "Terran", "Protoss").

        Returns:
            Dict mapping StrategyCategory to prior weight. Flat [1,1,1,1] if
            opponent unknown or no data available.
        """
        if not opponent_id or not self._loaded:
            return dict(FLAT_PRIOR)

        key = (opponent_id, enemy_race)
        if key not in self._profiles:
            return dict(FLAT_PRIOR)

        alphas = self._profiles[key]
        prior = {cat: alpha for cat, alpha in zip(_CATEGORY_ORDER, alphas)}

        # Debug: log when a known opponent's prior is applied
        total_games = sum(alphas) - 4.0  # Subtract baseline [1,1,1,1]
        if total_games >= 1:
            alpha_str = ", ".join(f"{cat.value}={a:.0f}" for cat, a in zip(_CATEGORY_ORDER, alphas))
            print(f"[OpponentBelief] Known opponent {opponent_id} vs {enemy_race}: "
                  f"{int(total_games)} games, priors: {alpha_str}")

        return prior

    def update(
        self,
        opponent_id: Optional[str],
        enemy_race: str,
        predicted_category: StrategyCategory,
    ) -> None:
        """Update alpha parameters after a game.

        Increments alpha[predicted_category] by 1 for this opponent.
        No-op if opponent_id is None (local game).

        Args:
            opponent_id: Opponent identifier from ladder (None for local games).
            enemy_race: Enemy race name.
            predicted_category: The strategy category the bot concluded.
        """
        if not opponent_id:
            return  # No opponent tracking in local games

        key = (opponent_id, enemy_race)
        if key not in self._profiles:
            # New opponent — start from flat prior
            self._profiles[key] = [1.0, 1.0, 1.0, 1.0]
            self._insertion_order.append(key)

        # Increment the predicted category's alpha
        cat_idx = _CATEGORY_ORDER.index(predicted_category)
        self._profiles[key][cat_idx] += 1.0

        # Evict oldest if over cap
        while len(self._profiles) > _MAX_OPPONENTS:
            oldest_key = self._insertion_order.pop(0)
            self._profiles.pop(oldest_key, None)

    def save(self) -> None:
        """Save opponent profiles to data/opponent_profiles.json.

        Creates the data/ directory if it doesn't exist.
        Writes atomically via temp file to avoid corruption on crash.
        """
        if not self._profiles:
            return  # Nothing to save

        _RUNTIME_PROFILES.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "schema_version": _SCHEMA_VERSION,
            "categories": [cat.value for cat in _CATEGORY_ORDER],
            "profiles": {},
        }

        for (opp_id, race), alphas in self._profiles.items():
            data["profiles"][f"{opp_id}:{race}"] = alphas

        # Write atomically — write to temp file then rename
        tmp_path = _RUNTIME_PROFILES.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.rename(_RUNTIME_PROFILES)
        except OSError as e:
            print(f"[OpponentBelief] Failed to save profiles: {e}")
            # Clean up temp file if rename failed
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_file(self, path: Path) -> bool:
        """Load profiles from a JSON file. Returns True if loaded successfully."""
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            # Validate schema version
            if data.get("schema_version") != _SCHEMA_VERSION:
                print(f"[OpponentBelief] Schema mismatch in {path}: "
                      f"expected {_SCHEMA_VERSION}, got {data.get('schema_version')}. "
                      "Using flat priors.")
                return False

            # Validate categories match
            file_categories = data.get("categories", [])
            expected = [cat.value for cat in _CATEGORY_ORDER]
            if file_categories != expected:
                print(f"[OpponentBelief] Category mismatch in {path}. Using flat priors.")
                return False

            # Load profiles
            profiles = data.get("profiles", {})
            for key_str, alphas in profiles.items():
                if ":" not in key_str or len(alphas) != 4:
                    continue
                opp_id, race = key_str.rsplit(":", 1)
                self._profiles[(opp_id, race)] = [float(a) for a in alphas]
                self._insertion_order.append((opp_id, race))

            print(f"[OpponentBelief] Loaded {len(self._profiles)} profiles from {path}")
            return True

        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"[OpponentBelief] Failed to load {path}: {e}. Using flat priors.")
            return False