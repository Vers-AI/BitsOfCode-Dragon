"""Lightweight Bayesian Network inference via numpy array lookup.

Purpose: Replace pgmpy DiscreteBayesianNetwork inference with a direct numpy
         conditional probability table lookup. The BN structure is naive Bayes
         (all evidence variables are parents of the strategy node), so inference
         reduces to a single array index when all evidence is observed.

Key Decisions: Loads a .npz file containing the CPD array and state names.
               Falls back to .pkl (pgmpy) if .npz is missing, for backward
               compatibility during the transition period.
               Validates state names at load time to catch model/version mismatches.

Limitations: Only supports naive Bayes structure (all parents → single child).
             If the model gains intermediate nodes, this must be replaced with
             proper inference (pgmpy or similar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from bot.constants import StrategyCategory

# Schema version for forward compatibility — bump when .npz format changes
_SCHEMA_VERSION = 1

# Expected variable order in the CPD array (must match training export)
# Axis 0 = strategy, remaining axes = parent variables in this order
_PARENT_VARS = ["bases_bin", "duration_bin", "enemy_race", "factory_bin", "gateway_bin", "pool_bin", "rax_bin"]

# Expected state names for each variable (must match training bins)
_EXPECTED_STATES = {
    "strategy": ["all_in", "cheese", "macro", "timing_attack"],
    "enemy_race": ["Protoss", "Random", "Terran", "Zerg"],
    "duration_bin": ["long", "medium", "short", "very_long"],
    "pool_bin": ["early", "late", "mid", "very_early"],
    "rax_bin": ["few", "many", "none"],
    "gateway_bin": ["few", "many", "none"],
    "bases_bin": ["one", "three_plus", "two"],
    "factory_bin": ["no", "yes"],
}


class BNInference:
    """Numpy-based BN inference for the strategy classifier.

    Loads a pre-exported .npz file containing:
      - cpd: numpy array of shape (4, 3, 4, 4, 2, 3, 4, 3)
             Axis 0 = strategy categories, remaining axes = parent variables
      - state_names: dict mapping variable name → list of state strings
      - schema_version: int for format validation

    Inference is a single array lookup: cpd[:, bases_idx, duration_idx, ...].
    """

    def __init__(self):
        self._cpd: Optional[np.ndarray] = None
        self._state_names: dict[str, list[str]] = {}
        self._loaded: bool = False
        self._source: str = ""  # "npz" or "pkl" for telemetry
        self.load()  # Auto-load on init

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def source(self) -> str:
        return self._source

    def load(self) -> bool:
        """Load the BN model. Tries .npz first, then .pkl fallback.

        Returns True if a model was loaded successfully.
        """
        if self._try_load_npz():
            return True
        if self._try_load_pkl():
            return True
        return False

    def _try_load_npz(self) -> bool:
        """Load from .npz file (preferred — no pgmpy dependency)."""
        npz_path = Path("bot/models/strategy_belief_model.npz")
        if not npz_path.exists():
            return False

        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"[BNInference] Failed to load .npz: {e}")
            return False

        try:
            # Validate schema version
            version = int(data.get("schema_version", 0))
            if version != _SCHEMA_VERSION:
                print(f"[BNInference] Schema mismatch: expected {_SCHEMA_VERSION}, got {version}")
                return False

            cpd = data["cpd"]
            # Reconstruct state_names from individual arrays (no pickle needed)
            state_names = {}
            for var in data["node_order"].astype(str):
                state_names[var] = list(data[f"states_{var}"].astype(str))
            categories = list(data["categories"].astype(str))

            # Validate CPD shape
            expected_shape = (4, 3, 4, 4, 2, 3, 4, 3)
            if cpd.shape != expected_shape:
                print(f"[BNInference] CPD shape mismatch: expected {expected_shape}, got {cpd.shape}")
                return False

            # Validate state names match expected
            for var, expected_states in _EXPECTED_STATES.items():
                if var not in state_names:
                    print(f"[BNInference] Missing variable '{var}' in state_names")
                    return False
                if state_names[var] != expected_states:
                    print(f"[BNInference] State name mismatch for '{var}': "
                          f"expected {expected_states}, got {state_names[var]}")
                    return False

            self._cpd = cpd
            self._state_names = state_names
            self._loaded = True
            self._source = "npz"
            print(f"[BNInference] Loaded .npz model (schema v{version})")
            return True

        except Exception as e:
            print(f"[BNInference] Error parsing .npz: {e}")
            return False

    def _try_load_pkl(self) -> bool:
        """Fallback: load from .pkl file (requires pgmpy + joblib).

        Extracts the CPD array and state names from the pgmpy model so
        subsequent inference calls don't need pgmpy.
        """
        pkl_path = Path("bot/models/strategy_belief_model.pkl")
        if not pkl_path.exists():
            return False

        try:
            import joblib
            saved = joblib.load(pkl_path)

            if isinstance(saved, dict) and "model" in saved:
                model = saved["model"]
            else:
                model = saved

            # Extract CPD for the strategy node
            strategy_cpd = None
            for cpd in model.cpds:
                if cpd.variable == "strategy":
                    strategy_cpd = cpd
                    break

            if strategy_cpd is None:
                print("[BNInference] No strategy CPD found in .pkl model")
                return False

            # Extract the numpy array and state names
            cpd_array = strategy_cpd.values.copy()
            state_names = {}
            for var in strategy_cpd.variables:
                state_names[var] = list(strategy_cpd.state_names[var])

            # Validate shape
            expected_shape = (4, 3, 4, 4, 2, 3, 4, 3)
            if cpd_array.shape != expected_shape:
                print(f"[BNInference] CPD shape mismatch from .pkl: "
                      f"expected {expected_shape}, got {cpd_array.shape}")
                return False

            self._cpd = cpd_array
            self._state_names = state_names
            self._loaded = True
            self._source = "pkl"
            print("[BNInference] Loaded .pkl model (extracted CPD for numpy inference)")
            return True

        except Exception as e:
            print(f"[BNInference] Failed to load .pkl: {e}")
            return False

    def predict(
        self,
        enemy_race: str,
        duration_bin: str,
        pool_bin: str,
        rax_bin: str,
        gateway_bin: str,
        bases_bin: str,
        factory_bin: str,
    ) -> dict[StrategyCategory, float]:
        """Look up P(strategy | evidence) from the CPD array.

        Maps string evidence values to array indices, performs a single
        array lookup, and returns a probability dict.

        Args:
            enemy_race: One of "Protoss", "Random", "Terran", "Zerg"
            duration_bin: One of "long", "medium", "short", "very_long"
            pool_bin: One of "early", "late", "mid", "very_early"
            rax_bin: One of "few", "many", "none"
            gateway_bin: One of "few", "many", "none"
            bases_bin: One of "one", "three_plus", "two"
            factory_bin: One of "no", "yes"

        Returns:
            Dict mapping StrategyCategory → probability (sums to 1.0).
        """
        if not self._loaded or self._cpd is None:
            # Uniform prior — shouldn't happen but safe fallback
            return {cat: 0.25 for cat in StrategyCategory}

        # Map evidence strings to indices
        # CPD axis order: strategy, bases_bin, duration_bin, enemy_race,
        #                  factory_bin, gateway_bin, pool_bin, rax_bin
        try:
            idx_bases = self._state_names["bases_bin"].index(bases_bin)
            idx_duration = self._state_names["duration_bin"].index(duration_bin)
            idx_race = self._state_names["enemy_race"].index(enemy_race)
            idx_factory = self._state_names["factory_bin"].index(factory_bin)
            idx_gateway = self._state_names["gateway_bin"].index(gateway_bin)
            idx_pool = self._state_names["pool_bin"].index(pool_bin)
            idx_rax = self._state_names["rax_bin"].index(rax_bin)
        except ValueError as e:
            print(f"[BNInference] Invalid evidence value: {e}")
            return {cat: 0.25 for cat in StrategyCategory}

        # Single array lookup: P(strategy | all evidence)
        probs = self._cpd[:, idx_bases, idx_duration, idx_race,
                          idx_factory, idx_gateway, idx_pool, idx_rax]

        # Map array indices to StrategyCategory
        strategy_states = self._state_names["strategy"]
        result = {}
        for cat in StrategyCategory:
            if cat.value in strategy_states:
                idx = strategy_states.index(cat.value)
                result[cat] = float(probs[idx])
            else:
                result[cat] = 0.0

        # Normalize (should already sum to 1.0, but guard against float errors)
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def get_evidence_dict(
        self,
        enemy_race: str,
        duration_bin: str,
        pool_bin: str,
        rax_bin: str,
        gateway_bin: str,
        bases_bin: str,
        factory_bin: str,
    ) -> dict[str, str]:
        """Return evidence as a plain dict (for telemetry/logging).

        Replaces the pandas DataFrame that _build_bn_evidence used to return.
        """
        return {
            "enemy_race": enemy_race,
            "duration_bin": duration_bin,
            "pool_bin": pool_bin,
            "rax_bin": rax_bin,
            "gateway_bin": gateway_bin,
            "bases_bin": bases_bin,
            "factory_bin": factory_bin,
        }