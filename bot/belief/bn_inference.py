"""Lightweight Bayesian Network inference via numpy array lookup.

Purpose: Replace pgmpy DiscreteBayesianNetwork inference with a direct numpy
         conditional probability table lookup. The BN structure is naive Bayes
         (all evidence variables are parents of the strategy node), so inference
         reduces to a single array index when all evidence is observed.

Key Decisions: Loads a .npz file containing the CPD array and state names.
               Falls back to .pkl (pgmpy) if .npz is missing, for backward
               compatibility during the transition period.
               Fully dynamic — reads parent variable order, state names, and CPD
               shape from the model file. No hardcoded shapes or expected state lists.
               Supports schema v1 (7 vars) and v2 (15 vars with position + timing).

Limitations: Only supports naive Bayes structure (all parents → single child).
             If the model gains intermediate nodes, this must be replaced with
             proper inference (pgmpy or similar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from bot.constants import StrategyCategory

# Schema versions — v1 = 7 vars, v2 = 15 vars (position + timing features)
_SCHEMA_V1 = 1
_SCHEMA_V2 = 2

# Parent variable names per schema (for logging/debugging only — actual order
# comes from the model file's node_order field)
_PARENT_VARS_V1 = ["bases_bin", "duration_bin", "enemy_race", "factory_bin", "gateway_bin", "pool_bin", "rax_bin"]

_PARENT_VARS_V2 = [
    "bases_bin", "duration_bin", "enemy_race", "factory_bin", "gateway_bin",
    "pool_bin", "rax_bin",
    "rax_near_base", "gw_near_base", "cannon_near_base", "bunker_near_base",
    "rax_timing", "pool_timing", "gw_timing", "nat_timing",
]


class BNInference:
    """Numpy-based BN inference for the strategy classifier.

    Fully dynamic: reads parent variable order, state names, and CPD shape
    from the model file. No hardcoded shapes or expected state lists.
    Auto-detects schema version from the .npz schema_version field.
    """

    def __init__(self):
        self._cpd: Optional[np.ndarray] = None
        self._state_names: dict[str, list[str]] = {}
        self._loaded: bool = False
        self._source: str = ""  # "npz" or "pkl" for telemetry
        self._schema_version: int = 0
        self._parent_vars: list[str] = []  # Actual order from model file
        self.load()  # Auto-load on init

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def source(self) -> str:
        return self._source

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def is_v2(self) -> bool:
        """True if the loaded model uses schema v2 (15 parent variables)."""
        return self._schema_version == _SCHEMA_V2

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
            version = int(data.get("schema_version", 0))
            if version not in (_SCHEMA_V1, _SCHEMA_V2):
                print(f"[BNInference] Unknown schema version: {version}")
                return False

            cpd = data["cpd"]
            # Reconstruct state_names from individual arrays (no pickle needed)
            state_names = {}
            for var in data["node_order"].astype(str):
                state_names[var] = list(data[f"states_{var}"].astype(str))
            categories = list(data["categories"].astype(str))

            # Determine parent variable order from the CPD shape.
            # Axis 0 = strategy, remaining axes = parent vars in node_order
            # (excluding strategy itself).
            # pgmpy stores CPD variables as [child, parent1, parent2, ...]
            # so node_order from the export is [strategy, parent1, ...]
            all_vars = list(data["node_order"].astype(str))
            parent_vars = [v for v in all_vars if v != "strategy"]

            # Validate: CPD should have 1 + len(parent_vars) dimensions
            if cpd.ndim != 1 + len(parent_vars):
                print(f"[BNInference] CPD ndim {cpd.ndim} != 1 + {len(parent_vars)} parent vars")
                return False

            # Validate: strategy axis should have 4 states
            if cpd.shape[0] != 4:
                print(f"[BNInference] Strategy axis has {cpd.shape[0]} states, expected 4")
                return False

            # Validate: each parent axis size matches state_names count
            for i, var in enumerate(parent_vars):
                expected = len(state_names.get(var, []))
                actual = cpd.shape[1 + i]
                if expected != actual:
                    print(f"[BNInference] Axis mismatch for '{var}': "
                          f"states={expected}, cpd_axis={actual}")
                    return False

            self._cpd = cpd
            self._state_names = state_names
            self._loaded = True
            self._source = "npz"
            self._schema_version = version
            self._parent_vars = parent_vars
            print(f"[BNInference] Loaded .npz model (schema v{version}, "
                  f"{len(parent_vars)} parent vars, shape={cpd.shape})")
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

            # Detect schema from number of parent variables
            n_parents = len(state_names) - 1  # minus strategy node
            if n_parents == 7:
                version = _SCHEMA_V1
            elif n_parents == 15:
                version = _SCHEMA_V2
            else:
                print(f"[BNInference] Unexpected parent count from .pkl: {n_parents}")
                return False

            # Parent variable order from the CPD (excludes strategy which is first)
            parent_vars = list(strategy_cpd.variables[1:])

            self._cpd = cpd_array
            self._state_names = state_names
            self._loaded = True
            self._source = "pkl"
            self._schema_version = version
            self._parent_vars = parent_vars
            print(f"[BNInference] Loaded .pkl model (schema v{version}, "
                  f"{n_parents} parent vars, shape={cpd_array.shape})")
            return True

        except Exception as e:
            print(f"[BNInference] Failed to load .pkl: {e}")
            return False

    def predict(self, **evidence) -> dict[StrategyCategory, float]:
        """Look up P(strategy | evidence) from the CPD array.

        Accepts keyword arguments matching the parent variable names.
        For schema v1: enemy_race, duration_bin, pool_bin, rax_bin, gateway_bin,
                       bases_bin, factory_bin
        For schema v2: all v1 args + rax_near_base, gw_near_base, cannon_near_base,
                       bunker_near_base, rax_timing, pool_timing, gw_timing, nat_timing

        If an evidence value isn't in the model's state names (e.g., the model
        only has "unknown" for a position feature but the bot passes "yes"),
        the closest available state is used as a fallback.

        Returns:
            Dict mapping StrategyCategory → probability (sums to 1.0).
        """
        if not self._loaded or self._cpd is None:
            return {cat: 0.25 for cat in StrategyCategory}

        # Build index tuple for array lookup
        # CPD axis order: strategy, then parent vars in _parent_vars order
        try:
            indices = []
            for var in self._parent_vars:
                val = evidence.get(var)
                if val is None:
                    # Use sensible defaults for missing evidence
                    if var in ("rax_near_base", "gw_near_base", "cannon_near_base", "bunker_near_base"):
                        val = "unknown"
                    elif var in ("rax_timing", "pool_timing", "gw_timing", "nat_timing"):
                        val = "none"
                    else:
                        print(f"[BNInference] Missing evidence for '{var}'")
                        return {cat: 0.25 for cat in StrategyCategory}

                states = self._state_names.get(var, [])
                if val in states:
                    indices.append(states.index(val))
                elif states:
                    # Value not in model states — use first state as fallback.
                    # This happens when the model was trained with limited data
                    # (e.g., only "unknown" for position features) but the runtime
                    # bot has real values. The model will improve when retrained
                    # with position data from the API.
                    indices.append(0)
                else:
                    print(f"[BNInference] No states for variable '{var}'")
                    return {cat: 0.25 for cat in StrategyCategory}
        except Exception as e:
            print(f"[BNInference] Evidence indexing error: {e}")
            return {cat: 0.25 for cat in StrategyCategory}

        # Single array lookup: P(strategy | all evidence)
        probs = self._cpd[(slice(None),) + tuple(indices)]

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

    def get_evidence_dict(self, **evidence) -> dict[str, str]:
        """Return evidence as a plain dict (for telemetry/logging).

        Filters to only the variables relevant to the loaded schema.
        """
        result = {}
        for var in self._parent_vars:
            if var in evidence:
                result[var] = evidence[var]
        return result