"""Export pgmpy BN model to lightweight .npz format for runtime inference.

Purpose: Convert the trained pgmpy DiscreteBayesianNetwork (.pkl) into a
         numpy .npz file that BNInference can load without pgmpy/joblib.
         Run this after retraining the model with train_strategy_belief.py.

Usage:
    poetry run python scripts/export_bn_model.py

Output:
    bot/models/strategy_belief_model.npz

The .npz file contains:
    - cpd: numpy array of shape (4, 3, 4, 4, 2, 3, 4, 3)
           P(strategy | bases_bin, duration_bin, enemy_race, factory_bin,
             gateway_bin, pool_bin, rax_bin)
    - state_names: dict mapping variable name → list of state strings
    - categories: list of strategy category names
    - schema_version: int for format validation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PKL_PATH = Path("bot/models/strategy_belief_model.pkl")
NPZ_PATH = Path("bot/models/strategy_belief_model.npz")

SCHEMA_VERSION = 2  # v2 = 15 parent variables (position + timing features)


def export_model(pkl_path: Path = PKL_PATH, npz_path: Path = NPZ_PATH) -> bool:
    """Load .pkl model and export to .npz format.

    Returns True if export succeeded, False otherwise.
    """
    if not pkl_path.exists():
        print(f"[export] .pkl not found at {pkl_path}")
        return False

    try:
        import joblib
        saved = joblib.load(pkl_path)
    except ImportError:
        print("[export] joblib not installed — run: poetry install")
        return False
    except Exception as e:
        print(f"[export] Failed to load .pkl: {e}")
        return False

    try:
        # Extract model
        if isinstance(saved, dict) and "model" in saved:
            model = saved["model"]
            categories = saved.get("categories", [])
        else:
            model = saved
            categories = []

        # Find the strategy CPD
        strategy_cpd = None
        for cpd in model.cpds:
            if cpd.variable == "strategy":
                strategy_cpd = cpd
                break

        if strategy_cpd is None:
            print("[export] No strategy CPD found in model")
            return False

        # Extract CPD array and state names
        cpd_array = strategy_cpd.values.copy()
        state_names = {}
        for var in strategy_cpd.variables:
            state_names[var] = list(strategy_cpd.state_names[var])

        if not categories:
            categories = list(state_names.get("strategy", []))

        print(f"[export] CPD shape: {cpd_array.shape}")
        print(f"[export] Variables: {list(state_names.keys())}")
        print(f"[export] Categories: {categories}")

        # Detect schema from number of parent variables
        n_parents = len(state_names) - 1  # minus strategy node
        if n_parents == 7:
            schema_ver = 1
        elif n_parents == 15:
            schema_ver = 2
        else:
            print(f"[export] Unexpected parent count: {n_parents}")
            return False

        # No hardcoded shape validation — pgmpy only creates states for values
        # present in training data, so the CPD shape varies. BNInference reads
        # the actual shape from the file at load time.
        print(f"[export] Schema v{schema_ver}, {n_parents} parent vars, CPD shape: {cpd_array.shape}")

        # Verify CPD sums to 1.0 along strategy axis
        sums = cpd_array.sum(axis=0)
        if not np.allclose(sums, 1.0, atol=1e-6):
            bad = np.abs(sums - 1.0).max()
            print(f"[export] WARNING: CPD sums deviate from 1.0 by up to {bad:.6f}")

        # Save as .npz — state_names stored as individual arrays (not object arrays)
        # to avoid allow_pickle issues at load time
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "cpd": cpd_array,
            "categories": np.array(categories, dtype="U20"),
            "schema_version": np.array(schema_ver, dtype=np.int32),
        }
        # Store state_names as individual arrays keyed by variable name
        for var, states in state_names.items():
            save_dict[f"states_{var}"] = np.array(states, dtype="U20")
        # Store variable order so we can reconstruct at load time
        save_dict["node_order"] = np.array(list(state_names.keys()), dtype="U20")

        np.savez(npz_path, **save_dict)

        print(f"[export] Saved .npz to {npz_path}")
        print(f"[export] File size: {npz_path.stat().st_size / 1024:.1f} KB")

        # Verify round-trip
        verify_result = verify_export(npz_path, cpd_array, state_names, categories)
        return verify_result

    except Exception as e:
        print(f"[export] Failed to export: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_export(
    npz_path: Path,
    original_cpd: np.ndarray,
    original_state_names: dict,
    original_categories: list,
) -> bool:
    """Verify the exported .npz loads correctly and matches original data."""
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        print(f"[verify] Failed to load .npz: {e}")
        return False

    try:
        version = int(data["schema_version"])
        # Accept either schema version — the export detects which one from the model
        if version not in (1, 2):
            print(f"[verify] Unknown schema version: {version}")
            return False

        loaded_cpd = data["cpd"]
        loaded_categories = list(data["categories"].astype(str))

        # Reconstruct state_names from individual arrays
        loaded_state_names = {}
        for var in data["node_order"].astype(str):
            loaded_state_names[var] = list(data[f"states_{var}"].astype(str))

        # Check CPD matches
        if not np.allclose(loaded_cpd, original_cpd):
            print("[verify] CPD array mismatch!")
            return False

        # Check state names match
        for var, states in original_state_names.items():
            if var not in loaded_state_names:
                print(f"[verify] Missing variable '{var}' in loaded state_names")
                return False
            if loaded_state_names[var] != states:
                print(f"[verify] State name mismatch for '{var}'")
                return False

        # Check categories match
        if loaded_categories != original_categories:
            print(f"[verify] Categories mismatch: {loaded_categories} != {original_categories}")
            return False

        print("[verify] ✅ Export verified — .npz matches .pkl")
        return True

    except Exception as e:
        print(f"[verify] Failed: {e}")
        return False


if __name__ == "__main__":
    success = export_model()
    if not success:
        print("\nExport failed. Make sure the .pkl model exists and pgmpy is installed.")
        exit(1)