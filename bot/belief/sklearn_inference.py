"""sklearn inference for the strategy classifier (Phase B hard cutover).

Purpose: Replace the numpy BN CPD lookup with a joblib-loaded sklearn
         GradientBoostingClassifier. Same interface as the old BNInference:
         load() on init, is_loaded, predict(**evidence) -> dict[StrategyCategory, float].

Key Decisions: Artifact is a single .pkl dict {model, feature_cols, classes}
               where model is a sklearn Pipeline (OneHotEncoder + GBC) —
               one file, one load, no pgmpy, no .npz export step.
               Evidence keys are the 15 discretized feature names; values are
               bin strings (same as the BN used). One-hot encoding happens
               inside the loaded pipeline.
               Load-time sanity check: one synthetic prediction on load —
               a corrupt pickle or version-skewed sklearn marks the model
               not-loaded and StrategyBelief falls back to guards/rules.

Limitations: predict() needs the evidence values to be bin strings the
             encoder knows; unknown categories are one-hot ignored
             (handle_unknown="ignore") — the prediction silently loses that
             feature's signal rather than crashing. Runtime evidence builder
             must keep bin names in sync with training discretization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from joblib import load as joblib_load

from bot.constants import StrategyCategory

MODEL_PATH = Path("bot/models/strategy_model_sklearn.pkl")


class SklearnInference:
    """joblib-loaded sklearn Pipeline with BNInference-compatible interface."""

    def __init__(self):
        self._model: Optional[object] = None
        self._feature_cols: list[str] = []
        self._classes: list[str] = []
        self._loaded: bool = False
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the sklearn artifact and run a sanity prediction.

        On any failure (missing file, corrupt pickle, schema drift) the model
        stays not-loaded — StrategyBelief's guards/rules fallback takes over.
        """
        if not MODEL_PATH.exists():
            print(f"[SklearnInference] No model at {MODEL_PATH} — guards/rules will be used")
            return

        try:
            artifact = joblib_load(MODEL_PATH)
            self._model = artifact["model"]
            self._feature_cols = list(artifact["feature_cols"])
            self._classes = [str(c) for c in artifact["classes"]]

            # Sanity check: corrupt pickle or incompatible sklearn must fail
            # HERE at load time, not mid-game on the first predict call
            synthetic = {col: "unknown" for col in self._feature_cols}
            self._raw_predict(synthetic)

            self._loaded = True
            print(f"[SklearnInference] Loaded {MODEL_PATH.name} "
                  f"({len(self._feature_cols)} features, {len(self._classes)} classes)")
        except Exception as e:
            self._loaded = False
            print(f"[SklearnInference] Failed to load model: {e} — guards/rules will be used")

    def _build_X(self, evidence: dict[str, str]):
        """Build a single-row DataFrame with training column order/names.

        The OneHotEncoder was fitted on a DataFrame — passing one here keeps
        feature-name alignment guaranteed (a raw numpy array would rely on
        positional order matching, which is fragile).
        """
        import pandas as pd
        return pd.DataFrame(
            [[str(evidence.get(col, "unknown")) for col in self._feature_cols]],
            columns=self._feature_cols,
        )

    def _raw_predict(self, evidence: dict[str, str]) -> str:
        """Run the pipeline on an evidence dict, return the predicted class string."""
        return str(self._model.predict(self._build_X(evidence))[0])

    def predict(self, **evidence) -> dict[StrategyCategory, float]:
        """Predict P(strategy | evidence) via predict_proba.

        Accepts keyword args matching the training feature names (the same
        15 discretized variables the BN used). Unknown keys are ignored;
        missing keys default to "unknown".

        Returns:
            Dict mapping StrategyCategory → probability (sums to 1.0).
            Falls back to uniform on any error.
        """
        if not self._loaded:
            return {cat: 0.25 for cat in StrategyCategory}

        try:
            proba = self._model.predict_proba(self._build_X(evidence))[0]
            class_map = {cls: float(p) for cls, p in zip(self._classes, proba)}

            result = {}
            for cat in StrategyCategory:
                result[cat] = class_map.get(cat.value, 0.0)

            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            else:
                result = {cat: 0.25 for cat in StrategyCategory}
            return result
        except Exception as e:
            print(f"[SklearnInference] Prediction error: {e}")
            return {cat: 0.25 for cat in StrategyCategory}