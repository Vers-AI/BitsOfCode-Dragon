"""Strategy Belief — P(strategy=s | observations, race, game_time).

Purpose: Replace per-race boolean cheese detection with a unified probabilistic
         strategy classifier. Outputs P(strategy) over four Level-1 categories:
         cheese, all_in, timing_attack, macro. Each category can carry a Level-2
         build label (e.g., cheese→12_pool, cheese→cannon_rush).

Key Decisions: Three-layer evaluation: auto-TRUE guards → BN model → flat prior.
               Auto-TRUE guards are deterministic and override probabilities (P=1.0).
               ARES mediator booleans (marine_rush, proxy_zealot, etc.) map to
               Level-1 categories as deterministic guards with P→1.0 for their class.
               BN model slot accepts a trained pgmpy DiscreteBayesianNetwork;
               falls back to rule-based scoring when model file is missing.

Limitations: Level-2 labels for Terran and Protoss are limited to what ARES
             mediator detects. Full Level-2 taxonomy requires BN training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from bot.constants import (
    STRATEGY_CATEGORY_PRIOR,
    STRATEGY_LABELS,
    STRATEGY_TIMING_GUARDS,
    StrategyCategory,
)

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


@dataclass
class StrategyPrediction:
    """Read-only prediction snapshot consumed by downstream decisions.

    Not frozen because probs is a mutable dict. Consumers should treat
    this as read-only — use the p_cheese/p_all_in/p_timing/p_macro
    properties to read individual values.
    """
    probs: dict[StrategyCategory, float]
    label: StrategyCategory
    level2: str
    source: str
    game_time: float

    @property
    def p_cheese(self) -> float:
        return self.probs.get(StrategyCategory.CHEESE, 0.0)

    @property
    def p_all_in(self) -> float:
        return self.probs.get(StrategyCategory.ALL_IN, 0.0)

    @property
    def p_timing(self) -> float:
        return self.probs.get(StrategyCategory.TIMING_ATTACK, 0.0)

    @property
    def p_macro(self) -> float:
        return self.probs.get(StrategyCategory.MACRO, 0.0)

    @property
    def confidence(self) -> float:
        return max(self.probs.values())


class StrategyBelief:
    """Tracks P(strategy = s | observations) for the opponent's strategy.

    Updated once per frame. Produces a StrategyPrediction snapshot for
    BeliefState consumption.

    Three-layer evaluation:
      1. Auto-TRUE guards: deterministic, override everything (P→1.0)
      2. BN model: trained network, produces posterior probabilities
      3. Rule-based scoring: fallback when model unavailable
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._last_prediction: Optional[StrategyPrediction] = None
        self._load_model()

    def _load_model(self) -> None:
        model_path = Path("bot/models/strategy_belief_model.pkl")
        if model_path.exists():
            try:
                import joblib
                self._model = joblib.load(model_path)
                self._model_loaded = True
            except Exception as e:
                print(f"[StrategyBelief] Failed to load model: {e}")
                self._model_loaded = False

    def update(self, bot: "PiG_Bot", game_time: float) -> StrategyPrediction:
        """Produce a StrategyPrediction from current game observations.

        Args:
            bot: The bot instance for accessing enemy info and mediator.
            game_time: Current game time in seconds.

        Returns:
            StrategyPrediction with probs, label, level2, source, game_time.
        """
        # Layer 1: Auto-TRUE guards (deterministic overrides)
        guard_result = self._evaluate_guards(bot, game_time)
        if guard_result is not None:
            self._last_prediction = guard_result
            return guard_result

        # Layer 2: BN model (if available)
        if self._model_loaded and self._model is not None:
            model_result = self._evaluate_model(bot, game_time)
            if model_result is not None:
                self._last_prediction = model_result
                return model_result

        # Layer 3: Rule-based scoring
        rule_result = self._evaluate_rules(bot, game_time)
        self._last_prediction = rule_result
        return rule_result

    def _evaluate_guards(
        self, bot: "PiG_Bot", game_time: float
    ) -> Optional[StrategyPrediction]:
        """Layer 1: Deterministic auto-TRUE guards.

        These are observations that unambiguously indicate a strategy category.
        When a guard fires, P(category) → 1.0 and all others → 0.0.
        Returns None if no guard fires.
        """

        # --- Worker rush: ARES mediator ---
        if bot.mediator.get_enemy_worker_rushed and bot.game_state == 0:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="worker_rush",
                source="auto-TRUE:worker_rush",
                game_time=game_time,
            )

        # --- Cannon rush: detected by intel.py ---
        from bot.utilities.intel import get_enemy_cannon_rushed
        if get_enemy_cannon_rushed(bot):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="cannon_rush",
                source="auto-TRUE:cannon_rush",
                game_time=game_time,
            )

        # --- Zerg ling rush: auto-TRUE guards from cheese_detection ---
        guard = self._evaluate_zerg_guards(bot, game_time)
        if guard is not None:
            return guard

        # --- Terran proxies and rushes: ARES mediator ---
        terran_guard = self._evaluate_terran_guards(bot, game_time)
        if terran_guard is not None:
            return terran_guard

        # --- Protoss all-ins: ARES mediator ---
        protoss_guard = self._evaluate_protoss_guards(bot, game_time)
        if protoss_guard is not None:
            return protoss_guard

        # --- Zerg all-ins beyond ling rush: ARES mediator ---
        zerg_guard = self._evaluate_zerg_allin_guards(bot, game_time)
        if zerg_guard is not None:
            return zerg_guard

        return None

    def _evaluate_zerg_guards(
        self, bot: "PiG_Bot", game_time: float
    ) -> Optional[StrategyPrediction]:
        """Zerg-specific auto-TRUE guards: early lings, fast pool, speed contact."""
        enemy_race = bot.enemy_race.name

        # Only Zerg and Random (after seeing lings) have Zerg-specific guards
        if enemy_race not in ("Zerg", "Random"):
            return None
        # Random with unknown race: skip until we've seen Zerg units
        if enemy_race == "Random" and not hasattr(bot, "_first_ling_seen_time"):
            return None

        # Guard A: Ling seen at natural ≤ 1:45 → 12_pool
        if (
            hasattr(bot, "_first_ling_seen_time")
            and bot._first_ling_seen_time is not None
            and bot._first_ling_seen_time <= 105.0
        ):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="12_pool",
                source="auto-TRUE:ling_seen_early",
                game_time=game_time,
            )

        # Guard B: Slow-ling contact ≤ 2:00 → 12_pool
        if (
            hasattr(bot, "_first_ling_contact_nat_time")
            and bot._first_ling_contact_nat_time is not None
            and bot._first_ling_contact_nat_time <= 120.0
            and hasattr(bot, "_ling_has_speed")
            and not bot._ling_has_speed
        ):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="12_pool",
                source="auto-TRUE:slow_ling_contact",
                game_time=game_time,
            )

        # Guard C: Speed-ling contact ≤ 2:40 → speedling
        if (
            hasattr(bot, "_first_ling_contact_nat_time")
            and bot._first_ling_contact_nat_time is not None
            and bot._first_ling_contact_nat_time <= 160.0
            and hasattr(bot, "_ling_has_speed")
            and bot._ling_has_speed
        ):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="speedling",
                source="auto-TRUE:speed_ling_contact",
                game_time=game_time,
            )

        return None

    def _evaluate_terran_guards(
        self, bot: "PiG_Bot", game_time: float
    ) -> Optional[StrategyPrediction]:
        """Terran-specific auto-TRUE guards via ARES mediator booleans."""
        if bot.enemy_race.name != "Terran":
            return None

        # Marine rush (detected by ARES)
        if bot.mediator.get_enemy_marine_rush:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.7,
                    StrategyCategory.ALL_IN: 0.3,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="proxy_rax",
                source="auto-TRUE:marine_rush",
                game_time=game_time,
            )

        # Marauder rush — committed early aggression
        if bot.mediator.get_enemy_marauder_rush and game_time < 150.0:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.3,
                    StrategyCategory.ALL_IN: 0.5,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2="marauder_push",
                source="auto-TRUE:marauder_rush",
                game_time=game_time,
            )

        # Marine rush transitioned (went marine but may be transitioning)
        if bot.mediator.get_enemy_went_marine_rush:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.2,
                    StrategyCategory.ALL_IN: 0.3,
                    StrategyCategory.TIMING_ATTACK: 0.3,
                    StrategyCategory.MACRO: 0.2,
                },
                label=StrategyCategory.ALL_IN,
                level2="marine_rush_transition",
                source="auto-TRUE:went_marine_rush",
                game_time=game_time,
            )

        return None

    def _evaluate_protoss_guards(
        self, bot: "PiG_Bot", game_time: float
    ) -> Optional[StrategyPrediction]:
        """Protoss-specific auto-TRUE guards via ARES mediator booleans."""
        if bot.enemy_race.name != "Protoss":
            return None

        # Proxy zealot rush
        if bot.mediator.get_is_proxy_zealot:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="proxy_gateway",
                source="auto-TRUE:proxy_zealot",
                game_time=game_time,
            )

        # Four gate — committed all-in
        if bot.mediator.get_enemy_four_gate:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.1,
                    StrategyCategory.ALL_IN: 0.8,
                    StrategyCategory.TIMING_ATTACK: 0.1,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2="four_gate",
                source="auto-TRUE:four_gate",
                game_time=game_time,
            )

        return None

    def _evaluate_zerg_allin_guards(
        self, bot: "PiG_Bot", game_time: float
    ) -> Optional[StrategyPrediction]:
        """Zerg all-in guards beyond ling rush (roach, ravager)."""
        if bot.enemy_race.name not in ("Zerg", "Random"):
            return None

        # Roach rush
        if bot.mediator.get_enemy_roach_rushed:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.3,
                    StrategyCategory.ALL_IN: 0.5,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2="roach_rush",
                source="auto-TRUE:roach_rush",
                game_time=game_time,
            )

        # Ravager rush
        if bot.mediator.get_enemy_ravager_rush:
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.2,
                    StrategyCategory.ALL_IN: 0.6,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2="ravager_push",
                source="auto-TRUE:ravager_rush",
                game_time=game_time,
            )

        return None

    def _evaluate_model(
        self, bot: "PiG_Bot", game_time: float
    ) -> Optional[StrategyPrediction]:
        """Layer 2: BN model prediction.

        Builds feature vector from current observations and queries the
        trained pgmpy DiscreteBayesianNetwork.
        Returns None if model is unavailable or produces invalid output.
        """
        if self._model is None:
            return None

        try:
            features = self._build_feature_vector(bot, game_time)
            probs = self._model.predict_proba(features)
            # Model outputs dict of {category_name: probability}
            category_probs = {}
            for cat in StrategyCategory:
                category_probs[cat] = float(probs.get(cat.value, 0.0))

            # Normalize to sum to 1.0
            total = sum(category_probs.values())
            if total > 0:
                category_probs = {k: v / total for k, v in category_probs.items()}

            # Find best category
            best_cat = max(category_probs, key=category_probs.get)
            best_l2 = self._infer_level2(bot, best_cat, game_time)

            return StrategyPrediction(
                probs=category_probs,
                label=best_cat,
                level2=best_l2,
                source="BN",
                game_time=game_time,
            )
        except Exception as e:
            print(f"[StrategyBelief] BN prediction error: {e}")
            return None

    def _build_feature_vector(self, bot: "PiG_Bot", game_time: float) -> np.ndarray:
        """Build feature vector for the BN model from current observations.

        Feature order must match training script's ML_FEATURE_COLS.
        Missing values encoded as -1.
        """
        def _opt(attr: str, default: float = -1.0) -> float:
            return float(getattr(bot, attr, default)) if hasattr(bot, attr) and getattr(bot, attr) is not None else default

        features = np.array([[
            game_time,
            _opt("_pool_seen_time"),
            _opt("_enemy_nat_started_at"),
            _opt("_last_nat_scout_time"),
            1.0 if (hasattr(bot, "_nat_present_on_last_scout") and bot._nat_present_on_last_scout is True)
            else (0.0 if (hasattr(bot, "_nat_present_on_last_scout") and bot._nat_present_on_last_scout is False)
                  else -1.0),
            _opt("_extractor_seen_time"),
            _opt("_queen_started_time"),
            _opt("_first_ling_seen_time"),
            _opt("_first_ling_contact_nat_time"),
            _opt("_speed_research_time"),
            1.0 if (hasattr(bot, "_ling_has_speed") and bot._ling_has_speed) else 0.0,
            float(getattr(bot, "_gas_workers_count", 0)),
            float(
                {"Terran": 0, "Zerg": 1, "Protoss": 2}.get(
                    bot.enemy_race.name, 3
                )
            ),
        ]])
        return features

    def _evaluate_rules(
        self, bot: "PiG_Bot", game_time: float
    ) -> StrategyPrediction:
        """Layer 3: Rule-based scoring fallback.

        Uses the same scoring system as cheese_detection.py but outputs
        probabilities across all four categories instead of a binary flag.
        Produces soft probabilities from accumulated evidence.
        """
        probs = dict(STRATEGY_CATEGORY_PRIOR)

        # Zerg: accumulate evidence from timing signals
        if bot.enemy_race.name in ("Zerg", "Random"):
            self._score_zerg(bot, game_time, probs)

        # ARES soft signals: if a mediator flag recently flipped, add evidence
        self._score_ares_signals(bot, game_time, probs)

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        best_cat = max(probs, key=probs.get)
        best_l2 = self._infer_level2(bot, best_cat, game_time)

        return StrategyPrediction(
            probs=probs,
            label=best_cat,
            level2=best_l2,
            source="rules",
            game_time=game_time,
        )

    def _score_zerg(
        self, bot: "PiG_Bot", game_time: float, probs: dict
    ) -> None:
        """Accumulate Zerg-specific evidence into the probability dict.

        Evidence adds to cheese/all_in and subtracts from macro.
        """
        cheese_evidence = 0.0
        allin_evidence = 0.0

        # Pool timing signals
        pool_time = getattr(bot, "_pool_seen_time", None)
        if pool_time is not None:
            # 12-pool timing: 38-42s is strong cheese signal
            if 38.0 <= pool_time <= 42.0:
                cheese_evidence += 0.4
            # Speedling timing: 48-52s
            elif 48.0 <= pool_time <= 52.0:
                cheese_evidence += 0.3
            # Pool before 60s — some kind of early aggression
            elif pool_time < 60.0:
                cheese_evidence += 0.2

        # No natural expansion (confirmed after scouting)
        nat_scout_time = getattr(bot, "_last_nat_scout_time", None)
        nat_present = getattr(bot, "_nat_present_on_last_scout", None)
        if (
            nat_scout_time is not None
            and nat_scout_time >= 105.0
            and nat_present is False
            and getattr(bot, "_enemy_nat_started_at", None) is None
        ):
            cheese_evidence += 0.35

        # Baneling nest (seen but not yet in auto-TRUE range = timing/all_in signal)
        bane_time = getattr(bot, "_baneling_nest_seen_time", None)
        if bane_time is not None:
            if bane_time < 120.0:
                allin_evidence += 0.4  # Fast baneling = ling/bane all-in
            elif bane_time < 180.0:
                allin_evidence += 0.2  # Baneling for timing

        # Roach warren timing (tracked via composition belief structure priors)
        # If we see roaches from composition belief, add all_in signal

        # Apply evidence to probability dict
        probs[StrategyCategory.CHEESE] += cheese_evidence
        probs[StrategyCategory.ALL_IN] += allin_evidence
        # Reduce macro proportionally
        probs[StrategyCategory.MACRO] -= (cheese_evidence + allin_evidence) * 0.5
        probs[StrategyCategory.MACRO] = max(0.01, probs[StrategyCategory.MACRO])

    def _score_ares_signals(
        self, bot: "PiG_Bot", game_time: float, probs: dict
    ) -> None:
        """Soft evidence from ARES mediator booleans that aren't auto-TRUE guards.

        Some ARES flags are already handled by guards (worker_rush, four_gate, etc.).
        This handles signals that only shift probabilities, not override.
        """

        # Marine rush (Terran): already in guards but ARES may lag behind
        # This gives soft evidence before the guard fires
        if bot.mediator.get_enemy_marine_rush and bot.enemy_race.name == "Terran":
            probs[StrategyCategory.CHEESE] += 0.2

    def _infer_level2(
        self, bot: "PiG_Bot", category: StrategyCategory, game_time: float
    ) -> str:
        """Infer Level-2 build label from observations and category context.

        Falls back to the most generic label for the category if no
        specific evidence is available.
        """
        race = bot.enemy_race.name
        category_labels = STRATEGY_LABELS.get(category, {})

        # Try to match specific Level-2 labels based on observations
        if category == StrategyCategory.CHEESE:
            if race in ("Zerg", "Random"):
                pool_time = getattr(bot, "_pool_seen_time", None)
                if pool_time is not None and pool_time <= 42.0:
                    return "12_pool"
                if getattr(bot, "_ling_has_speed", False):
                    return "speedling"
                return "12_pool"  # Default Zerg cheese label
            if race == "Terran":
                if bot.mediator.get_enemy_marine_rush:
                    return "proxy_rax"
                return "proxy_rax"
            if race == "Protoss":
                from bot.utilities.intel import get_enemy_cannon_rushed
                if get_enemy_cannon_rushed(bot):
                    return "cannon_rush"
                if bot.mediator.get_is_proxy_zealot:
                    return "proxy_gateway"
                return "cannon_rush"

        if category == StrategyCategory.ALL_IN:
            if race in ("Zerg", "Random"):
                bane_time = getattr(bot, "_baneling_nest_seen_time", None)
                if bane_time is not None and bane_time < 120.0:
                    return "ling_bane_all_in"
                if bot.mediator.get_enemy_roach_rushed:
                    return "roach_ravager_push"
                return "roach_ravager_push"
            if race == "Protoss":
                if bot.mediator.get_enemy_four_gate:
                    return "four_gate"
                return "four_gate"
            if race == "Terran":
                return "cyclone_push"

        if category == StrategyCategory.TIMING_ATTACK:
            if race in ("Zerg", "Random"):
                return "ling_bane_timing"
            if race == "Protoss":
                return "stargate_timing"
            if race == "Terran":
                return "bio_timing"

        # Default: macro
        if race in ("Zerg", "Random"):
            return "standard_hatch_first"
        if race == "Protoss":
            return "three_base_macro"
        if race == "Terran":
            return "bio_macro"

        return "unknown"

    def snapshot(self) -> "StrategyBelief":
        """Create an independent copy for BeliefState consumption.

        Shares the loaded model (read-only) but copies the prediction
        reference. StrategyPrediction should be treated as read-only
        by consumers — do not mutate the probs dict.
        """
        cp = StrategyBelief.__new__(StrategyBelief)
        cp._model = self._model
        cp._model_loaded = self._model_loaded
        cp._last_prediction = self._last_prediction
        return cp

    @property
    def last_prediction(self) -> Optional[StrategyPrediction]:
        """Most recent prediction, or None if update() hasn't been called."""
        return self._last_prediction