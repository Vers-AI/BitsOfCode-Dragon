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
        self._model_categories: list[str] = []  # e.g. ['cheese', 'macro']
        self._model_loaded = False
        self._last_prediction: Optional[StrategyPrediction] = None
        self._load_model()

    def _load_model(self) -> None:
        model_path = Path("bot/models/strategy_belief_model.pkl")
        if model_path.exists():
            try:
                import joblib
                saved = joblib.load(model_path)
                # Saved model is a dict with keys: model, categories, feature_cols, ...
                if isinstance(saved, dict) and 'model' in saved:
                    self._model = saved['model']
                    self._model_categories = saved.get('categories', [])
                else:
                    # Legacy: saved directly as a model object
                    self._model = saved
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

        Delegates to bot.intel.detect_cheese() which consolidates all
        race-specific detectors (timing-based + ARES mediator booleans).
        When a guard fires, P(category) → 1.0 and all others → 0.0.
        Returns None if no guard fires.
        """
        from bot.intel import detect_cheese

        if not detect_cheese(bot):
            return None

        # Map the detected strategy label to a StrategyPrediction
        # detect_cheese() sets various bot._*_label attributes
        race = bot.enemy_race.name

        # Worker rush (all races)
        if getattr(bot, '_worker_rush_detected', False):
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

        # Zerg: ling rush labels
        cheese_label = getattr(bot, '_cheese_label', 'none')
        if cheese_label in ("12_pool", "speedling"):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2=cheese_label,
                source="auto-TRUE:ling_rush",
                game_time=game_time,
            )

        # Zerg: all-in labels (roach, ravager)
        zerg_allin_label = getattr(bot, '_zerg_allin_label', 'none')
        if zerg_allin_label in ("roach_rush", "ravager_push"):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.25,
                    StrategyCategory.ALL_IN: 0.55,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2=zerg_allin_label,
                source="auto-TRUE:zerg_allin",
                game_time=game_time,
            )

        # Terran: strategy labels
        terran_label = getattr(bot, '_terran_strategy_label', 'none')
        if terran_label == 'proxy_rax':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.7,
                    StrategyCategory.ALL_IN: 0.3,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="proxy_rax",
                source="auto-TRUE:proxy_rax",
                game_time=game_time,
            )
        if terran_label == 'bunker_rush':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.8,
                    StrategyCategory.ALL_IN: 0.2,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="bunker_rush",
                source="auto-TRUE:bunker_rush",
                game_time=game_time,
            )
        if terran_label == 'marauder_push':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.3,
                    StrategyCategory.ALL_IN: 0.5,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2="marauder_push",
                source="auto-TRUE:marauder_push",
                game_time=game_time,
            )
        if terran_label == 'all_in':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.2,
                    StrategyCategory.ALL_IN: 0.6,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2="marine_all_in",
                source="auto-TRUE:terran_allin",
                game_time=game_time,
            )

        # Protoss: strategy labels
        protoss_label = getattr(bot, '_protoss_strategy_label', 'none')
        if protoss_label == 'cannon_rush':
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
        if protoss_label == 'proxy_gate':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 1.0,
                    StrategyCategory.ALL_IN: 0.0,
                    StrategyCategory.TIMING_ATTACK: 0.0,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.CHEESE,
                level2="proxy_gateway",
                source="auto-TRUE:proxy_gate",
                game_time=game_time,
            )
        if protoss_label == 'four_gate':
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
        if protoss_label == 'all_in':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.2,
                    StrategyCategory.ALL_IN: 0.6,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2="all_in",
                source="auto-TRUE:protoss_allin",
                game_time=game_time,
            )

        # detect_cheese() returned True but label didn't match known patterns
        return StrategyPrediction(
            probs={
                StrategyCategory.CHEESE: 0.8,
                StrategyCategory.ALL_IN: 0.2,
                StrategyCategory.TIMING_ATTACK: 0.0,
                StrategyCategory.MACRO: 0.0,
            },
            label=StrategyCategory.CHEESE,
            level2="unknown",
            source="auto-TRUE:unknown",
            game_time=game_time,
        )

    def _evaluate_model(
        self, bot: "PiG_Bot", game_time: float
    ) -> Optional[StrategyPrediction]:
        """Layer 2: BN model prediction.

        Discretizes current observations into the same bins used during
        training, then queries the pgmpy DiscreteBayesianNetwork via
        predict_probability().
        Returns None if model is unavailable or produces invalid output.
        """
        if self._model is None:
            return None

        try:
            import pandas as pd
            evidence = self._build_bn_evidence(bot, game_time)
            result_df = self._model.predict_probability(evidence)

            # Result columns are like 'strategy_cheese', 'strategy_macro'
            # Extract probabilities for each known category
            category_probs = {}
            for cat in StrategyCategory:
                col_name = f"strategy_{cat.value}"
                if col_name in result_df.columns:
                    category_probs[cat] = float(result_df[col_name].iloc[0])
                else:
                    category_probs[cat] = 0.0

            # Normalize to sum to 1.0 (in case some categories are missing)
            total = sum(category_probs.values())
            if total > 0:
                category_probs = {k: v / total for k, v in category_probs.items()}

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

    def _build_bn_evidence(self, bot: "PiG_Bot", game_time: float):
        """Build discretized evidence DataFrame for the BN model.

        The BN has nodes: enemy_race, duration_bin, pool_bin → strategy.
        Discretization bins must match train_strategy_belief.py, but we also
        need to map to the model's actual trained states (which may be a subset
        of the full bin taxonomy if training data was sparse).
        """
        import pandas as pd

        # enemy_race: direct string (model knows: Protoss, Random, Terran, Zerg)
        race_map = {"Zerg": "Zerg", "Terran": "Terran", "Protoss": "Protoss"}
        enemy_race = race_map.get(bot.enemy_race.name, "Zerg")

        # duration_bin: game time in seconds
        # Training bins: 0-360=short, 360-720=medium, 720-1200=long, 1200+=very_long
        # Model states: long, medium, very_long (no "short" in training data)
        # Map "short" → "medium" (closest available, conservative)
        if game_time < 360:
            duration_bin = "medium"  # was "short", not in model
        elif game_time < 720:
            duration_bin = "medium"
        elif game_time < 1200:
            duration_bin = "long"
        else:
            duration_bin = "very_long"

        # pool_bin: pool seen time in seconds
        # Training bins: 0-42=very_early, 42-52=early, 52-80=mid, 80+=late
        # Model states: early, late, mid, very_early (no "unknown" in model)
        # Map "unknown" → "late" (conservative: assume macro if no pool seen)
        pool_time = getattr(bot, "_pool_seen_time", None)
        if pool_time is None:
            pool_bin = "late"  # was "unknown", not in model
        elif pool_time < 42:
            pool_bin = "very_early"
        elif pool_time < 52:
            pool_bin = "early"
        elif pool_time < 80:
            pool_bin = "mid"
        else:
            pool_bin = "late"

        return pd.DataFrame([{
            "enemy_race": enemy_race,
            "duration_bin": duration_bin,
            "pool_bin": pool_bin,
        }])

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

        # Terran: accumulate evidence from timing signals
        if bot.enemy_race.name == "Terran":
            self._score_terran(bot, game_time, probs)

        # Protoss: accumulate evidence from timing signals
        if bot.enemy_race.name == "Protoss":
            self._score_protoss(bot, game_time, probs)

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

    def _score_terran(
        self, bot: "PiG_Bot", game_time: float, probs: dict
    ) -> None:
        """Accumulate Terran-specific evidence into the probability dict.

        Consumes timing signals from bot.intel.enemy_timings.
        """
        cheese_evidence = 0.0
        allin_evidence = 0.0
        timing_evidence = 0.0

        # Proxy barracks near our base
        rax_proxy = getattr(bot, "_barracks_near_our_base", False)
        rax_time = getattr(bot, "_barracks_seen_time", None)
        rax_count = getattr(bot, "_barracks_count", 0)

        if rax_proxy:
            cheese_evidence += 0.4
        if rax_time is not None and rax_time < 65.0:
            cheese_evidence += 0.3
        if rax_count >= 2 and rax_proxy:
            cheese_evidence += 0.2

        # Bunker rush
        bunker_near = getattr(bot, "_bunker_near_base", False)
        bunker_time = getattr(bot, "_bunker_seen_time", None)
        if bunker_near:
            cheese_evidence += 0.3
        if bunker_time is not None and bunker_time < 120.0:
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
            if game_time < 180.0:
                allin_evidence += 0.35
            else:
                allin_evidence += 0.2

        # Early gas with workers
        gas_time = getattr(bot, "_enemy_gas_seen_time", None)
        gas_workers = getattr(bot, "_enemy_gas_workers_count", 0)
        if gas_time is not None and gas_time < 60.0 and gas_workers >= 2:
            allin_evidence += 0.2

        # Many rax, no natural = all-in commitment
        if rax_count >= 3 and getattr(bot, "_enemy_nat_started_at", None) is None and game_time < 180.0:
            allin_evidence += 0.3

        # Factory/starport timing = tech transition signals
        factory_time = getattr(bot, "_factory_seen_time", None)
        starport_time = getattr(bot, "_starport_seen_time", None)
        if factory_time is not None and game_time > 240.0:
            timing_evidence += 0.15
        if starport_time is not None and game_time > 300.0:
            timing_evidence += 0.1

        # Apply evidence
        probs[StrategyCategory.CHEESE] += cheese_evidence
        probs[StrategyCategory.ALL_IN] += allin_evidence
        probs[StrategyCategory.TIMING_ATTACK] += timing_evidence
        probs[StrategyCategory.MACRO] -= (cheese_evidence + allin_evidence + timing_evidence) * 0.5
        probs[StrategyCategory.MACRO] = max(0.01, probs[StrategyCategory.MACRO])

    def _score_protoss(
        self, bot: "PiG_Bot", game_time: float, probs: dict
    ) -> None:
        """Accumulate Protoss-specific evidence into the probability dict.

        Consumes timing signals from bot.intel.enemy_timings.
        """
        cheese_evidence = 0.0
        allin_evidence = 0.0
        timing_evidence = 0.0

        # Cannon rush: forge before gateway, forge early
        forge_time = getattr(bot, "_forge_seen_time", None)
        gw_time = getattr(bot, "_gateway_seen_time", None)
        cannon_time = getattr(bot, "_cannon_seen_time", None)

        if forge_time is not None and forge_time < 60.0:
            cheese_evidence += 0.3
        if cannon_time is not None and cannon_time < 150.0:
            cheese_evidence += 0.3
        if forge_time is not None and gw_time is not None and forge_time < gw_time:
            cheese_evidence += 0.3

        # Proxy gateways
        gw_proxy = getattr(bot, "_gateway_near_our_base", False)
        gw_count = getattr(bot, "_gateway_count", 0)
        if gw_proxy:
            cheese_evidence += 0.4
        if gw_time is not None and gw_time < 65.0:
            cheese_evidence += 0.3
        if gw_count >= 2 and gw_proxy:
            cheese_evidence += 0.2

        # Four-gate: 4+ gates, no natural, early gas
        if gw_count >= 4 and game_time < 300.0:
            allin_evidence += 0.25
        if gw_count >= 4 and getattr(bot, "_enemy_nat_started_at", None) is None and game_time > 180.0:
            allin_evidence += 0.3
        gas_time = getattr(bot, "_enemy_gas_seen_time", None)
        gas_workers = getattr(bot, "_enemy_gas_workers_count", 0)
        if gas_time is not None and gas_time < 80.0 and gas_workers >= 2:
            allin_evidence += 0.15
        # No stargate/robo by 4:00 with 4+ gates = four-gate commitment
        stargate_time = getattr(bot, "_stargate_seen_time", None)
        robo_time = getattr(bot, "_robotics_facility_seen_time", None)
        if gw_count >= 4 and game_time > 240.0 and stargate_time is None and robo_time is None:
            allin_evidence += 0.2

        # No natural expansion (confirmed after scouting)
        nat_scout_time = getattr(bot, "_last_nat_scout_time", None)
        nat_present = getattr(bot, "_nat_present_on_last_scout", None)
        if (
            nat_scout_time is not None
            and nat_scout_time >= 105.0
            and nat_present is False
            and getattr(bot, "_enemy_nat_started_at", None) is None
        ):
            if game_time < 180.0:
                allin_evidence += 0.3
            else:
                allin_evidence += 0.15

        # Tech structure timing = timing attack signals
        core_time = getattr(bot, "_cyber_core_seen_time", None)
        if core_time is not None and game_time > 240.0:
            timing_evidence += 0.1
        if stargate_time is not None and game_time > 300.0:
            timing_evidence += 0.15
        if robo_time is not None and game_time > 300.0:
            timing_evidence += 0.1

        # Apply evidence
        probs[StrategyCategory.CHEESE] += cheese_evidence
        probs[StrategyCategory.ALL_IN] += allin_evidence
        probs[StrategyCategory.TIMING_ATTACK] += timing_evidence
        probs[StrategyCategory.MACRO] -= (cheese_evidence + allin_evidence + timing_evidence) * 0.5
        probs[StrategyCategory.MACRO] = max(0.01, probs[StrategyCategory.MACRO])

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
                # Prefer the label already set by detect_cheese()
                cheese_label = getattr(bot, "_cheese_label", "none")
                if cheese_label in ("12_pool", "speedling"):
                    return cheese_label
                pool_time = getattr(bot, "_pool_seen_time", None)
                if pool_time is not None and pool_time <= 42.0:
                    return "12_pool"
                if getattr(bot, "_ling_has_speed", False):
                    return "speedling"
                return "12_pool"
            if race == "Terran":
                terran_label = getattr(bot, "_terran_strategy_label", "none")
                if terran_label in ("proxy_rax", "bunker_rush", "marauder_push"):
                    return terran_label
                if getattr(bot, "_barracks_near_our_base", False):
                    return "proxy_rax"
                if getattr(bot, "_bunker_near_base", False):
                    return "bunker_rush"
                return "proxy_rax"
            if race == "Protoss":
                protoss_label = getattr(bot, "_protoss_strategy_label", "none")
                if protoss_label in ("cannon_rush", "proxy_gate"):
                    return protoss_label
                if getattr(bot, "_gateway_near_our_base", False):
                    return "proxy_gateway"
                if getattr(bot, "_forge_seen_time", None) is not None:
                    return "cannon_rush"
                return "cannon_rush"

        if category == StrategyCategory.ALL_IN:
            if race in ("Zerg", "Random"):
                zerg_allin = getattr(bot, "_zerg_allin_label", "none")
                if zerg_allin in ("roach_rush", "ravager_push"):
                    return zerg_allin
                bane_time = getattr(bot, "_baneling_nest_seen_time", None)
                if bane_time is not None and bane_time < 120.0:
                    return "ling_bane_all_in"
                return "roach_ravager_push"
            if race == "Protoss":
                protoss_label = getattr(bot, "_protoss_strategy_label", "none")
                if protoss_label == "four_gate":
                    return "four_gate"
                if getattr(bot, "_gateway_count", 0) >= 4:
                    return "four_gate"
                return "four_gate"
            if race == "Terran":
                terran_label = getattr(bot, "_terran_strategy_label", "none")
                if terran_label == "all_in":
                    return "marine_all_in"
                if getattr(bot, "_barracks_count", 0) >= 3 and getattr(bot, "_enemy_nat_started_at", None) is None:
                    return "marine_all_in"
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
        cp._model_categories = self._model_categories
        cp._model_loaded = self._model_loaded
        cp._last_prediction = self._last_prediction
        return cp

    @property
    def last_prediction(self) -> Optional[StrategyPrediction]:
        """Most recent prediction, or None if update() hasn't been called."""
        return self._last_prediction