"""Strategy Belief — P(strategy=s | observations, race, game_time).

Purpose: Replace per-race boolean cheese detection with a unified probabilistic
          strategy classifier. Outputs P(strategy) over four Level-1 categories:
          cheese, all_in, timing_attack, macro. Each category can carry a Level-2
          build label (e.g., cheese→12_pool, cheese→cannon_rush).

Key Decisions: Three-layer evaluation: sklearn model → auto-TRUE guards → rule-based.
                sklearn GradientBoostingClassifier is the primary classifier
                (joblib .pkl, Phase B hard cutover — BN removed).
                Auto-TRUE guards are deterministic overrides (P=1.0) when model
                is unavailable. Rules are the last-resort fallback.

Limitations: Level-2 labels for Terran and Protoss are limited to what ARES
              mediator detects. Full Level-2 taxonomy requires training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json

import numpy as np

from bot.constants import (
    STRATEGY_CATEGORY_PRIOR,
    STRATEGY_LABELS,
    STRATEGY_TIMING_GUARDS,
    StrategyCategory,
)
from bot.belief.sklearn_inference import SklearnInference
from cython_extensions import cy_distance_to

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
    evidence: Optional[dict] = None  # BN evidence dict for telemetry

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

    Evaluation order:
      1. sklearn model: GradientBoostingClassifier via joblib .pkl (primary)
      2. Auto-TRUE guards: deterministic, only when model is missing (fallback)
      3. Rule-based scoring: soft probabilities from accumulated evidence (last resort)
    """

    def __init__(self):
        self._model = SklearnInference()
        self._model_loaded = self._model.is_loaded
        self._last_prediction: Optional[StrategyPrediction] = None
        self._category_prior = self._load_category_prior()

    @staticmethod
    def _load_category_prior() -> dict[StrategyCategory, float]:
        """Load data-driven marginal P(strategy) from replay-derived JSON.

        Falls back to the hardcoded STRATEGY_CATEGORY_PRIOR in constants.py
        when the file is missing or invalid. The JSON file is produced by
        train_strategy_belief.py from API replay ground-truth labels.
        """
        path = Path("bot/models/strategy_category_prior.json")
        if not path.exists():
            return dict(STRATEGY_CATEGORY_PRIOR)

        try:
            with open(path) as f:
                data = json.load(f)
            probs = data.get("probs", {})
            result = {}
            for cat in StrategyCategory:
                result[cat] = float(probs.get(cat.value, STRATEGY_CATEGORY_PRIOR[cat]))
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            print(f"[StrategyBelief] Loaded data-driven category prior from {path}")
            return result
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            print(f"[StrategyBelief] Failed to load category prior: {e}. "
                  f"Using hardcoded fallback.")
            return dict(STRATEGY_CATEGORY_PRIOR)

    def update(
        self,
        bot: "PiG_Bot",
        game_time: float,
        opponent_prior: Optional[dict[StrategyCategory, float]] = None,
    ) -> StrategyPrediction:
        """Produce a StrategyPrediction from current game observations.

        Evaluation order: sklearn model → guards (fallback) → rules (last resort).
        The sklearn model is the primary classifier. Guards only fire when the model
        is unavailable, providing deterministic coverage for known patterns.
        Rules are the final fallback when neither model nor guards activate.

        Args:
            bot: The bot instance for accessing enemy info and mediator.
            game_time: Current game time in seconds.
            opponent_prior: Optional Dirichlet alpha params from OpponentBelief.
                If provided, model output is multiplied by this prior then normalized.

        Returns:
            StrategyPrediction with probs, label, level2, source, game_time.
        """
        # Primary: sklearn model (always runs when available)
        if self._model_loaded:
            model_result = self._evaluate_model(bot, game_time, opponent_prior)
            if model_result is not None:
                self._last_prediction = model_result
                self._send_strategy_chat(bot, model_result, "SKL")
                return model_result

        # Fallback: Auto-TRUE guards (only when model is missing)
        guard_result = self._evaluate_guards(bot, game_time)
        if guard_result is not None:
            self._last_prediction = guard_result
            self._send_strategy_chat(bot, guard_result, "auto-TRUE")
            return guard_result

        # Last resort: Rule-based scoring
        rule_result = self._evaluate_rules(bot, game_time)
        self._last_prediction = rule_result
        self._send_strategy_chat(bot, rule_result, "rules")
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

        # Auto-TRUE guards already set _strategy_chat_pending via detect_cheese()
        # No additional chat needed here

        # Map the detected strategy label to a StrategyPrediction
        # detect_cheese() sets various bot._*_label attributes
        race = bot.enemy_race.name

        # Worker rush (all races) — check detection flag set by detect_cheese()
        # NOTE: Don't check reaction_manager here — it hasn't processed the
        # prediction yet at this point in the frame (belief update runs before
        # reaction_manager.update()). The detection flag is set in the same
        # frame by detect_cheese() → detect_worker_rush().
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

        # Zerg: all-in labels (roach, ravager, roach_ravager_push, etc.)
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
        if zerg_allin_label in ("roach_ravager_push", "one_base_all_in", "two_base_all_in"):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.1,
                    StrategyCategory.ALL_IN: 0.7,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2=zerg_allin_label,
                source="auto-TRUE:zerg_allin",
                game_time=game_time,
            )
        # Zerg timing labels
        zerg_timing_label = getattr(bot, '_zerg_timing_label', 'none')
        if zerg_timing_label == 'roach_timing':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.0,
                    StrategyCategory.ALL_IN: 0.2,
                    StrategyCategory.TIMING_ATTACK: 0.7,
                    StrategyCategory.MACRO: 0.1,
                },
                label=StrategyCategory.TIMING_ATTACK,
                level2="roach_timing",
                source="auto-TRUE:zerg_timing",
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
        if terran_label in ('all_in', 'marine_rush', 'one_base_all_in', 'two_base_all_in'):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.2,
                    StrategyCategory.ALL_IN: 0.6,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2=terran_label,
                source="auto-TRUE:terran_allin",
                game_time=game_time,
            )
        if terran_label in ('bio_timing', 'tank_timing', 'widow_mine_drop'):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.0,
                    StrategyCategory.ALL_IN: 0.15,
                    StrategyCategory.TIMING_ATTACK: 0.75,
                    StrategyCategory.MACRO: 0.1,
                },
                label=StrategyCategory.TIMING_ATTACK,
                level2=terran_label,
                source="auto-TRUE:terran_timing",
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
        if protoss_label in ('four_gate', 'six_gate'):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.1,
                    StrategyCategory.ALL_IN: 0.8,
                    StrategyCategory.TIMING_ATTACK: 0.1,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2=protoss_label,
                source="auto-TRUE:protoss_allin",
                game_time=game_time,
            )
        if protoss_label in ('all_in', 'two_base_colossus', 'two_base_all_in', 'one_base_all_in'):
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.2,
                    StrategyCategory.ALL_IN: 0.6,
                    StrategyCategory.TIMING_ATTACK: 0.2,
                    StrategyCategory.MACRO: 0.0,
                },
                label=StrategyCategory.ALL_IN,
                level2=protoss_label,
                source="auto-TRUE:protoss_allin",
                game_time=game_time,
            )
        if protoss_label == 'stargate_timing':
            return StrategyPrediction(
                probs={
                    StrategyCategory.CHEESE: 0.0,
                    StrategyCategory.ALL_IN: 0.15,
                    StrategyCategory.TIMING_ATTACK: 0.75,
                    StrategyCategory.MACRO: 0.1,
                },
                label=StrategyCategory.TIMING_ATTACK,
                level2="stargate_timing",
                source="auto-TRUE:protoss_timing",
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

    def _send_strategy_chat(
        self, bot: "PiG_Bot", prediction: StrategyPrediction, source: str
    ) -> None:
        """Send in-game chat for non-macro strategy predictions.

        Only sends once per category change to avoid spamming every frame.
        Uses bot._strategy_chat_pending so it integrates with the existing
        chat-send logic in bot.py on_step().
        """
        from bot.constants import StrategyCategory

        # Track last announced category to avoid repeat messages
        if not hasattr(bot, '_strategy_chat_last_category'):
            bot._strategy_chat_last_category = None

        label = prediction.label
        if label == StrategyCategory.MACRO:
            return  # No chat for standard macro games

        # Only send if category changed since last announcement
        if bot._strategy_chat_last_category == label:
            return
        bot._strategy_chat_last_category = label

        # Format: [BN] cheese 70% or [rules] all_in
        top_prob = max(prediction.probs.values())
        pct = int(top_prob * 100)
        level2 = prediction.level2 or ""
        if level2 and level2 != "unknown":
            msg = f"[{source}] {label.value} {pct}% ({level2})"
        else:
            msg = f"[{source}] {label.value} {pct}%"

        bot._strategy_chat_pending = msg
        print(f"{bot.time_formatted}: Strategy classified ({source}): "
              f"{label.value} {pct}% ({level2})")

    def _evaluate_model(
        self, bot: "PiG_Bot", game_time: float,
        opponent_prior: Optional[dict[StrategyCategory, float]] = None,
    ) -> Optional[StrategyPrediction]:
        """Layer 1: sklearn model prediction via joblib-loaded classifier.

        Discretizes current observations into the same bins used during
        training, then predicts P(strategy | evidence) via predict_proba.
        If an opponent prior is provided, the model output is multiplied by
        the prior then renormalized.

        Args:
            bot: The bot instance for accessing enemy info and mediator.
            game_time: Current game time in seconds.
            opponent_prior: Optional Dirichlet alpha params from OpponentBelief.
                Multiplied with model output, then renormalized.

        Returns None if model is unavailable or produces invalid output.
        """
        if not self._model_loaded:
            return None

        try:
            evidence = self._build_evidence(bot, game_time)
            category_probs = self._model.predict(**evidence)

            # Apply opponent prior if available: P(adjusted) = P(BN) * alpha, then normalize
            # This is a Dirichlet-multinomial posterior where BN provides the likelihood
            # and opponent history provides the prior.
            if opponent_prior is not None:
                adjusted = {}
                for cat in StrategyCategory:
                    adjusted[cat] = category_probs.get(cat, 0.0) * opponent_prior.get(cat, 1.0)
                adj_total = sum(adjusted.values())
                if adj_total > 0:
                    category_probs = {k: v / adj_total for k, v in adjusted.items()}

            best_cat = max(category_probs, key=category_probs.get)
            best_l2 = self._infer_level2(bot, best_cat, game_time)

            # Mark source as SKL+OPP if opponent prior was applied
            source = "SKL+OPP" if opponent_prior is not None else "SKL"

            return StrategyPrediction(
                probs=category_probs,
                label=best_cat,
                level2=best_l2,
                source=source,
                game_time=game_time,
                evidence=evidence,
            )
        except Exception as e:
            print(f"[StrategyBelief] sklearn prediction error: {e}")
            return None

    def _build_evidence(self, bot: "PiG_Bot", game_time: float) -> dict[str, str]:
        """Build discretized evidence dict for the sklearn model.

        Schema v1: 7 variables (enemy_race, duration_bin, pool_bin, rax_bin,
                   gateway_bin, bases_bin, factory_bin)
        Schema v2: 15 variables (v1 + 4 position + 4 timing)
        Schema v3: 17 variables (v2 + gas_timing, ling_timing — the
                   disambiguators that separate "safe pool → drone" from
                   "pool → ling flood")

        Discretization bins must match train_strategy_belief.py.
        Out-of-domain values (e.g., "unknown", "short") are mapped to
        the closest valid model state.
        """
        # enemy_race: direct string (model knows: Protoss, Random, Terran, Zerg)
        race_map = {"Zerg": "Zerg", "Terran": "Terran", "Protoss": "Protoss"}
        enemy_race = race_map.get(bot.enemy_race.name, "Zerg")

        # duration_bin: game time in seconds
        # Training bins: 0-360=short, 360-720=medium, 720-1200=long, 1200+=very_long
        # Model states: long, medium, very_long (no "short" in training data)
        # Map "short" → "medium" (closest available, conservative)
        if game_time < 360:
            duration_bin = "medium"  # "short" not in model
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
            pool_bin = "late"  # "unknown" not in model
        elif pool_time < 42:
            pool_bin = "very_early"
        elif pool_time < 52:
            pool_bin = "early"
        elif pool_time < 80:
            pool_bin = "mid"
        else:
            pool_bin = "late"

        # rax_bin: barracks count
        # Training bins: 0=none, 1-2=few, 3+=many
        rax_count = getattr(bot, "_barracks_count", 0)
        if rax_count <= 0:
            rax_bin = "none"
        elif rax_count <= 2:
            rax_bin = "few"
        else:
            rax_bin = "many"

        # gateway_bin: gateway + warpgate count
        # Training bins: 0=none, 1-3=few, 4+=many
        gw_count = getattr(bot, "_gateway_count", 0) + getattr(bot, "_warpgate_count", 0)
        if gw_count <= 0:
            gateway_bin = "none"
        elif gw_count <= 3:
            gateway_bin = "few"
        else:
            gateway_bin = "many"

        # bases_bin: enemy base count
        # Training bins: 1=one, 2=two, 3+=three_plus
        # Count enemy townhalls from scouted structures
        from sc2.ids.unit_typeid import UnitTypeId
        townhall_types = {
            UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
            UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS,
            UnitTypeId.NEXUS,
        }
        bases = sum(1 for s in bot.enemy_structures if s.type_id in townhall_types)
        if bases <= 1:
            bases_bin = "one"
        elif bases == 2:
            bases_bin = "two"
        else:
            bases_bin = "three_plus"

        # factory_bin: factory seen?
        # Training bins: yes/no
        factory_count = getattr(bot, "_factory_count", 0)
        factory_bin = "yes" if factory_count > 0 else "no"

        # === Schema v2 features (Step 1: position) ===

        # rax_near_base: proxy barracks near our base? (2 states: yes/not_yes)
        rax_near = getattr(bot, "_barracks_near_our_base", False)
        rax_near_base = "yes" if rax_near else "not_yes"

        # gw_near_base: proxy gateway near our base? (2 states: yes/not_yes)
        gw_near = getattr(bot, "_gateway_near_our_base", False)
        gw_near_base = "yes" if gw_near else "not_yes"

        # cannon_near_base: enemy cannon near our base? (2 states: yes/not_yes)
        # TRAIN/SSERVE ALIGNMENT: must use the same geometric check the training
        # rows used (game_report._cannon_near_base_int: visible PHOTONCANNON
        # within 25 of main/nat). The OLD code read _cannon_rush_active — a
        # reaction flag that's False until AFTER a cheese reaction fires, i.e.
        # circular: the model could never see 'yes' before predicting cheese.
        cannon_near_base = "not_yes"
        try:
            from sc2.ids.unit_typeid import UnitTypeId as _UT
            cannons = [s for s in bot.enemy_structures
                       if s.type_id == _UT.PHOTONCANNON]
            if cannons:
                our_nat = bot.mediator.get_own_nat
                near = any(
                    cy_distance_to(c.position, bot.start_location) < 25.0
                    or cy_distance_to(c.position, our_nat) < 25.0
                    for c in cannons
                )
                if near:
                    cannon_near_base = "yes"
        except Exception:
            pass  # enemy_structures unavailable mid-frame — stays not_yes

        # bunker_near_base: enemy bunker near our base? (2 states: yes/not_yes)
        bunker_near = getattr(bot, "_bunker_near_base", False)
        bunker_near_base = "yes" if bunker_near else "not_yes"

        # === Schema v2 features (Step 2: timing) ===

        # rax_timing: when was first barracks seen? (3 states: none/early/standard)
        # Collapsed from 5 bins to 3: early=<45s (proxy/cheese), standard=>=45s, none=unseen
        rax_time = getattr(bot, "_barracks_seen_time", None)
        if rax_time is None:
            rax_timing = "none"
        elif rax_time < 45:
            rax_timing = "early"
        else:
            rax_timing = "standard"

        # pool_timing: when was first spawning pool seen? (3 states: none/early/standard)
        # early=<40s (12-pool/speedling), standard=>=40s, none=unseen
        if pool_time is None:
            pool_timing = "none"
        elif pool_time < 40:
            pool_timing = "early"
        else:
            pool_timing = "standard"

        # gw_timing: when was first gateway seen? (3 states: none/early/standard)
        # early=<40s (proxy), standard=>=40s, none=unseen
        gw_time = getattr(bot, "_gateway_seen_time", None)
        if gw_time is None:
            gw_timing = "none"
        elif gw_time < 40:
            gw_timing = "early"
        else:
            gw_timing = "standard"

        # nat_timing: when did enemy natural start? (3 states: none/early/standard)
        # early=<120s (standard macro), standard=>=120s (late/all-in), none=no expansion
        nat_time = getattr(bot, "_enemy_nat_started_at", None)
        if nat_time is None:
            nat_timing = "none"
        elif nat_time < 120:
            nat_timing = "early"
        else:
            nat_timing = "standard"

        # Schema v3 — disambiguation features (bins must match
        # train_strategy_belief.py discretize_features):
        # gas_timing: all_in takes gas ~38-42s, cheese ~52-60s
        gas_time = getattr(bot, "_extractor_seen_time", None)
        if gas_time is None:
            gas_timing = "none"
        elif gas_time < 50:
            gas_timing = "early"
        elif gas_time < 90:
            gas_timing = "mid"
        else:
            gas_timing = "late"

        # ling_timing: rush lings get scouted MID (120-160s, attacking when
        # spotted) — macro's defensive lings show earlier or later
        ling_time = getattr(bot, "_first_ling_seen_time", None)
        if ling_time is None:
            ling_timing = "none"
        elif ling_time < 120:
            ling_timing = "early"
        elif ling_time < 160:
            ling_timing = "mid"
        else:
            ling_timing = "late"

        return {
            # Schema v1 (7 vars)
            "enemy_race": enemy_race,
            "duration_bin": duration_bin,
            "pool_bin": pool_bin,
            "rax_bin": rax_bin,
            "gateway_bin": gateway_bin,
            "bases_bin": bases_bin,
            "factory_bin": factory_bin,
            # Schema v2 — Step 1: position features (4 vars)
            "rax_near_base": rax_near_base,
            "gw_near_base": gw_near_base,
            "cannon_near_base": cannon_near_base,
            "bunker_near_base": bunker_near_base,
            # Schema v2 — Step 2: timing features (4 vars)
            "rax_timing": rax_timing,
            "pool_timing": pool_timing,
            "gw_timing": gw_timing,
            "nat_timing": nat_timing,
            # Schema v3 — disambiguation (2 vars)
            "gas_timing": gas_timing,
            "ling_timing": ling_timing,
        }

    def _evaluate_rules(
        self, bot: "PiG_Bot", game_time: float
    ) -> StrategyPrediction:
        """Layer 3: Rule-based scoring fallback.

        Uses the same scoring system as cheese_detection.py but outputs
        probabilities across all four categories instead of a binary flag.
        Produces soft probabilities from accumulated evidence.
        """
        probs = dict(self._category_prior)

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
        cp._model_loaded = self._model_loaded
        cp._last_prediction = self._last_prediction
        return cp

    @property
    def last_prediction(self) -> Optional[StrategyPrediction]:
        """Most recent prediction, or None if update() hasn't been called."""
        return self._last_prediction

    @property
    def model_epoch(self) -> Optional[int]:
        """The loaded model's generation stamp (None if model not loaded).
        BeliefUpdater passes this to OpponentBelief for retrain invalidation."""
        return self._model.model_epoch if self._model is not None else None