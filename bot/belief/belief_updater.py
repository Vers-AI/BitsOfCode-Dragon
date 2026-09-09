"""Belief Updater — Produces new BeliefState each frame from observations.

Purpose: Single update() call ingests all current observations (cached army,
          visible structures, destroyed units) and produces a new BeliefState.

Key Decisions: Composition Belief always updates when enabled. Strategy Belief
               is optional (feature-gated). Opponent Belief loads cross-game
               priors at game start and saves at game end. Each update() returns
               a snapshot so BeliefState consumers never see mid-frame mutation.

Limitations: Update must complete within the frame budget (~0.5ms for all beliefs).
"""

from typing import TYPE_CHECKING

from bot.belief.composition_belief import CompositionBelief
from bot.belief.belief_state import BeliefState
from bot.belief.strategy_belief import StrategyBelief
from bot.belief.opponent_belief import OpponentBelief
from bot.belief.map_prior import MapPrior
from bot.constants import StrategyCategory

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


class BeliefUpdater:
    """Ingests observations and produces a new BeliefState each frame.

    Usage in bot.py:
        self.belief_updater = BeliefUpdater()
        # In on_step:
        self.belief_state = self.belief_updater.update(
            bot=self,
            cached_army=self.mediator.get_cached_enemy_army,
            visible_structures=self.enemy_structures,
            game_time=self.time,
        )
    """

    def __init__(self, enable_strategy: bool = True, enable_opponent: bool = True):
        self._composition = CompositionBelief()
        self._destroyed_tags: set[int] = set()
        self._strategy: StrategyBelief | None = None
        self._opponent: OpponentBelief | None = None
        self._map_prior: MapPrior | None = None
        if enable_strategy:
            self._strategy = StrategyBelief()
        if enable_opponent:
            self._opponent = OpponentBelief()
        # Map prior is always loaded (no feature flag — it's lightweight
        # and provides value even for unknown opponents on known maps)
        self._map_prior = MapPrior()

    def is_known_enemy_tag(self, tag: int) -> bool:
        """Check if a tag belongs to a known enemy unit (tracked or already destroyed).

        Used by on_unit_destroyed to avoid bloating _destroyed with friendly tags.
        """
        return self._composition.is_known_tag(tag)

    def record_destruction(self, tag: int) -> None:
        """Record that an enemy unit tag has been confirmed destroyed.

        Called from on_unit_destroyed. Ground truth — P(exists) → 0.0.
        """
        self._destroyed_tags.add(tag)

    def update(
        self,
        bot: "PiG_Bot",
        cached_army,
        visible_structures,
        game_time: float,
    ) -> BeliefState:
        """Produce a new BeliefState from current observations.

        Args:
            bot: The bot instance (for accessing _enemy_unit_last_seen and config)
            cached_army: Units from ARES UnitCacheManager
            visible_structures: Currently visible enemy structures
            game_time: Current game time in seconds

        Returns:
            A new BeliefState with updated beliefs.
        """
        # Get external last-seen tracking (populated in update_enemy_intel_tracking)
        unit_last_seen = getattr(bot, "_enemy_unit_last_seen", {})

        self._composition.update(
            cached_army=cached_army,
            visible_structures=visible_structures,
            game_time=game_time,
            destroyed_tags=self._destroyed_tags,
            unit_last_seen=unit_last_seen,
        )

        # Strategy belief update (if enabled)
        strategy_snapshot = None
        if self._strategy is not None:
            # Combine opponent prior and map prior into a single prior.
            # Both are Dirichlet alpha params; we multiply them together
            # (equivalent to combining two independent Dirichlet sources).
            # Map prior applies even for unknown opponents (uses map name only).
            # Opponent prior applies only for known opponents (uses opponent_id).
            map_prior = None
            if self._map_prior is not None:
                game_info = getattr(bot, 'game_info', None)
                map_name = game_info.map_name if game_info else None
                map_prior = self._map_prior.get_prior(map_name)

            opponent_prior = None
            if self._opponent is not None:
                opponent_id = getattr(bot, 'opponent_id', None)
                enemy_race = bot.enemy_race.name if hasattr(bot, 'enemy_race') else "Unknown"
                opponent_prior = self._opponent.get_prior(opponent_id, enemy_race)

            # Combine: multiply alphas (independent Dirichlet sources)
            # If only one is available, use it; if both are flat, pass None
            combined_prior = None
            if map_prior is not None and opponent_prior is not None:
                combined_prior = {
                    cat: map_prior.get(cat, 1.0) * opponent_prior.get(cat, 1.0)
                    for cat in StrategyCategory
                }
            elif map_prior is not None:
                combined_prior = map_prior
            elif opponent_prior is not None:
                combined_prior = opponent_prior

            prediction = self._strategy.update(bot, game_time, opponent_prior=combined_prior)
            strategy_snapshot = self._strategy.snapshot()

        # Snapshot so each BeliefState is an independent copy.
        # Uses shallow dict copies (shares Unit refs — they're read-only
        # game engine snapshots). Avoids deepcopy which fails on Unit objects
        # that hold references to the unpicklable bot/client.
        return BeliefState(
            composition=self._composition.snapshot(),
            strategy=strategy_snapshot,
        )

    def load_opponent(self) -> None:
        """Load opponent profiles and map priors from disk. Call once at game start.

        Passes the strategy model's epoch so OpponentBelief can invalidate
        runtime profiles accumulated under a previous model generation.
        """
        if self._opponent is not None:
            model_epoch = None
            if self._strategy is not None:
                model_epoch = self._strategy.model_epoch
            self._opponent.load(model_epoch=model_epoch)
        if self._map_prior is not None:
            self._map_prior.load()

    def save_opponent(self, bot, opponent_id: str | None, enemy_race: str) -> None:
        """Update and save opponent profiles from OBSERVED facts. Call once at game end.

        Classifies the game via classify_observed_game() (evidence-anchored —
        reads only what actually happened, never the model's prediction).
        Ambiguous games write nothing to the profile.
        """
        if self._opponent is not None and opponent_id is not None:
            from bot.intel import classify_observed_game
            observed = classify_observed_game(bot)
            if observed is not None:
                category, _source = observed
                self._opponent.update(opponent_id, enemy_race, category)
                self._opponent.save()