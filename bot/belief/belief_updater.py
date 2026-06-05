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
        if enable_strategy:
            self._strategy = StrategyBelief()
        if enable_opponent:
            self._opponent = OpponentBelief()

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
            # Get opponent prior for this game (None if opponent belief disabled)
            opponent_prior = None
            if self._opponent is not None:
                opponent_id = getattr(bot, 'opponent_id', None)
                enemy_race = bot.enemy_race.name if hasattr(bot, 'enemy_race') else "Unknown"
                opponent_prior = self._opponent.get_prior(opponent_id, enemy_race)

            prediction = self._strategy.update(bot, game_time, opponent_prior=opponent_prior)
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
        """Load opponent profiles from disk. Call once at game start."""
        if self._opponent is not None:
            self._opponent.load()

    def save_opponent(self, opponent_id: str | None, enemy_race: str,
                      predicted_category) -> None:
        """Update and save opponent profiles. Call once at game end.

        Args:
            opponent_id: Opponent identifier from ladder (None for local games).
            enemy_race: Enemy race name.
            predicted_category: The StrategyCategory the bot concluded.
        """
        if self._opponent is not None and opponent_id is not None:
            from bot.constants import StrategyCategory
            if isinstance(predicted_category, StrategyCategory):
                self._opponent.update(opponent_id, enemy_race, predicted_category)
                self._opponent.save()