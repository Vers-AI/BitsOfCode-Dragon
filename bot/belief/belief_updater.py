"""Belief Updater — Produces new BeliefState each frame from observations.

Purpose: Single update() call ingests all current observations (cached army,
         visible structures, destroyed units) and produces a new BeliefState.

Key Decisions: Composition Belief is the only model in Phase 1.
               Other beliefs (Strategy, ScoutVOI, Opponent) are added in future phases.
               Feature-gated behind config.yml `belief.enable_composition`.
               Each update() returns a snapshot so BeliefState consumers
               never see mid-frame mutation.

Limitations: Update must complete within the frame budget (~0.5ms for all beliefs).
"""

from typing import TYPE_CHECKING

from bot.belief.composition_belief import CompositionBelief
from bot.belief.belief_state import BeliefState

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

    def __init__(self):
        self._composition = CompositionBelief()
        self._destroyed_tags: set[int] = set()

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

        # Snapshot so each BeliefState is an independent copy.
        # Uses shallow dict copies (shares Unit refs — they're read-only
        # game engine snapshots). Avoids deepcopy which fails on Unit objects
        # that hold references to the unpicklable bot/client.
        return BeliefState(composition=self._composition.snapshot())