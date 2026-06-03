"""Belief State — Read-only snapshot consumed by decision-making code.

Purpose: Frozen dataclass holding all belief models. Each frame, BeliefUpdater
         produces a new BeliefState that downstream systems read from.

Key Decisions: Frozen dataclass prevents accidental mutation by consumers.
               BeliefState is replaced each frame, never mutated in-place.

Limitations: Phase 1 — Composition Belief. Phase 2 — Strategy Belief (optional).
              ScoutVOI and Opponent beliefs will be added in future phases.
"""

from dataclasses import dataclass

from bot.belief.composition_belief import CompositionBelief
from bot.belief.strategy_belief import StrategyBelief


@dataclass(frozen=True)
class BeliefState:
    """Read-only belief state consumed by combat, reactions, and scouting.

    Produced once per frame by BeliefUpdater. All fields are immutable —
    downstream code should never modify belief state directly.
    """

    composition: CompositionBelief
    strategy: StrategyBelief | None = None


def create_empty_belief_state() -> BeliefState:
    """Create a default BeliefState with empty beliefs — used at game start."""
    return BeliefState(composition=CompositionBelief())