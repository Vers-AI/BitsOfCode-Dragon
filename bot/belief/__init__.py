"""Belief Layer — Probabilistic reasoning between observation and decision-making.

Purpose: Replaces hard thresholds and binary cutoffs with probability distributions
         that decay, accumulate evidence, and produce calibrated confidence scores.

Key Decisions: Phase 1 only — Composition Belief. Feature-gated behind config.yml.
               All beliefs disabled by default in competition builds.

Limitations: Future phases (Strategy, ScoutVOI, Opponent) not yet implemented.
"""

from bot.belief.belief_state import BeliefState, create_empty_belief_state
from bot.belief.belief_updater import BeliefUpdater
from bot.belief.composition_belief import CompositionBelief, WeightedUnit

__all__ = [
    "BeliefState",
    "BeliefUpdater",
    "CompositionBelief",
    "WeightedUnit",
    "create_empty_belief_state",
]