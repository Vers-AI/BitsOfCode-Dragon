"""Belief Layer — Probabilistic reasoning between observation and decision-making.

Purpose: Replaces hard thresholds and binary cutoffs with probability distributions
         that decay, accumulate evidence, and produce calibrated confidence scores.

Key Decisions: Phase 1 — Composition Belief. Phase 2 — Strategy Belief.
               Phase 4 — Opponent Belief (cross-game Dirichlet priors).
               Phase 3 — Scout VOI (staleness × relevance ranking).
               Level-2 routing (proxy/cannon/rush) lives in scouting.py, not here.
               All beliefs disabled by default in competition builds.

Limitations: Scout VOI uses fixed relevance weights, not full entropy computation.
"""

from bot.belief.belief_state import BeliefState, create_empty_belief_state
from bot.belief.belief_updater import BeliefUpdater
from bot.belief.composition_belief import CompositionBelief, WeightedUnit
from bot.belief.strategy_belief import StrategyBelief, StrategyPrediction
from bot.belief.opponent_belief import OpponentBelief
from bot.belief.scout_voi import get_voi_destinations, update_location_sightings

__all__ = [
    "BeliefState",
    "BeliefUpdater",
    "CompositionBelief",
    "WeightedUnit",
    "StrategyBelief",
    "StrategyPrediction",
    "OpponentBelief",
    "create_empty_belief_state",
    "get_voi_destinations",
    "update_location_sightings",
]