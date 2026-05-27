# Bayesian Belief Layer Plan

Probabilistic reasoning layer between observation and decision-making. Replaces hard thresholds and binary cutoffs with probability distributions that decay, accumulate evidence, and produce calibrated confidence scores.

## Problem This Solves

The bot currently handles uncertainty through **hard thresholds + hysteresis**:

| System | Current | Limitation |
|--------|---------|------------|
| Intel freshness | 0-1 linear decay, binary gates at 0.2/0.7 | "Stale" is a cliff, not a gradient |
| Threat assessment | Deterministic army_value calculation | Memory units treated as truth; no confidence |
| Combat sim | Categorical `EngagementResult` (7 levels) | No probability, no confidence interval |
| Rush detection | LogisticRegression + rule scoring (Zerg only) | No Terran/Protoss detection; baneling_nest tracked but unused |
| Scouting | Fixed urgency thresholds (0.3/0.5/0.7) | No value-of-information reasoning |
| Cross-game memory | None | Every game starts from zero |

These work, but they create brittleness: stale intel is a cliff (all-or-nothing), threats are over/under-estimated (no memory of threat history), combat predictions are uncalibrated (VICTORY_MARGINAL means different things at different supply margins), and scouting doesn't prioritize by decision impact.

---

## Architecture Overview

```
OBSERVATION ──→ BELIEF UPDATE ──→ BELIEF STATE ──→ DECISION CONSUMERS
  (existing)      (new layer)      (new layer)       (existing, modified)
```

A **BeliefState** object sits between the observation pipeline and decision-making code. It maintains probability distributions over things the bot cannot directly observe. Decisions consume beliefs, not raw cached snapshots.

### Key Principle: Conjugate Updates Only

All belief updates use **closed-form conjugate priors** (Beta-Binomial, Dirichlet-Multinomial, Exponential decay). No MCMC, no iterative sampling, no matrix inversion. Each update is O(1) or O(n) where n = number of unit types/tags. Target: ≤0.5ms per frame for all belief updates combined.

---

## The 6 Belief Models

### Model 1: Enemy Composition Belief

**What**: `P(unit_type_count = k | observations, time)` — soft probability over enemy army composition instead of binary age filter.

**Current gap**: `bot.enemy_army` uses `age < MEMORY_EXPIRY_TIME (30s)` as a hard cutoff. Units older than 30s vanish. Units 29.9s old are treated as fully real. No forward model of what the enemy likely has based on structures seen.

**Model**:
- Per-unit-type count with exponential decay confidence
- `P(unit still exists | age)` uses unit-type-specific half-lives (workers: 60s, combat units: 15-25s)
- Observed enemy structures inform priors for expected-but-unseen units (e.g., Factory → expect tanks/medivacs)

**Library**: `scipy.stats.expon` (already installed)

**Data source**: `intel` periodic events (`scouted_enemy_units`, `scouted_enemy_structures`) + `bot.mediator.get_cached_enemy_army` at runtime

**Decisions affected**:
- `can_win_fight()` → weighted unit list instead of binary-filtered list
- `assess_threat()` → weighted army_value
- `handle_attack_toggles()` → continuous `P(composition reliable)` instead of freshness cliff

**LOC estimate**: ~120 in `bot/belief/composition_belief.py`, ~20 in integration points

---

### Model 2: Strategy Belief (Generalized Rush Detection)

**What**: `P(strategy = s | timings, buildings, race, game_time)` — multi-class, multi-race strategy classifier replacing per-race booleans.

**Current gap**: Rush detection covers Zerg ling rushes only. Cannon rush is a boolean in `intel.py`. No Terran proxy detection. `baneling_nest_seen_time` is tracked but unused. The ML model returns 3 classes (12_pool, speedling, none).

**Model**: `pgmpy` DiscreteBayesianNetwork with structure encoding SC2 domain knowledge:

```
enemy_race ──→ strategy
map_distance ──→ strategy
strategy ──→ pool_timing
strategy ──→ rax_timing
strategy ──→ forge_timing
strategy ──→ nat_timing
strategy ──→ gas_timing
```

Strategy classes: `standard`, `12_pool`, `speedling`, `baneling_bust`, `proxy_rax`, `cannon_rush`, `2base_allin`

Rule-based auto-TRUE guards stay (deterministic, cheap, correct for unambiguous cases). The BN handles the ambiguous zone with probabilities.

**Library**: `pgmpy` (new dependency — pure Python, ~2.4MB wheel, sklearn-compatible)

**Data source**: Match record `cheese_type` (ground truth label), `rush_detect` events (features), `reactions` events (timing features for Terran/Protoss)

**Decisions affected**:
- `early_threat_sensor()` → consumes `P(strategy = cheese_class)` instead of per-race booleans
- `cheese_reaction()` → threshold on `P_cheese > 0.6` instead of `if cannon_rush`
- `macro.py` → nudge composition based on `P(2base_allin)` vs `P(standard)`

**LOC estimate**: ~180 in `bot/belief/strategy_belief.py`, ~50 in training script, ~30 in integration points

---

### Model 3: Threat Belief

**What**: `P(threat_class | observations, game_time)` — hierarchical probability over threat severity, with evidence accumulation.

**Current gap**: `assess_threat()` returns a deterministic float. Memory units near bases get full weight even though they may have moved. No "expected but unseen" threats. No threat persistence — each frame's assessment is independent.

**Model**: Beta-Binomial accumulation. Prior starts uninformative (alpha=1, beta=1). Evidence updates:
- Enemy near bases → alpha += weighted_signal
- Time with no enemy near bases → beta += decay
- Damage taken → alpha += bonus
- Result: `P(threat > harassment)`, `P(threat > combat)`, `P(threat > overwhelm)`

**Library**: `scipy.stats.beta` (already installed)

**Data source**: `combat` periodic events (role counts), `reactions` transition events (`under_attack`, `cheese_type`)

**Decisions affected**:
- `threat_detection()` → consumes `P(threat_class)` distribution instead of raw army_value comparison
- `allocate_defensive_forces()` → proportional to threat probability, not absolute value
- `handle_attack_toggles()` → `P(imminent_push)` factors into attack initiation

**LOC estimate**: ~100 in `bot/belief/threat_belief.py`, ~20 in integration points

---

### Model 4: Engagement Outcome Calibration

**What**: `P(actual_win | sim_result, supply_margin, intel_freshness)` — calibrated probability overlay on the ARES combat sim.

**Current gap**: `can_win_fight()` returns a categorical `EngagementResult`. Two VICTORY_MARGINAL results can have vastly different confidences (10 supply advantage vs 2), but the bot treats them identically.

**Model**:
- Record engagement outcomes: (sim_result, supply_margin, intel_freshness) → actual win/loss
- Offline calibration using `sklearn.CalibratedClassifierCV` (isotonic regression or Platt scaling)
- Runtime: continuous `P(actual_win)` instead of categorical threshold
- Decision threshold becomes `if P_win > 0.6: attack` instead of `if result >= VICTORY_MARGINAL: attack`

**Library**: `sklearn.calibration.CalibratedClassifierCV` (already installed)

**Data source**: `combat` periodic events (`can_win_fight`), `intel` snapshots (`scouted_enemy_units` for freshness), match record `result` (ground truth). **New telemetry needed**: engagement outcome labels per combat event.

**Decisions affected**:
- `handle_attack_toggles()` → `P_win > THRESHOLD` instead of `sim_result >= VICTORY_MARGINAL`
- `control_main_army()` per-squad → calibrated confidence instead of 7-level enum
- `try_mass_recall()` → `P(actual_loss > 0.8)` instead of `LOSS_DECISIVE_OR_WORSE`

**LOC estimate**: ~80 in `bot/belief/engagement_belief.py`, ~80 in calibration script

**Prerequisite**: Need engagement outcome telemetry events (track whether attack succeeded or failed)

---

### Model 5: Scout Value-of-Information

**What**: `E[I(scout_target) | current_beliefs]` — rank scout targets by expected belief entropy reduction.

**Current gap**: Scouting uses fixed urgency thresholds. Observers cycle expansions mechanically. No sense of *which information would reduce the most uncertainty*.

**Model**: For each candidate scout target, estimate expected reduction in composition belief entropy:
- Deciding whether to attack → highest VOI is seeing the enemy army
- Defending a rush → highest VOI is seeing the natural
- Macro mode → highest VOI is seeing tech structures

Uses `scipy.stats.entropy()` on current composition belief distribution to compute information gain.

**Library**: `scipy.stats.entropy` (already installed)

**Data source**: Runs entirely from Composition Belief + Threat Belief state at runtime. No offline training.

**Decisions affected**:
- `control_observers()` → rank targets by VOI instead of cycling expansions
- `control_worker_scout()` → dispatch to highest-VOI location instead of urgency threshold
- `control_hallucination_scout()` → prioritize areas with highest composition entropy

**LOC estimate**: ~60 in `bot/belief/scout_voi.py`, ~20 in integration points

**Prerequisite**: Composition Belief (Model 1) and Threat Belief (Model 3) must exist first

---

### Model 6: Opponent Style (Cross-Game)

**What**: `P(style | opponent_id, historical_games)` — Dirichlet-multinomial per opponent tracking aggressive/defensive/macro tendencies across games.

**Current gap**: No cross-game memory. Every game starts from zero.

**Model**: Simple Dirichlet concentration parameters per opponent:
- `alpha = [aggressive_count + 1, defensive_count + 1, macro_count + 1]`
- Updated after each game from match record (result, cheese_type, game length, economy metrics)
- Prior for unknown opponent: uninformative `[1, 1, 1]`
- Persists to `data/opponent_profiles.json`

**Library**: `scipy.stats.dirichlet` (already installed)

**Data source**: Match records (`result`, `cheese_type`, `length`, `opponent_id`, `enemy_race`). ARES DataManager already writes per-opponent JSON to `data/<opponent_id>-<race>.json`.

**Decisions affected**:
- Strategy Belief prior → aggressive opponent inflates `P(cheese)`
- Composition Belief prior → aggressive opponent inflates `P(combat_units)`
- Threat Belief prior → aggressive opponent inflates `P(threat > harassment)`

**LOC estimate**: ~80 in `bot/belief/opponent_belief.py`, ~40 for persistence + validation

---

## Library Dependencies

| Library | Version | Status | Used By | Why |
|---------|---------|--------|---------|-----|
| `scipy.stats` | 1.17.1 | **Already installed** (transitive via sklearn) | Models 1, 3, 5, 6 | Beta, Dirichlet, Exponential, Entropy — all conjugate priors and distributions |
| `scikit-learn` | 1.8.0 | **Already installed** (ares dep) | Model 4 | `CalibratedClassifierCV` for engagement calibration |
| `numpy` | 2.4.4 | **Already installed** | All models | Array operations for belief state |
| `pgmpy` | 1.1.2 | **New dependency** | Model 2 | `DiscreteBayesianNetwork`, `VariableElimination`, `MaximumLikelihoodEstimator` |

**`pgmpy` is the only new runtime dependency.** It is pure Python (~2.4MB wheel), requires Python ≥3.10 (compatible with our ≥3.11 constraint), has no heavy transitive deps beyond numpy/scipy which we already have, and is sklearn-compatible. Variable elimination on our small networks (7-10 nodes) runs in <1ms.

**Libraries explicitly not used**:
- **PyMC / Pyro**: MCMC is too slow for per-frame updates. Useful for offline model analysis only.
- **pomegranate**: Requires PyTorch — too heavy for an SC2 bot 0.5ms frame budget.
- **TensorFlow Probability**: Same — too heavy, wrong use case.

---

## Data Requirements

### Data Already Flowing (Telemetry)

| Belief Model | Telemetry Source | Fields Used |
|-------------|-----------------|-------------|
| Composition | `intel` periodic events | `scouted_enemy_units`, `scouted_enemy_structures`, `visible_enemy_count` |
| Strategy | `rush_detect` events + match record | All timing features, `ml_probs`, `ml_confidence`, `cheese_type` |
| Threat | `combat` periodic + `reactions` transitions | `can_win_fight`, `under_attack`, role counts, `game_phase` |
| Engagement | `combat` periodic + match record | `can_win_fight`, role counts, `result`, `length` |
| Scout VOI | Runs from other beliefs at runtime | No separate telemetry needed |
| Opponent | Match record | `opponent_id`, `result`, `cheese_type`, `length`, `enemy_race`, `map` |

### New Telemetry Events Needed

| Event | Subsystem | Why | Fields |
|-------|-----------|-----|--------|
| Engagement outcome | `combat` | Ground truth for calibration | `engagement_result: str` (win/loss/retreat), `own_supply: int`, `enemy_supply_visible: int`, `sim_result_category: str`, `intel_freshness: float` |
| Per-unit last-seen | `intel` | Foundation for composition belief decay | Omitted — will populate `_enemy_unit_last_seen` dict (already declared, never written to). This is an internal state change, not a telemetry event. |
| Observer target | `scouting` | Training data for VOI | `observer_role: str`, `target_type: str`, `target_position: str`, `observation_result: str` (saw_army/saw_structures/saw_nothing/observer_died) |
| Engagement decision | `combat` | Record why bot chose to engage or retreat | `decision: str` (engage/retreat/hold), `reason: str`, `P_win: float` (once Model 4 exists), `freshness: float` |

These follow the existing telemetry schema pattern (subsystem/action/reason + flat optional fields). Add to the "Other subsystems — to be defined" section in the telemetry plan.

### Data Volume Needed for Training

| Model | Minimum Training Games | Source |
|-------|----------------------|--------|
| Composition | 0 (online-only, conjugate priors) | N/A |
| Strategy (Zerg) | 50+ vs Zerg (already have some) | Existing + new telemetry |
| Strategy (Terran/Protoss) | 50+ vs each race | New games needed |
| Engagement calibration | 100+ engagements with outcome labels | New telemetry needed |
| Scout VOI | 0 (derived from other beliefs) | N/A |
| Opponent profiles | Accumulates over time (0 min, improves with games) | Match records |

**Critical**: `data/` directory is currently empty. The `data/games/` directory doesn't exist yet. No games have been run locally with telemetry enabled. Phase 2 (Strategy Belief) requires collecting 50+ games per race before the BN can be trained.

---

## File Structure

```
bot/
  belief/                          ← new package
    __init__.py                    ← Re-exports: create_belief_state, BeliefState
    belief_state.py                ← BeliefState dataclass (read-only snapshot consumed by decisions)
    belief_updater.py              ← Ingests observations, produces new BeliefState each frame
    composition_belief.py          ← Model 1: enemy composition with soft decay
    strategy_belief.py             ← Model 2: multi-race strategy classifier (pgmpy BN + rule guards)
    threat_belief.py               ← Model 3: threat probability with evidence accumulation
    engagement_belief.py            ← Model 4: calibrated combat outcome probability
    scout_voi.py                   ← Model 5: value-of-information for scout targeting
    opponent_belief.py             ← Model 6: cross-game opponent style profiles
  models/
    rush_detector_model.pkl        ← Existing: Zerg rush LogisticRegression (kept as fallback during transition)
    strategy_belief_model.pkl      ← New: pgmpy BN structure + fitted parameters (Phase 2)
    engagement_calibrator.pkl      ← New: CalibratedClassifierCV model (Phase 4)
scripts/
  aggregate_training_data.py       ← Aggregate telemetry JSONL into per-model training CSVs
  train_strategy_belief.py         ← Train pgmpy BN from aggregated data
  train_engagement_calibrator.py   ← Train CalibratedClassifierCV from engagement outcomes
  train_rush_model.py              ← Existing: update to read from telemetry format (currently broken)
data/
  games/                           ← Per-game JSONL telemetry (created on first write)
  training/                        ← Aggregated training datasets
    strategy_belief_train.csv
    engagement_calib_train.csv
    rush_detection_train.csv
  opponent_profiles.json           ← Cross-game opponent profiles (Model 6 persistence)
```

---

## Integration With Existing Code

### Replace Strategy (Gradual, Safety-Gated)

Each model follows this integration path:

1. **Build** the belief model alongside existing code
2. **Wire in** as an additional input (not replacing) — emit belief values to telemetry for validation
3. **Validate** that belief outputs match existing heuristics on known cases (side-by-side comparison)
4. **Switch** consumers over to belief-driven decisions
5. **Remove** the old heuristic function

This means at any point the bot works — we're never running with a broken decision pipeline.

### Specific Integration Points

| Existing Function | Belief Consumer | Change |
|-------------------|-----------------|--------|
| `bot.py:296-300` | Composition Belief | Replace binary `age < 30s` filter with weighted unit list |
| `intel.py:get_enemy_intel_quality()` | Composition Belief | Freshness score becomes one component of composition belief |
| `intel.py:update_enemy_intel_tracking()` | Composition Belief | Populate `_enemy_unit_last_seen` dict (ghost field, currently empty) |
| `reactions.py:assess_threat()` | Threat Belief | Return `P(threat_class)` distribution instead of float |
| `reactions.py:threat_detection()` | Threat Belief | Consume `P(threat_class)` for defender allocation |
| `combat.py:handle_attack_toggles()` | Engagement Belief | `P_win > THRESHOLD` instead of `sim_result >= CATEGORY` |
| `combat.py:control_main_army()` | Composition + Engagement | Weighted army in combat sim, calibrated confidence for squad decisions |
| `rush_detection.py:get_enemy_ling_rushed_v2()` | Strategy Belief | Superseded by `strategy_belief.py` (kept as fallback during transition) |
| `intel.py:get_enemy_cannon_rushed()` | Strategy Belief | Superseded by `P(cannon_rush)` from BN |
| `scouting.py:control_observers()` | Scout VOI | Rank targets by VOI instead of cycling expansions |
| `scouting.py:control_worker_scout()` | Scout VOI | Dispatch to highest-VOI location |
| `scouting.py:get_hunt_target()` | Scout VOI | Target selection by information gain |

### BeliefState Dataclass (Read-Only Snapshot)

```python
@dataclass(frozen=True)
class BeliefState:
    composition: CompositionBelief  # Model 1
    strategy: StrategyBelief        # Model 2
    threat: ThreatBelief            # Model 3
    engagement: EngagementBelief     # Model 4
    scout_voi: ScoutVOI             # Model 5
    opponent: OpponentBelief        # Model 6
```

Each sub-belief exposes probability properties:

```python
composition.P_unit_exists(tag: int) -> float           # P(unit still alive | age, type)
composition.get_weighted_army() -> list[WeightedUnit]   # For combat sim
composition.freshness -> float                          # Aggregated (replaces intel freshness)

strategy.P_cheese -> float                              # P(any cheese strategy)
strategy.P_strategy(s: str) -> float                    # P(strategy = s)
strategy.rush_label -> str                              # MAP estimate (backward compat)

threat.P_harassment -> float                            # P(threat class = harassment)
threat.P_combat -> float                                # P(threat class = combat)
threat.P_overwhelm -> float                             # P(threat class = overwhelm)

engagement.P_win(freshness: float, supply_margin: float) -> float  # Calibrated P(actual win)

scout_voi.rank_targets() -> list[tuple[Point2, float]]  # Target → expected info gain

opponent.P_aggressive -> float                           # P(opponent plays aggressive)
opponent.P_defensive -> float                            # P(opponent plays defensive)
```

### Integration Into bot.py

```python
# In PiG_Bot.on_step():
self.belief_state = self.belief_updater.update(
    cached_army=self.mediator.get_cached_enemy_army,
    visible_structures=self.enemy_structures,
    game_time=self.time,
    # ... other observations
)
# self.belief_state is now read-only for all consumers
```

Each consumer accesses what it needs:

```python
# combat.py
if bot.belief_state.engagement.P_win(freshness=0.6, supply_margin=8) > 0.6:
    commence_attack()

# reactions.py
if bot.belief_state.threat.P_combat > 0.5:
    allocate_defenders(proportional_to=bot.belief_state.threat.P_combat)

# scouting.py
target = bot.belief_state.scout_voi.rank_targets()[0]  # Highest VOI target
```

---

## Implementation Phases

### Phase 1: Composition Belief — ~120 LOC bot + 0 LOC scripts

**Files to create:**
- `bot/belief/__init__.py`
- `bot/belief/belief_state.py`
- `bot/belief/composition_belief.py`

**Files to modify:**
- `bot/bot.py` — instantiate `BeliefUpdater`, call `update()` in `on_step()`, replace binary filter
- `bot/utilities/intel.py` — populate `_enemy_unit_last_seen` dict, wire composition belief into `get_enemy_intel_quality()`
- `bot/combat/combat.py` — consume `composition.get_weighted_army()` in combat sim calls
- `bot/constants.py` — add `UNIT_HALF_LIFE` constant dict

**No new dependencies.** Uses `scipy.stats.expon` only.

**Validation**: Side-by-side comparison — emit composition belief freshness alongside existing intel freshness to telemetry. Confirm they agree on known cases (fresh units, stale units, edge cases).

---

### Phase 2: Strategy Belief — ~180 LOC bot + ~150 LOC scripts

**Files to create:**
- `bot/belief/strategy_belief.py`
- `scripts/aggregate_training_data.py`
- `scripts/train_strategy_belief.py`

**Files to modify:**
- `bot/managers/reactions.py` — `early_threat_sensor()` consumes `P_cheese` instead of per-race booleans
- `bot/utilities/rush_detection.py` — kept as fallback, `get_enemy_ling_rushed_v2()` becomes optional path
- `bot/utilities/intel.py` — `get_enemy_cannon_rushed()` becomes optional path

**New dependency**: `poetry add pgmpy` (~2.4MB pure Python wheel)

**Data prerequisite**: Need 50+ games vs each race. The `data/` directory is currently empty. Before Phase 2 can ship, must:
1. Create `data/games/` directory (handled by telemetry on first write)
2. Run 50+ games vs Zerg, 50+ vs Terran, 50+ vs Protoss
3. Run `aggregate_training_data.py` to extract training CSVs
4. Run `train_strategy_belief.py` to fit BN parameters
5. Save model to `bot/models/strategy_belief_model.pkl`

**Backward compatibility**: Existing `rush_detector_model.pkl` is kept. Strategy belief model falls back to rule-based guards (auto-TRUE) if the BN model file is missing. No regression if the model isn't ready.

---

### Phase 3: Threat Belief — ~100 LOC bot + 0 LOC scripts

**Files to create:**
- `bot/belief/threat_belief.py`

**Files to modify:**
- `bot/managers/reactions.py` — `assess_threat()` returns threat probability distribution; `threat_detection()` consumes `P_combat`, `P_overwhelm`
- `bot/combat/combat.py` — `handle_attack_toggles()` consumes `P(imminent_push)` for attack initiation

**No new dependencies.** Uses `scipy.stats.beta`.

**Depends on**: Phase 1 (composition belief feeds threat belief with weighted army).

---

### Phase 4: Engagement Belief — ~80 LOC bot + ~80 LOC scripts

**Files to create:**
- `bot/belief/engagement_belief.py`
- `scripts/train_engagement_calibrator.py`

**Files to modify:**
- `bot/combat/combat.py` — `handle_attack_toggles()` uses `P_win` instead of categorical thresholds; `control_main_army()` uses calibrated confidence
- `bot/utilities/telemetry.py` / `bot/utilities/game_report.py` — add `engagement` subsystem events

**No new dependencies.** Uses `sklearn.calibration.CalibratedClassifierCV`.

**Data prerequisite**: Need 100+ engagement events with outcome labels. This requires new telemetry for engagement outcomes (which attacks succeeded vs failed).

---

### Phase 5: Scout VOI — ~60 LOC bot + 0 LOC scripts

**Files to create:**
- `bot/belief/scout_voi.py`

**Files to modify:**
- `bot/managers/scouting.py` — `control_observers()`, `control_worker_scout()`, `get_hunt_target()` rank by VOI

**No new dependencies.** Uses `scipy.stats.entropy`.

**Depends on**: Phase 1 (Composition Belief) and Phase 3 (Threat Belief) must exist to compute entropy reduction.

---

### Phase 6: Opponent Belief — ~80 LOC bot + ~40 LOC scripts

**Files to create:**
- `bot/belief/opponent_belief.py`
- `data/opponent_profiles.json` (created on first write, schema-versioned)

**Files to modify:**
- `bot/bot.py` — load opponent profile at game start, save at game end
- `bot/belief/strategy_belief.py` — adjust prior based on opponent style
- `bot/belief/threat_belief.py` — adjust prior based on opponent style

**No new dependencies.** Uses `scipy.stats.dirichlet`.

**Persistence**: `data/opponent_profiles.json` with schema versioning, rolling cap (max 200 opponents), safe fallback on corruption.

**Competition safety**: Opponent belief defaults to uninformative prior `[1, 1, 1]` if no data. No per-frame I/O. Profile is loaded once at game start and written once at game end.

---

## Performance Guardrails

| Constraint | Implementation |
|-----------|---------------|
| No MCMC or iterative sampling | All updates are conjugate (closed-form). Beta-Binomial, Dirichlet-Multinomial, Exponential decay are O(1) per update. |
| Per-frame budget: ≤0.5ms for all beliefs | Profile with `PerformanceMonitor`. If over budget, skip non-critical updates (threat can update every 10 frames; composition must update every frame). |
| pgmpy VariableElimination: ≤1ms per query | Our networks are sparse (7-10 nodes, max 3-4 states per node). VE on this size is sub-ms. Profile to confirm. If too slow, cache inference results and re-query only when evidence changes. |
| Model loading: one-time at game start | `strategy_belief_model.pkl` and `engagement_calibrator.pkl` loaded in `on_start()`. No per-frame I/O. |
| Persistence: ≤1KB per write, ≥30s between writes | Opponent profile written once per game. Telemetry events are existing infrastructure. |
| Competition-safe defaults | If belief models fail to load or produce NaN, fall back to existing heuristics. Feature-gated behind `config.yml` flags (`belief.enable_composition`, `belief.enable_strategy`, etc.), all defaulting to **off** in competition builds. |

### Feature Flags (config.yml)

```yaml
belief:
  enable_composition: false    # Phase 1
  enable_strategy: false       # Phase 2
  enable_threat: false          # Phase 3
  enable_engagement: false     # Phase 4
  enable_scout_voi: false      # Phase 5
  enable_opponent: false        # Phase 6
```

Each flag defaults to `false`. Beliefs are disabled in competition until validated. When disabled, existing heuristics run unchanged. Flags are documented in config and in the telemetry plan.

---

## Training Pipeline

### Broken: `scripts/train_rush_model.py`

The existing training script reads from `data/rush_detection_log.jsonl` which no longer exists. It must be updated to read from telemetry JSONL format first, regardless of whether Phase 2 is implemented. This is the **smallest change to unblock model retraining**.

### New: `scripts/aggregate_training_data.py`

Reads all `data/games/*.jsonl`, joins match records with events by `match_id`, and outputs per-model training CSVs:

- `data/training/strategy_belief_train.csv` — one row per game, all timing features + cheese_type label
- `data/training/engagement_calib_train.csv` — one row per engagement, sim_result + supply_margin + freshness + actual_outcome
- `data/training/rush_detection_train.csv` — backward-compat with existing script format

### New: `scripts/train_strategy_belief.py`

1. Load `data/training/strategy_belief_train.csv`
2. Define BN structure (arcs from SC2 domain knowledge)
3. Fit parameters with `MaximumLikelihoodEstimator`
4. Validate accuracy against held-out test set
5. Save model to `bot/models/strategy_belief_model.pkl`

### New: `scripts/train_engagement_calibrator.py`

1. Load `data/training/engagement_calib_train.csv`
2. Train a `CalibratedClassifierCV` with isotonic regression
3. Validate calibration (reliability diagram)
4. Save model to `bot/models/engagement_calibrator.pkl`

---

## Complexity Budget Assessment

Per AGENTS.md rules: +5 points max per task, refactor exemption for splits >500 LOC.

| Phase | New Files | New Classes | New Deps | API Changes | Cross-Module | Total | Status |
|-------|-----------|-------------|----------|-------------|-------------|-------|--------|
| 1 | +2 | +2 (CompositionBelief, BeliefState) | 0 | +1 (belief_state on bot) | +1 (combat, intel) | 4 | Within budget |
| 2 | +3 | +1 (StrategyBelief) | +1 (pgmpy) | 0 | +1 (reactions) | 4 | Within budget (pgmpy = +3 dep cost, but -2 from splitting rush_detect into belief module) |
| 3 | +1 | +1 (ThreatBelief) | 0 | +1 (assess_threat return type) | +1 (reactions, combat) | 3 | Within budget |
| 4 | +2 | +1 (EngagementBelief) | 0 | +1 (P_win in handle_attack) | +1 (combat) | 3 | Within budget |
| 5 | +1 | +1 (ScoutVOI) | 0 | 0 | +1 (scouting) | 2 | Within budget |
| 6 | +2 | +1 (OpponentBelief) | 0 | 0 | +1 (bot, strategy, threat) | 2 | Within budget |

Each phase is a separate task. Budget resets per phase.

---

## Over-Engineering Triggers (Self-Check)

- ✅ No class for logic with <3 methods and no state — `CompositionBelief`, `ThreatBelief`, etc. all have ≥5 related state variables
- ✅ No new config flags beyond the feature gates (which are the existing pattern from `config.yml`)
- ✅ No adapters/interfaces with only one implementation — each belief model has one implementation, no adapter pattern
- ✅ No new dependency replacing ≤10 LOC — pgmpy replaces 300+ LOC of hand-rolled rules in `rush_detection.py` and `reactions.py`
- ✅ No pipelines/state machines where a loop + guard works — belief updates are simple functions called sequentially from `belief_updater.py`

---

## Shadow Review

1. **Biggest assumption**: That pgmpy's `VariableElimination` runs fast enough for per-frame inference on our networks (7-10 nodes). It should be — small networks, sparse connectivity, 3-4 states per node. But it must be profiled on the first implementation. Plan: if >0.5ms, cache inference results and only re-query when evidence changes (the evidence only changes when a scout report comes in, which is at most once per second).

2. **Most likely failure/edge**: `data/` is empty — no games have been run locally with telemetry enabled. The training pipeline for Phase 2 (Strategy Belief) cannot ship without collecting 50+ games per race. The first thing to do is run games and validate the telemetry pipeline works end-to-end before attempting any model training.

3. **Smallest change to improve robustness**: Populate `_enemy_unit_last_seen` (10 LOC in `update_enemy_intel_tracking()`). This is the foundation for composition belief decay and is currently a ghost — declared but never written to. It costs nothing, breaks nothing, and unblocks Phase 1.