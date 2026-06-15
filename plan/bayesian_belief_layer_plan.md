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
| Scouting | Fixed urgency thresholds (0.3/0.5/0.7) | No value-of-information reasoning for destinations |
| Cross-game memory | None | Every game starts from zero |

These work, but they create brittleness: stale intel is a cliff (all-or-nothing), threats are over/under-estimated (no confidence weighting), and scouting doesn't prioritize destinations by decision impact.

---

## Architecture Overview

```
OBSERVATION ──→ BELIEF UPDATE ──→ BELIEF STATE ──→ DECISION CONSUMERS
  (existing)      (new layer)      (new layer)       (existing, modified)

TELEMETRY ──→ Cloud API ──→ Training scripts (model fitting)
                   │
                   └──→ DuckDB (local JSONL) ──→ Streamlit (monitoring/analysis)

REPLAY DATA ──→ DuckDB ──→ Ground truth for calibration
```

A **BeliefState** object sits between the observation pipeline and decision-making code. It maintains probability distributions over things the bot cannot directly observe. Decisions consume beliefs, not raw cached snapshots.

### Key Principle: Conjugate Updates Only

All belief updates use **closed-form conjugate priors** (Beta-Binomial, Dirichlet-Multinomial, Exponential decay). No MCMC, no iterative sampling, no matrix inversion. Each update is O(1) or O(n) where n = number of unit types/tags. Target: ≤0.5ms per frame for all belief updates combined.

### Key Principle: Threat Assessment Is a Consumer, Not a Model

`assess_threat()` is a damage estimator — it answers "how much force is near our bases?" Bayesian reasoning improves it by feeding probability-weighted unit lists (Composition Belief) and predictive strategy priors (Strategy Belief) into the existing function, not by replacing it with a separate Threat Belief model. See Phase 1 for details.

---

## The 4 Phases

### Phase 1: Composition Belief — Enemy Army Probability ✅ COMPLETE

**What**: `P(unit still exists | age, type)` + structure-based priors for expected-but-unseen units.

**Current gap**: `bot.enemy_army` uses `age < MEMORY_EXPIRY_TIME (30s)` as a hard cutoff. Units older than 30s vanish. Units 29.9s old are treated as fully real. No forward model of what the enemy likely has based on structures seen.

**Model**:

Per-unit exponential decay confidence:

```
P(unit still exists | age, type) = 0.5^(age / half_life)
```

Half-lives vary by type because workers survive longer than combat units:

| Unit Category | Half-Life (seconds) | Rationale |
|---------------|---------------------|-----------|
| Workers (SCV, Drone, Probe) | 60 | Workers tend to stay alive mining |
| Combat units (Marine, Stalker, etc.) | 20 | Combat units die in fights |
| Fragile units (Oracle, Banshee, Mutalisk) | 15 | High-value targets get focused |
| Structure-like units (Warp Prism, Medivac) | 30 | Support units survive if army does |
| Buildings (visible as units) | 90 | Buildings don't move or die quickly |

**Unit destruction handling**: When a unit is confirmed destroyed (via `on_unit_destroyed` callback), it's immediately removed from the belief state with `P(exists) = 0.0`. This is ground truth — no decay needed.

**Units that vanish into fog of war** (not destroyed, just out of vision): These are where exponential decay applies. We don't know if they died or moved, so the probability decays over time. If we re-see the unit, it snaps back to `P(exists) = 1.0`.

**Structure-based priors**: When we see enemy structures, we update our expectation of what units they can produce:

```
P(seeing Marine in next 30s | Barracks seen) = 0.7
P(seeing Tank in next 30s | Factory seen) = 0.6
P(seeing Medivac in next 30s | Starport seen) = 0.5
```

These are simple lookup tables derived from SC2 tech trees, not machine learning. They seed the composition belief with "expected but unseen" units — units we haven't spotted yet but the structures tell us are likely coming.

**How threat assessment consumes this**: `assess_threat()` currently sums `army_value * health_factor` for each cached unit near bases. With Composition Belief, it becomes `army_value * health_factor * P(unit still exists)`. A stalker seen 25s ago near our base with `P = 0.35` contributes only 35% of its threat value. This fixes the bug where stale memory units get full weight.

**How combat sim consumes this**: `can_win_fight()` receives a weighted unit list instead of a binary-filtered list. When intel is stale, the sim sees fewer "effective" enemy units, making the bot naturally more cautious about attacking — without the harsh stale/fresh cliff.

**Library**: `scipy.stats.expon` (already installed)

**Data source**: `intel` periodic events (`scouted_enemy_units`, `scouted_enemy_structures`) + `bot.mediator.get_cached_enemy_army` at runtime + `on_unit_destroyed` callback

**Decisions affected**:
- `can_win_fight()` → weighted unit list instead of binary-filtered list — **this is the primary combat value**
- `assess_threat()` → weighted `army_value * P(exists)` instead of full-weight stale units
- `handle_attack_toggles()` → continuous `P(composition reliable)` instead of freshness cliff
- `threat_detection()` → weighted known_army_value in threat ratio calculation

**Files to create**:
- `bot/belief/__init__.py`
- `bot/belief/belief_state.py`
- `bot/belief/composition_belief.py`

**Files to modify**:
- `bot/bot.py` — instantiate `BeliefUpdater`, call `update()` in `on_step()`, replace binary filter, wire `on_unit_destroyed` callback
- `bot/intel/intel_quality.py` — populate `_enemy_unit_last_seen` dict (ghost field, currently empty), wire composition belief into `get_enemy_intel_quality()`
- `bot/managers/reactions.py` — `assess_threat()` uses `composition.get_weighted_army()` for near-base threat calculation
- `bot/combat/combat.py` — pass weighted army to `can_win_fight()` and `handle_attack_toggles()`
- `bot/constants.py` — add `UNIT_HALF_LIFE` constant dict

**No new dependencies.** Uses `0.5 ** (age / half_life)` directly — no scipy import needed at runtime.

**LOC estimate**: ~150 in `bot/belief/`, ~30 in integration points

**Validation**: Side-by-side comparison — emit composition belief freshness alongside existing intel freshness to telemetry. Confirm they agree on known cases (fresh units, stale units, edge cases).

#### Implementation Status (Completed)

**Files created**:
- `bot/belief/__init__.py` — Re-exports BeliefState, BeliefUpdater, CompositionBelief, WeightedUnit
- `bot/belief/belief_state.py` — Frozen dataclass holding BeliefState (composition only for Phase 1)
- `bot/belief/composition_belief.py` — Core model: exponential decay, structure priors, weighted army, on_unit_destroyed
- `bot/belief/belief_updater.py` — Produces new BeliefState each frame from observations

**Files modified**:
- `bot/constants.py` — Added UNIT_HALF_LIFE (65 unit types), DEFAULT_HALF_LIFE (20s), STRUCTURE_SEEN_UNIT_PRIOR (12 structures)
- `bot/intel/intel_quality.py` — Populated `_enemy_unit_last_seen[tag] = bot.time` for visible units
- `bot/bot.py` — BeliefUpdater and BeliefState in __init__; belief update in on_step() (feature-gated); record_destruction in on_unit_destroyed()
- `bot/combat/combat.py` — 3 integration points: weighted army for can_win_fight, belief freshness for intel gate
- `bot/managers/reactions.py` — threat_detection(): confidence-weighted army value when belief enabled
- `bot/utilities/game_report.py` — Added `belief` periodic telemetry event
- `config.yml` — Added `Belief.enable_composition: True` feature flag

**Feature-gated**: All changes behind `config.yml: Belief.enable_composition`. When disabled, zero impact on existing behavior.

---

### Phase 2: Enemy Strategy Belief — Multi-Race Strategy Classification

**What**: `P(strategy = s | timings, buildings, race, game_time)` — multi-class, multi-race strategy classifier replacing per-race booleans.

**Current gap**: Rush detection covers Zerg ling rushes only. Cannon rush is a boolean in `intel.py`. No Terran proxy detection. `baneling_nest_seen_time` is tracked but unused. The ML model returns 3 classes (12_pool, speedling, none). The new taxonomy (Section "Strategy classes") replaces this with a 4-category system (cheese/all_in/timing_attack/macro) with per-race Level 2 build labels, covering all matchups.

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

**Strategy classes** (two-level taxonomy):

Level 1 (strategy category) — the top-level prediction target for Strategy Belief:

```
Enemy Strategy Belief (Level 1)
├── cheese         (early aggression before standard economy, no transition plan)
├── all_in         (committed attack from 1-2 bases, no transition if it fails)
├── timing_attack  (attack at a specific power spike, with a transition plan)
└── macro          (standard economic play, focus on long-term advantage)
```

These map directly to Spawning Tool's proven taxonomy (Cheese, All-In, Timing Attack, Economic)
used across thousands of community-labeled replays.

Level 2 (build label) — the fine-grained prediction target for Composition Belief,
conditioned on the Level 1 category and observed buildings:

```
Zerg (opponent):
  cheese:      12_pool, proxy_hatch_spine
  all_in:      roach_ravager_push, mutalisk_all_in, hydra_all_in
  timing:      ling_bane_timing, roach_timing, drop_timing
  macro:       standard_hatch_first, roach_macro, hydra_lurker, mutalisk_harass, brood_lord_late

Protoss (opponent):
  cheese:      cannon_rush, proxy_gateway
  all_in:      four_gate, two_base_colossus, two_base_blink
  timing:      stargate_timing, immortal_timing, chargelot_archon, dt_drop
  macro:       three_base_macro, sky_toss, tempest_turtle

Terran (opponent):
  cheese:      proxy_rax, bunker_rush
  all_in:      battlecruiser_rush, cyclone_push
  timing:      bio_timing, widow_mine_drop, tank_timing
  macro:       bio_macro, mech, ghost_late
```

Key design decisions:
- **`macro`** replaces the old `standard` label — same meaning, clearer term, aligns with Spawning Tool
- **`cheese`** subsumes rush (rush is a type of cheese), same as before
- **`timing_attack`** is distinct from cheese — mid-game aggression off a macro opening
- **`all_in`** captures committed aggression with no transition planned
- Level 2 labels are observable from build order timings (building X before time T → label Y)
- New labels can be added as edge cases emerge (e.g., `proxy_zealot` for PvP-specific cheese)

Rule-based auto-TRUE guards stay (deterministic, cheap, correct for unambiguous cases). The BN handles the ambiguous zone with probabilities.

**Connection to Opponent Belief**: Opponent Belief (Phase 4) provides the **prior** for Strategy Belief. If you've played this opponent before and they tend to cheese, your Strategy Belief starts with a higher prior for cheese. If it's a new opponent, you start with a flat prior and let observations speak. This is the Bayesian connection — Opponent Belief doesn't classify opponents into separate categories from Strategy Belief; it informs the prior:

```
New opponent:     P(cheese) = 0.15  (flat prior)
Known cheesy:     P(cheese) = 0.70  (7 out of 10 games were cheese)
Known macro:      P(cheese) = 0.05  (rarely cheeses)
```

The categories in Opponent Belief (aggressive/defensive/macro) map directly to Strategy Belief priors — they're not a separate classification system, they're the cross-game memory that bootstraps in-game classification.

**Library**: `pgmpy` (new dependency — pure Python, ~2.4MB wheel, sklearn-compatible)

**Data source**: Match record `cheese_type` (ground truth label), `rush_detect` events (features), `reactions` events (timing features for Terran/Protoss). Training via cloud telemetry API (`/api/features/match-level-full`). Local DuckDB+JSONL for Streamlit analysis.

**Decisions affected**:
- `early_threat_sensor()` → consumes `P(strategy = cheese_class)` instead of per-race booleans
- `cheese_reaction()` → threshold on `P_cheese > 0.6` instead of `if cannon_rush`
- `macro.py` → nudge composition based on `P(timing_attack)` vs `P(macro)` (Phase B: strategy_nudge_proportions)
- `_under_attack` flag → Strategy Belief informs threat detection thresholds (Phase A: STRATEGY_THREAT_MULTIPLIER)
- `scouting.py` → strategy-aware hunt targets (Phase C) and scout waypoints (Phase D)

**Files to create**:
- `bot/belief/strategy_belief.py`
- `scripts/train_strategy_belief.py`

**Files to modify**:
- `bot/managers/reactions.py` — `early_threat_sensor()` consumes `P_cheese` instead of per-race booleans
- `bot/intel/strategy_detect.py` — `detect_cheese()` becomes optional path (called by strategy belief fallback)
- `bot/intel/enemy_timings.py` — race-agnostic timing observations (replaces per-file tracking)

**New dependency**: `poetry add pgmpy` (~2.4MB pure Python wheel). **Note**: pgmpy is now a **dev-only dependency** — runtime loads the pickled model via `joblib`; only the training script requires pgmpy directly.

**Data prerequisite**: Need 50+ games vs each race. The cloud API (`/api/features/match-level-full`) already has 143+ games. Must:
1. Run `scripts/train_strategy_belief.py` (fetches from telemetry API, not local JSONL)
2. Verify data quality via API responses (class balance, feature distributions, NULL rates)
3. Train pgmpy DiscreteBayesianNetwork (discretize timing features → fit BN)
4. Save model to `bot/models/strategy_belief_model.pkl`
5. For local monitoring/analysis: ensure `data/games/` JSONL files exist, query via DuckDB+Streamlit

**Backward compatibility**: Existing `rush_detector_model.pkl` is superseded by `strategy_belief_model.pkl`. Strategy belief model falls back to rule-based scoring if the BN model file is missing. No regression if the model isn't ready.

**LOC estimate**: ~950 in `bot/belief/strategy_belief.py`, ~300 in training script, ~50 in integration points

#### Implementation Status (Complete ✅)

**Architecture**: Three-layer evaluation — BN model → auto-TRUE guards (fallback) → rule-based scoring (last resort). BN model is the primary path when available. Guards are deterministic (P=1.0) and fire when the model is missing or produces no result. Rule-based scoring produces soft probabilities from accumulated evidence. Falls back to flat prior when no evidence exists.

**Files created**:
- `bot/belief/strategy_belief.py` — Core module: StrategyCategory enum (4 categories), StrategyPrediction dataclass, StrategyBelief class with 3-layer evaluation, per-race rule scoring, chat messages, Level-2 label inference. ~950 LOC.
- `scripts/train_strategy_belief.py` — BN training script: fetches from telemetry API, discretizes timing features, trains pgmpy DiscreteBayesianNetwork, saves to `bot/models/strategy_belief_model.pkl`. Includes opponent priors builder with `--priors-output` and `--skip-priors` CLI flags. ~300 LOC.

**Files modified**:
- `bot/constants.py` — Added StrategyCategory enum, STRATEGY_CATEGORY_PRIOR (4 category priors biased toward macro), STRATEGY_LABELS (per-race Level-2 taxonomy), STRATEGY_TIMING_GUARDS (per-race timing thresholds). Later: STRATEGY_THREAT_MULTIPLIER, STRATEGY_THREAT_CLEAR_MULTIPLIER, STRATEGY_EXPECTED_UNITS, STRATEGY_NUDGE_MAX, STRATEGY_NUDGE_THRESHOLD, STRATEGY_HUNT_TARGETS, STRATEGY_HUNT_THRESHOLD, STRATEGY_SCOUT_WAYPOINTS, STRATEGY_SCOUT_OVERRIDE_THRESHOLD.
- `bot/belief/belief_state.py` — Added `strategy: StrategyBelief | None = None` field to BeliefState dataclass
- `bot/belief/belief_updater.py` — Added `enable_strategy` + `enable_opponent` constructor params; creates StrategyBelief when enabled; includes strategy snapshot in BeliefState
- `bot/belief/__init__.py` — Added StrategyBelief, StrategyPrediction, OpponentBelief to exports
- `bot/bot.py` — Passes `enable_strategy` config flag to BeliefUpdater constructor
- `bot/managers/reactions.py` — `early_threat_sensor()` now checks strategy belief when `enable_strategy` is on; uses `P(cheese) >= 0.6` threshold to trigger cheese response; falls back to existing boolean system when disabled. `threat_detection()` uses strategy-aware thresholds (Phase A).
- `bot/managers/macro.py` — `strategy_nudge_proportions()` (Layer 1.5) shifts composition toward units effective vs predicted enemies using COUNTER_TABLE + STRATEGY_EXPECTED_UNITS (Phase B). `_FakeEnemy` class for virtual enemy type representation.
- `bot/managers/scouting.py` — `get_strategy_hunt_targets()` overrides hunt target order (Phase C). `get_strategy_scout_waypoints()` overrides build runner scout waypoints (Phase D).
- `bot/intel/enemy_timings.py` — Race-agnostic timing observation layer (all races)
- `bot/intel/strategy_detect.py` — Multi-race strategy classification (auto-TRUE guards + rule scoring)
- `bot/intel/intel_quality.py` — Intel freshness/urgency tracking (replaces old `bot/utilities/intel.py`)
- `bot/utilities/game_report.py` — Added strategy belief telemetry event (periodic `strategy` subsystem) and strategy fields to match record
- `bot/utilities/debug.py` — Added strategy belief line to in-game debug overlay (label, source, probability bars)
- `bot/utilities/game_report.py` — Added console print of strategy prediction every 30s for live validation
- `config.yml` — Added `Belief.enable_strategy: True` feature flag

**Design decisions**:
- `StrategyCategory` enum lives in `bot/constants.py` (not `strategy_belief.py`) to avoid circular imports with the constants dict
- Evaluation order: BN model (primary) → auto-TRUE guards (fallback when model missing) → rule-based scoring (last resort). Guards are deterministic (P=1.0) but only fire as fallback, not override.
- ARES mediator booleans mapped to Level-1 categories with calibrated probabilities (e.g., `four_gate → all_in 0.8`, `marine_rush → cheese 0.7`) since ARES flags can lag behind ground truth
- BN model slot loaded from `bot/models/strategy_belief_model.pkl` via `joblib`; falls back gracefully when missing
- `pgmpy` is a **dev-only dependency** — runtime loads pickled model via `joblib`; training script requires `poetry install` with dev group
- Auto-TRUE guards delegate to `bot.intel.detect_cheese()` which consolidates all race-specific detectors
- Rule-based scoring accumulates soft evidence (pool timing, nat expansion, baneling nest) into probability adjustments, normalized to sum to 1.0
- `Level-2` label inference (`_infer_level2()`) maps observations to specific build labels within the predicted category
- Training script uses `/api/features/match-level-full` endpoint (45 columns including new engagement/economy metrics)
- Training script has 3-tier label derivation: API `strategy_category` column → `cheese_type` mapping → game-heuristic fallback
- Debug overlay format: `Strat: macro(rules) C:15% A:10% T:25% M:50%`, console adds `[level2_build_label]`
- Nudge pipeline: counter-table (Step 1) → strategy nudge (Step 1.5, predictive forward model) → resource-pressure (Step 2) → priority reorder (Step 3)

**Not yet done**:
- Integration testing with games (`enable_strategy: True` is on; needs live validation)

**Done (this session — Phases A-E)**:
- ✅ Phase A: Strategy-aware threat thresholds — `STRATEGY_THREAT_MULTIPLIER` and `STRATEGY_THREAT_CLEAR_MULTIPLIER` in `threat_detection()` (reactions.py)
- ✅ Phase B: Strategy-aware composition nudging — `strategy_nudge_proportions()` using COUNTER_TABLE derivation with `STRATEGY_EXPECTED_UNITS` (macro.py)
- ✅ Phase C: Strategy-aware hunt targets — `get_strategy_hunt_targets()` modifying `get_hunt_target()` fallback (scouting.py)
- ✅ Phase D: Strategy-aware build runner scout routing — `get_strategy_scout_waypoints()` overriding YAML waypoints (scouting.py)
- ✅ BN model training — trained and saved to `bot/models/strategy_belief_model.pkl` (previous session)
- ✅ Opponent Belief — Dirichlet priors from API, saved to `bot/models/opponent_priors.json` (previous session)

---

### Phase 3: Scout VOI — Destination Selection by Information Gain

**What**: Rank scout **destinations** by expected belief entropy reduction. The type of scout (observer, hallucinated phoenix, worker probe) remains determined by stage and availability, as it is today.

**Current gap**: Scouting uses fixed urgency thresholds. Observers cycle expansions mechanically. No sense of *which destination would reduce the most uncertainty for the current decision*.

**Model**: For each candidate scout destination, estimate expected reduction in composition belief entropy:
- Deciding whether to attack → highest VOI is seeing the enemy army
- Defending a rush → highest VOI is seeing the natural (did they expand or commit?)
- Macro mode → highest VOI is seeing tech structures (what composition are they building?)

Uses `scipy.stats.entropy()` on current composition belief distribution to compute information gain.

**Important clarification**: VOI affects **where** scouts go, not **which type** of scout to use. Observer > hallucinated phoenix > worker probe selection stays in the existing `scouting.py` logic, which correctly prioritizes by capability and availability. VOI only changes the destination ranking within that scout type.

**Library**: `scipy.stats.entropy` (already installed)

**Data source**: Runs entirely from Composition Belief + Strategy Belief state at runtime. No offline training.

**Decisions affected**:
- `get_hunt_target()` → rank destinations by VOI instead of cycling expansions mechanically
- `control_observers()` → prioritize the destination with highest information gain
- `control_hallucination_scout()` → send to highest-VOI destination (not just based on urgency)

**Files to create**:
- `bot/belief/scout_voi.py`

**Files to modify**:
- `bot/managers/scouting.py` — `get_hunt_target()`, observer assignment, hallucination scout targeting

**No new dependencies.** Uses `scipy.stats.entropy`.

**Depends on**: Phase 1 (Composition Belief) must exist to compute entropy. Phase 2 (Strategy Belief) enhances VOI by adding "what strategy are they on?" as a question to resolve.

**LOC estimate**: ~60 in `bot/belief/scout_voi.py`, ~20 in integration points

---

### Phase 4: Opponent Belief — Cross-Game Priors

**What**: `P(style | opponent_id, historical_games)` — Dirichlet-multinomial per opponent, tracking strategy tendencies across games.

**Current gap**: No cross-game memory. Every game starts from zero.

**Model**: Simple Dirichlet concentration parameters per opponent. The categories match Strategy Belief's parent categories (cheese, all_in, timing_attack, macro):

```
Opponent Belief alpha params → Strategy Belief priors:
├── alpha[cheese]      → inflates P(cheese) in Strategy Belief
├── alpha[all_in]      → inflates P(all_in) in Strategy Belief
├── alpha[timing]      → inflates P(timing_attack) in Strategy Belief
└── alpha[macro]       → inflates P(macro) in Strategy Belief
```

Updated after each game from match record (predicted strategy category). Prior for unknown opponent: uninformative `[1, 1, 1, 1]`. Persists to `data/opponent_profiles.json`.

**Connection to Strategy Belief**: Opponent Belief is the **prior** for Strategy Belief. It's not a separate classification of the opponent's playstyle — it's the cross-game memory that informs what Strategy Belief should start believing before seeing any in-game evidence:

```
Game start against known cheesy opponent:
  Strategy Belief prior: P(cheese) = 0.70 (from Opponent Belief)
  After seeing early pool at 0:40:
  Strategy Belief posterior: P(12_pool) = 0.92 (prior pushed the update)

Game start against unknown opponent:
  Strategy Belief prior: P(cheese) = 0.15 (flat prior)
  After seeing early pool at 0:40:
  Strategy Belief posterior: P(12_pool) = 0.78 (only in-game evidence)
```

**Integration formula**: `P(adjusted) = P(BN) * alpha / sum(alpha)`, then normalize. This is equivalent to a Dirichlet-multinomial posterior where BN provides the likelihood and opponent history provides the prior. When no opponent prior is available, BN output passes through unchanged.

**Two data loops keep priors fresh**:

1. **Ladder Learning** (real-time): Bot updates alpha params after each game and saves to `data/opponent_profiles.json`. AI Arena preserves the `data/` folder across games, so priors accumulate over a ladder session.
2. **Offline Rebuild** (when re-uploading zip): `scripts/train_strategy_belief.py` queries the API, aggregates matches by `(opponent_id, enemy_race)`, and saves `bot/models/opponent_priors.json`. This is baked into the next ladder zip upload and becomes the new baseline.

**Load order** (first found wins):
1. `data/opponent_profiles.json` — runtime, ladder-accumulated (most recent)
2. `bot/models/opponent_priors.json` — baseline from training script (baked into zip)
3. Flat `[1,1,1,1]` — no data at all (first game ever)

**Library**: No new dependencies. Uses plain dict math for Dirichlet alpha params.

**Data source**: Match records (`opponent_id`, `enemy_race`, `strategy_label` from telemetry). Training script aggregates from cloud API. Runtime updates from bot's own predictions.

**Decisions affected**:
- Strategy Belief prior → known cheesy opponent inflates `P(cheese)`
- Composition Belief prior → known aggressive opponent inflates `P(combat_units over workers)` (future)
- Scouting priority → unknown opponent gets more early scouts (higher VOI for information) (future)

**Files to create**:
- `bot/belief/opponent_belief.py`

**Files to modify**:
- `bot/bot.py` — load opponent profile at game start, save at game end
- `bot/belief/strategy_belief.py` — accept and apply opponent prior in `_evaluate_model()`
- `bot/belief/belief_updater.py` — create OpponentBelief, pass prior to strategy, load/save methods
- `bot/belief/__init__.py` — export OpponentBelief
- `bot/utilities/game_report.py` — add `opponent_prior_applied` to match-end telemetry
- `config.yml` — add `enable_opponent` feature flag
- `scripts/train_strategy_belief.py` — add `build_opponent_priors()` function + CLI flags

**No new dependencies.** Pure dict math for Dirichlet alpha params.

**Persistence**: `data/opponent_profiles.json` with schema versioning, rolling cap (max 200 opponents), safe fallback on corruption. Atomic write via temp file + rename.

**Competition safety**: Opponent belief defaults to uninformative prior `[1, 1, 1, 1]` if no data or opponent is unknown. No per-frame I/O. Profile is loaded once at game start and written once at game end. `data/` folder persists on AI Arena across games.

**LOC estimate**: ~170 in `bot/belief/opponent_belief.py`, ~80 in training script, ~40 in integration points

#### Implementation Status (Complete ✅)

**Files created**:
- `bot/belief/opponent_belief.py` — OpponentBelief class with load(), get_prior(), update(), save(). Dirichlet alpha params per (opponent_id, enemy_race). Schema-versioned JSON persistence. Rolling cap 200 opponents. Atomic write. Debug logging for known opponents. Flat prior fallback for unknown opponents.

**Files modified**:
- `bot/belief/strategy_belief.py` — `update()` now accepts `opponent_prior` param; `_evaluate_model()` multiplies BN output by prior then normalizes; source label changes to `"BN+OPP"` when prior is applied
- `bot/belief/belief_updater.py` — Added `OpponentBelief` instance; `load_opponent()` and `save_opponent()` methods; passes opponent prior to strategy update each frame
- `bot/belief/__init__.py` — Added `OpponentBelief` to exports
- `bot/bot.py` — Loads opponent profiles in `on_start()`, saves in `on_end()` with final strategy prediction; `enable_opponent` config flag
- `bot/utilities/game_report.py` — Added `opponent_prior_applied: True` to match-end telemetry when BN+OPP source is used
- `config.yml` — Added `enable_opponent: True` under `Belief:`
- `scripts/train_strategy_belief.py` — Added `build_opponent_priors()` function that aggregates matches by `(opponent_id, enemy_race)`, computes Dirichlet alpha params, and saves to `bot/models/opponent_priors.json`; added `--priors-output` and `--skip-priors` CLI flags; `MIN_GAMES_PER_OPPONENT = 3` threshold
- `bot/__init__.py` — Version bumped to 0.10.0

**Design decisions**:
- Two-file architecture: `data/opponent_profiles.json` (runtime, ladder-accumulated) and `bot/models/opponent_priors.json` (training, API-derived). Runtime file takes precedence.
- `data/` folder persists on AI Arena across games — bot can learn on the ladder
- `bot/models/` is inside the zip (zipped by `create_ladder_zip.py` which zips entire `bot/` dir) — baseline priors are baked into uploads
- Flat prior `[1,1,1,1]` for unknown opponents — no adjustment, same as today
- Race-specific priors keyed by `(opponent_id, enemy_race)` — same opponent can play differently vs P/T/Z
- Minimum games threshold of ≥3 in training script (avoids overfitting to 1-game samples); no threshold at runtime (even 1 game is better than flat)
- `opponent_id` already flows through telemetry (in match context) — no new telemetry fields needed for the data loop
- `opponent_prior_applied: True` added to match-end telemetry for debugging/verification

---

## What Happened to Threat Belief and Engagement Belief

### Threat Belief → Merged into Composition + Strategy

After analyzing the actual code, `assess_threat()` is a **damage estimator** — it answers "how much army value is near our bases?" That's valuable, and it doesn't need to be replaced with a Bayesian model. What it needs is **better inputs**:

- **Composition Belief** provides probability-weighted unit lists → `assess_threat()` gets `army_value * P(exists)` instead of `army_value * 1.0` for stale units
- **Strategy Belief** provides `P(imminent_push)` → this informs whether we should be on alert for incoming threats we haven't seen yet

These two consume into the existing `assess_threat()` and `handle_attack_toggles()`, making threat assessment a **consumer** of other beliefs rather than a standalone model. No separate Threat Belief module needed.

### Engagement Belief → Data Collection + Analysis First

The primary value of Composition Belief for combat is **better inputs to the existing sim**. Currently the sim gets binary-filtered units; with belief, it gets probability-weighted units. This is the Bayesian contribution to combat decisions — better signal going in.

For **calibration** of the sim itself (are our thresholds right?), the right approach is:

1. **Add engagement outcome telemetry** — track whether attacks we initiated actually won or lost
2. **Accumulate data in DuckDB** — `combat` events with `can_win_fight`, supply margins, intel freshness, and actual outcomes
3. **Analyze in Streamlit** — query: "when the sim says VICTORY_MARGINAL with +5 supply, what's our actual win rate?"
4. **Tune thresholds based on data** — if VICTORY_MARGINAL with fresh intel actually wins 85% of the time, adjust the attack threshold accordingly
5. **Add calibration overlay only if thresholds can't fix it** — if the data reveals systematic bias (e.g., the sim consistently overestimates vs Zerg), add a `CalibratedClassifierCV` trained on the accumulated data

Steps 1-4 are telemetry + analysis, not a belief model. Step 5 may never be needed. This replaces the "Engagement Belief" phase with a data collection + analysis phase that's simpler and more grounded.

---

## Library Dependencies

| Library | Version | Status | Used By | Why |
|---------|---------|--------|---------|-----|
| `scipy.stats` | 1.17.1 | **Already installed** (transitive via sklearn) | Phases 1, 3, 4 | Exponential (composition decay), Entropy (VOI), Dirichlet (opponent) |
| `scikit-learn` | 1.8.0 | **Already installed** (ares dep) | Potential future calibration | `CalibratedClassifierCV` if step 5 is needed |
| `numpy` | 2.4.4 | **Already installed** | All phases | Array operations for belief state |
| `pgmpy` | 1.1.2 | **Dev-only dependency** (not runtime) | Phase 2 training | `DiscreteBayesianNetwork`, `VariableElimination`, `MaximumLikelihoodEstimator` — only needed by `scripts/train_strategy_belief.py`; runtime loads pickled model via `joblib` |

**`pgmpy` is no longer a runtime dependency.** It was moved to `[tool.poetry.group.dev.dependencies]` — the training script requires `poetry install` with the dev group, but the runtime bot only needs `joblib` (already installed via scikit-learn) to load the pickled model.

**Libraries explicitly not used**:
- **PyMC / Pyro**: MCMC is too slow for per-frame updates. Useful for offline model analysis only.
- **pomegranate**: Requires PyTorch — too heavy for an SC2 bot 0.5ms frame budget.
- **TensorFlow Probability**: Same — too heavy, wrong use case.

---

## BeliefState Dataclass (Read-Only Snapshot)

```python
@dataclass(frozen=True)
class BeliefState:
    composition: CompositionBelief   # Phase 1
    strategy: StrategyBelief | None = None  # Phase 2 (optional, feature-gated)
    # Phase 3: ScoutVOI — not yet implemented
    # Phase 4: OpponentBelief lives in BeliefUpdater, not BeliefState
    #          (it's a cross-game prior, not a per-frame belief)
```

Each sub-belief exposes probability properties:

```python
# Phase 1: Composition Belief
composition.P_unit_exists(tag: int) -> float           # P(unit still alive | age, type)
composition.get_weighted_army() -> list[WeightedUnit]   # For combat sim and threat assessment
composition.freshness -> float                          # Aggregated (replaces intel freshness)
composition.P_expected_unit(unit_type: UnitTypeId) -> float  # P(unit coming | structures seen)

# Phase 2: Strategy Belief
strategy.P_strategy(s: str) -> float                    # P(strategy = s)
strategy.P_cheese -> float                              # P(any cheese strategy)
strategy.P_timing_attack -> float                        # P(timing attack)
strategy.P_all_in -> float                              # P(all-in)
strategy.P_macro -> float                            # P(macro play)
strategy.label -> str                                   # MAP estimate (backward compat)

# Phase 3: Scout VOI
scout_voi.rank_targets() -> list[tuple[Point2, float]]  # Destination → expected info gain
# Scout TYPE selection (observer/phoenix/worker) stays in existing scouting.py logic

# Phase 4: Opponent Belief (lives in BeliefUpdater, not BeliefState)
# Cross-game prior applied to Strategy Belief at game start
opponent_belief.get_prior(opponent_id, enemy_race) -> dict[StrategyCategory, float]  # Dirichlet alpha params
opponent_belief.update(opponent_id, enemy_race, predicted_category)  # After each game
opponent_belief.load()  # At game start
opponent_belief.save()  # At game end
```

### Integration Into bot.py

```python
# In PiG_Bot.on_step():
self.belief_state = self.belief_updater.update(
    cached_army=self.mediator.get_cached_enemy_army,
    visible_structures=self.enemy_structures,
    game_time=self.time,
    destroyed_units=self._destroyed_enemy_tags,  # from on_unit_destroyed callback
    # ... other observations
)
# self.belief_state is now read-only for all consumers
```

Each consumer accesses what it needs:

```python
# combat.py — the primary combat value: weighted army in combat sim
enemy_army_weighted = bot.belief_state.composition.get_weighted_army()
fight_result = can_win_fight(bot, own_army, enemy_army_weighted)

# reactions.py — threat assessment with probability-weighted units
threat_value = assess_threat(bot, enemy_units, own_forces)
# assess_threat internally uses composition.get_weighted_army() for near-base threat calc

# combat.py — attack decision with strategy-informed risk
if bot.belief_state.strategy.P_timing_attack > 0.6:
    # Be more cautious — opponent may be on a timing push
    attack_threshold = VICTORY_CLOSE
else:
    attack_threshold = TIE_OR_BETTER

# scouting.py — destination selection by VOI (type selection unchanged)
target = bot.belief_state.scout_voi.rank_targets()[0]  # Highest VOI destination
# observer/phoenix/worker selection still uses existing logic
```

---

## Specific Integration Points

| Existing Function | Belief Consumer | Change |
|-------------------|-----------------|-------|
| `bot.py:296-300` | Composition Belief | Replace binary `age < 30s` filter with weighted unit list |
| `bot.py:on_unit_destroyed` | Composition Belief | Remove destroyed units with `P=0.0` immediately |
| `intel/intel_quality.py:get_enemy_intel_quality()` | Composition Belief | Freshness score becomes one component of composition belief |
| `intel/intel_quality.py:update_enemy_intel_tracking()` | Composition Belief | Populate `_enemy_unit_last_seen` dict (ghost field, currently empty) |
| `reactions.py:assess_threat()` | Composition Belief | Use `composition.get_weighted_army()` for near-base threat calculation |
| `reactions.py:threat_detection()` | Strategy Belief | Strategy-aware thresholds via `STRATEGY_THREAT_MULTIPLIER` (Phase A) |
| `combat.py:handle_attack_toggles()` | Composition + Strategy | Weighted army in combat sim; strategy-informed risk thresholds |
| `combat.py:can_win_fight()` | Composition Belief | Receives weighted army instead of binary-filtered army |
| `intel/strategy_detect.py:detect_cheese()` | Strategy Belief | Superseded by `strategy_belief.py` as primary path; kept as fallback |
| `macro.py:strategy_nudge_proportions()` | Strategy Belief | Predictive composition nudging using `STRATEGY_EXPECTED_UNITS` (Phase B) |
| `scouting.py:get_hunt_target()` | Strategy Belief | Strategy-aware hunt target order via `STRATEGY_HUNT_TARGETS` (Phase C) |
| `scouting.py:control_build_runner_scout()` | Strategy Belief | Strategy-aware scout waypoints via `STRATEGY_SCOUT_WAYPOINTS` (Phase D) |
| `scouting.py:get_hunt_target()` | Scout VOI | Rank destinations by information gain (Phase 3, not yet implemented) |
| `scouting.py:control_observers()` | Scout VOI | Prioritize highest-VOI destination (Phase 3, not yet implemented) |

---

## Data Requirements

### Data Already Flowing (Telemetry)

| Belief Model | Telemetry Source | Fields Used |
|-------------|-----------------|-------------|
| Composition | `intel` periodic events + `on_unit_destroyed` callback | `scouted_enemy_units`, `scouted_enemy_structures`, `visible_enemy_count` |
| Strategy | `rush_detect` events + match record (cloud API) | All timing features, `ml_probs`, `ml_confidence`, `cheese_type` |
| Scout VOI | Runs from other beliefs at runtime | No separate telemetry needed |
| Opponent | Match record | `opponent_id`, `result`, `cheese_type`, `length`, `enemy_race`, `map` |

### New Telemetry Events Needed

| Event | Subsystem | Why | Fields |
|-------|-----------|-----|--------|
| Per-unit last-seen | `intel` | Foundation for composition belief decay | Internal state change, not a telemetry event. Populate `_enemy_unit_last_seen` dict. |
| Observer target | `scouting` | Training data for VOI destination effectiveness | `observer_role: str`, `target_type: str`, `target_position: str`, `observation_result: str` (saw_army/saw_structures/saw_nothing/observer_died) |
| Engagement decision | `combat` | Record why bot chose to engage/retreat (future threshold tuning) | `decision: str` (engage/retreat/hold), `reason: str`, `freshness: float`, `own_supply: int` |
| Engagement outcome | `combat` | Ground truth for sim calibration analysis | `engagement_result: str` (win/loss/retreat), `sim_result_category: str`, `own_supply: int`, `enemy_supply_visible: int`, `intel_freshness: float` |

These follow the existing telemetry schema pattern (subsystem/action/reason + flat optional fields). Add to the "Other subsystems — to be defined" section in the telemetry plan.

### Replay Data for Ground Truth

From replays, the bot can get **ground truth** that its own perception cannot provide. The most valuable signals:

| Replay Data | Why It's Valuable | Which Belief Model |
|-------------|-------------------|-------------------|
| **Actual enemy army composition** (exact unit counts at each timestamp) | Ground truth for composition belief — compare `P(unit exists)` estimates against reality. Trains half-life parameters. | Composition Belief |
| **Actual enemy structures** (what they built and when) | Ground truth for "structure → expected units" priors. "If we see a Factory at 5:00, what units did they actually build?" | Composition Belief (priors) |
| **Actual engagement outcomes** (who won each fight, exact unit counts) | Ground truth for combat sim calibration. "Sim said VICTORY_MARGINAL, actual result was LOSS." | Engagement Analysis |
| **Enemy production queues** (what they were building when we had vision) | Trains "expected unseen units from production" priors. | Composition Belief (priors) |
| **Game time of key events** (attack timings, expand timings) | Ground truth for "timing_attack" classification in Strategy Belief. "What percentage of macro openings lead to a push at 7:00?" | Strategy Belief |
| **Actual enemy resource collection rate** | Strongly correlated with strategy (aggressive → lower eco, macro → higher eco). Cannot be observed in-game. | Strategy Belief |
| **Upgrades completed** (when the enemy completed upgrades) | Ground truth for tech transition detection. Complete data vs partial from observers. | Strategy Belief |

**Minimum viable replay data** for belief model training:
1. **Actual enemy army composition at engagement time** — trains composition belief decay rates
2. **Actual engagement outcomes** — enables combat sim calibration analysis
3. **Actual enemy structure build order** — trains structure→unit priors

Everything else is bonus.

### Data Volume Needed for Training

| Model | Minimum Training Games | Source |
|-------|----------------------|--------|
| Composition | 0 (online-only, conjugate priors) | N/A |
| Strategy (Zerg) | 50+ vs Zerg (API has ~143 games total) | Cloud telemetry API |
| Strategy (Terran/Protoss) | 50+ vs each race | Cloud telemetry API (needs more games) |
| Scout VOI | 0 (derived from other beliefs) | N/A |
| Opponent profiles | Accumulates over time (0 min, improves with games) | Match records |

**Critical**: Local `data/` directory is empty — no local JSONL telemetry files yet. However, the **cloud telemetry API** already has 143+ games and is the primary training data source. `data/games/` JSONL files are needed for local Streamlit analysis but not for training (scripts fetch from API). Phase 2 can proceed with training against the API; local DuckDB pipeline is for monitoring, not model training.

---

## Engagement Data Collection (Not a Belief Model)

Instead of a formal Engagement Belief phase, we add telemetry and analyze in Streamlit:

### New Telemetry Events

Add to `combat` subsystem in telemetry plan:

| Action | Category | Fields |
|--------|----------|--------|
| `engage` | Decision | `decision: str`, `reason: str`, `freshness: float`, `own_supply: int` |
| `outcome` | Decision | `engagement_result: str`, `sim_result_category: str`, `own_supply: int`, `enemy_supply_visible: int`, `intel_freshness: float` |

### Analysis in Streamlit

```sql
-- After 200 engagements, what's our actual win rate by sim prediction?
SELECT
    sim_result_category,
    AVG(CASE WHEN engagement_result = 'win' THEN 1.0 ELSE 0.0 END) as actual_win_rate,
    AVG(own_supply - enemy_supply_visible) as avg_supply_margin,
    AVG(intel_freshness) as avg_freshness,
    COUNT(*) as sample_size
FROM engagement_events
GROUP BY sim_result_category
ORDER BY actual_win_rate DESC;
```

This tells you: "when the sim says VICTORY_MARGINAL with +5 supply and 0.8 freshness, we actually win 73% of the time." Use this data to tune `handle_attack_toggles()` thresholds. If thresholds can't fix systematic bias, THEN add a calibration overlay — but start with data.

---

## File Structure

```
bot/
  belief/                          ← belief package
    __init__.py                    ← Re-exports: BeliefState, BeliefUpdater, CompositionBelief, StrategyBelief, StrategyPrediction, OpponentBelief
    belief_state.py                ← BeliefState dataclass (read-only snapshot consumed by decisions)
    belief_updater.py              ← Ingests observations, produces new BeliefState each frame; owns OpponentBelief
    composition_belief.py          ← Phase 1: enemy composition with soft decay + structure priors
    strategy_belief.py             ← Phase 2: multi-race strategy classifier (BN + rule guards)
    opponent_belief.py             ← Phase 4: cross-game opponent style priors (Dirichlet alpha params)
  intel/                           ← observation + classification package (replaces old utilities/intel.py + cheese_detection.py)
    __init__.py                    ← Re-exports: track_enemy_timings, detect_cheese, get_enemy_intel_quality, etc.
    enemy_timings.py               ← Race-agnostic timing observation layer (all races)
    strategy_detect.py             ← Multi-race strategy classification (auto-TRUE guards + rule scoring)
    intel_quality.py               ← Intel freshness/urgency tracking
  models/
    strategy_belief_model.pkl      ← pgmpy BN structure + fitted parameters (Phase 2, loaded via joblib)
    opponent_priors.json           ← Baseline opponent priors from training script (Phase 4)
scripts/
  train_strategy_belief.py         ← Train pgmpy BN + build opponent priors (fetches from cloud telemetry API; requires pgmpy dev dep)
data/
  games/                           ← Per-game JSONL telemetry (created on first write)
  opponent_profiles.json           ← Cross-game opponent profiles (Phase 4, runtime, ladder-accumulated)
```

---

## Performance Guardrails

| Constraint | Implementation |
|-----------|---------------|
| No MCMC or iterative sampling | All updates are conjugate (closed-form). Exponential, Beta, Dirichlet updates are O(1) per update. |
| Per-frame budget: ≤0.5ms for all beliefs | Profile with `PerformanceMonitor`. If over budget, skip non-critical updates (strategy belief only re-queries when evidence changes, not every frame). |
| pgmpy VariableElimination: ≤1ms per query | Networks are sparse (7-10 nodes, max 3-4 states per node). VE on this size is sub-ms. Cache inference results and re-query only when evidence changes (scout reports come at most once per second). |
| Model loading: one-time at game start | `strategy_belief_model.pkl` loaded in `on_start()`. No per-frame I/O. |
| Persistence: ≤1KB per write, ≥30s between writes | Opponent profile written once per game. Telemetry events are existing infrastructure. |
| Unit destruction: O(1) removal | `on_unit_destroyed` callback immediately removes unit from belief state. No decay calculation needed. |
| Competition-safe defaults | If belief models fail to load or produce NaN, fall back to existing heuristics. Feature-gated behind `config.yml` flags, all currently `true` for testing; set to `false` for competition if needed. |

### Feature Flags (config.yml)

```yaml
belief:
  enable_composition: true     # Phase 1
  enable_strategy: true        # Phase 2
  enable_scout_voi: false      # Phase 3
  enable_opponent: true        # Phase 4
```

Each flag currently defaults to `true` except `enable_scout_voi` (Phase 3 not yet implemented). Beliefs are feature-gated behind these flags; when disabled, existing heuristics run unchanged. Flags are documented in config and in the telemetry plan.

---

## Data Pipeline

### Dual Data Sources: Cloud API + Local JSONL

Telemetry data flows through two complementary paths:

```
Game Telemetry (JSONL / stdout TELEM)
          │
          ├──────────────────────────────┐
          ▼                              ▼
   Cloud Telemetry API            Local JSONL files
   (telemetry.pownz.com)         (data/games/*.jsonl)
          │                              │
          ▼                              ▼
   Training scripts              DuckDB (in-memory)
   (fetch via HTTP)              │
                                  ├──→ Streamlit dashboard (monitoring/analysis)
                                  └──→ Local ad-hoc queries
```

**Training path** — `train_strategy_belief.py` fetches from the cloud API (`/api/features/match-level-full`). This endpoint returns 45+ pre-aggregated columns per match (engagement metrics, economy stats, rush detection, strategy labels). The API already has 143+ games. Training scripts do NOT read local JSONL files.

**Local analysis path** — DuckDB reads local `data/games/*.jsonl` files via `read_json_auto()` for Streamlit dashboards, ad-hoc queries, and monitoring. No persisted database file, no import step. This path requires running games locally with telemetry enabled to populate the JSONL files.

**Why two paths?** The cloud API has accumulated games from competition runs; local JSONL captures dev sessions. The API provides pre-aggregated match-level features ideal for training; local JSONL provides raw events ideal for debugging and monitoring.

### Shared View Definitions: `queries/views.py` (Local Analysis Only)

All local analysis (Streamlit dashboards, ad-hoc queries) share common DuckDB view definitions. Training scripts do **NOT** use this — they fetch from the cloud API via `requests`.

```python
# queries/views.py
QUERIES_DIR = Path(__file__).parent

STRATEGY_TRAINING_VIEW = """
CREATE OR REPLACE VIEW strategy_training AS
SELECT
    m.match_id, m.enemy_race, m.map, m.result, m.cheese_type AS label,
    -- ... (full SQL from below)
;
"""

ENGAGEMENT_TRAINING_VIEW = """
CREATE OR REPLACE VIEW engagement_training AS
SELECT
    -- ... (engagement outcome data)
;
"""

def get_connection(games_dir: str = "data/games/*.jsonl"):
    """Connect to DuckDB and create shared views over JSONL files."""
    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE VIEW IF NOT EXISTS events AS SELECT * FROM read_json_auto('{games_dir}')")
    con.execute(STRATEGY_TRAINING_VIEW)
    con.execute(ENGAGEMENT_TRAINING_VIEW)
    return con
```

Training scripts import `get_connection` and query the pre-created views. No `.duckdb` file is stored on disk.

### DuckDB Views (Aligned With Telemetry)

The JSONL files in `data/games/` are queried directly. The relevant views for belief model training:

| View | Purpose | Key JOIN |
|------|---------|----------|
| `match_records` | Game-level summaries (result, cheese_type, opponent_id, map, race) | `match_id` |
| `rush_detect_events` | Strategy timing features (pool_time, gas_time, nat_time, scores, ml_probs) | `match_id` |
| `combat_events` | Engagement snapshots (can_win_fight, role counts, defender composition) | `match_id` |
| `intel_events` | Scouted compositions and structures | `match_id` |
| `reactions_events` | Threat transitions (under_attack, game_phase, cheese detections) | `match_id` |
| `engagement_events` | Engagement outcomes for calibration analysis | `match_id` |

Local analysis scripts create these views via DuckDB SQL (training scripts use the cloud API instead). Example for local strategy analysis:

```sql
CREATE OR REPLACE VIEW strategy_training AS
SELECT
    m.match_id, m.enemy_race, m.map, m.result, m.cheese_type AS label,
    -- scout_report timing features (latest values before classification)
    sr.pool_seen_time,
    sr.extractor_seen_time     AS gas_time,
    sr.enemy_nat_started_at    AS nat_start,
    sr.queen_started_time,
    sr.first_ling_seen_time    AS ling_seen,
    sr.speed_research_time     AS speed_start,
    sr.gas_workers_count,
    sr.nat_present_on_last_scout,
    sr.last_nat_scout_time,
    sr.ling_has_speed,
    -- classification scores (final values)
    cls.score_12p, cls.score_speed, cls.auto_true_fired
FROM match_records m
LEFT JOIN (
    -- Latest scout_report per match — contains all timing observations
    SELECT match_id,
        MAX(pool_seen_time)          AS pool_seen_time,
        MAX(extractor_seen_time)     AS extractor_seen_time,
        MAX(enemy_nat_started_at)    AS enemy_nat_started_at,
        MAX(queen_started_time)      AS queen_started_time,
        MAX(first_ling_seen_time)    AS first_ling_seen_time,
        MAX(first_ling_contact_nat_time) AS first_ling_contact_nat_time,
        MAX(speed_research_time)     AS speed_research_time,
        MAX(gas_workers_count)       AS gas_workers_count,
        MAX(CASE WHEN nat_present_on_last_scout = true THEN 1
                 WHEN nat_present_on_last_scout = false THEN 0 ELSE -1 END) AS nat_present_on_last_scout,
        MAX(last_nat_scout_time)     AS last_nat_scout_time,
        MAX(CASE WHEN ling_has_speed = true THEN 1 ELSE 0 END) AS ling_has_speed
    FROM rush_detect_events
    WHERE action = 'scout_report'
    GROUP BY match_id
) sr ON m.match_id = sr.match_id
LEFT JOIN (
    -- Final classification event per match
    SELECT match_id,
        MAX(score_12p)      AS score_12p,
        MAX(score_speed)    AS score_speed,
        MAX(auto_true_fired) AS auto_true_fired
    FROM rush_detect_events
    WHERE action = 'classification'
    GROUP BY match_id
) cls ON m.match_id = cls.match_id
WHERE m.enemy_race IS NOT NULL;
```

### New: `queries/views.py`

Shared DuckDB view definitions and a connection helper. All training scripts import `get_connection()` which:

1. Creates an in-memory DuckDB connection
2. Reads JSONL files via `read_json_auto('data/games/*.jsonl')`
3. Creates shared views: `events`, `strategy_training`, `engagement_training`
4. Returns the connection for querying

No `.duckdb` file is stored on disk — the JSONL files are the database, DuckDB reads them in-place each time. This means:
- No stale data — always reads the latest JSONL files
- No separate import step — DuckDB discovers schema from the JSONL automatically
- Single source of truth — `queries/views.py` is the only place view definitions live
- Streamlit dashboards and training scripts share the same views

### Broken: `scripts/train_rush_model.py`

The existing training script reads from local `data/rush_detection_log.jsonl` which no longer exists and never had competition-quality data. It needs rewriting to either:
- (a) Use the cloud API (like `train_strategy_belief.py` does) for rush_detect event data
- (b) Use local `queries/views.py` + DuckDB if local JSONL files exist

Option (a) is preferred — the API has accumulated game data already. The rush_detect per-match events endpoint currently returns empty for tested matches (see Data Pipeline Readiness), so option (a) also depends on fixing that endpoint.

### Current: `scripts/train_strategy_belief.py`

1. Fetch match data from cloud API (`/api/features/match-level-full`) via `requests`
2. Batch-fetch rush_detect events for Zerg/Random games (`/api/matches/{id}/events?subsystem=rush_detect`)
3. Derive strategy labels (3-tier: API `strategy_category` → `cheese_type` mapping → game heuristic)
4. Discretize timing features into categorical bins for pgmpy
5. Define BN structure (arcs from SC2 domain knowledge)
6. Fit parameters with `BayesianEstimator` (BDeu prior for sparse data)
7. Validate via `VariableElimination` test query
8. Save model to `bot/models/strategy_belief_model.pkl`

Does NOT use `queries/views.py` or local DuckDB — all data comes from the cloud API.

### To Fix: `scripts/train_rush_model.py`

1. Replace `data/rush_detection_log.jsonl` read with API fetch (or DuckDB if local data exists)
2. Filter to Zerg games (existing approach)
3. Train LogisticRegression (existing approach)
4. Save model to `bot/models/rush_detector_model.pkl`
5. Blocked by: rush_detect per-match events endpoint returning empty (see Data Pipeline Readiness)

### Removed: Intermediate CSV Files and Persistent DuckDB File

No `scripts/aggregate_training_data.py` and no `data/training/*.csv`. No `data/botalytics.duckdb` persisted file. DuckDB reads JSONL files in-place via `read_json_auto()` and creates views in-memory. View definitions live in `queries/views.py`, shared by all consumers (training scripts, Streamlit dashboards, analysis notebooks).

---

## Complexity Budget Assessment

Per AGENTS.md rules: +5 points max per task, refactor exemption for splits >500 LOC.

| Phase | New Files | New Classes | New Deps | API Changes | Cross-Module | Total | Status |
|-------|-----------|-------------|----------|-------------|-------------|-------|--------|
| 1 | +3 | +3 (CompositionBelief, BeliefState, BeliefUpdater) | 0 | +1 (belief_state on bot, on_unit_destroyed) | +2 (combat, reactions, intel) | 5 | Within budget |
| 2 | +3 | +1 (StrategyBelief) | +1 (pgmpy) | 0 | +1 (reactions) | 4 | Within budget (+queries/views.py shared infra) |
| 3 | +1 | +1 (ScoutVOI) | 0 | 0 | +1 (scouting) | 2 | Within budget |
| 4 | +1 | +1 (OpponentBelief) | 0 | 0 | +4 (bot, strategy_belief, belief_updater, game_report, config, train script) | 5 | ✅ Complete |

Each phase is a separate task. Budget resets per phase.

---

## Over-Engineering Triggers (Self-Check)

- ✅ No class for logic with <3 methods and no state — each belief model manages ≥5 related state variables
- ✅ No new config flags beyond the feature gates (which are the existing pattern from `config.yml`)
- ✅ No adapters/interfaces with only one implementation — each belief model has one implementation, no adapter pattern
- ✅ No new dependency replacing ≤10 LOC — pgmpy replaces 300+ LOC of hand-rolled rules in `rush_detection.py` and `reactions.py`
- ✅ No pipelines/state machines where a loop + guard works — belief updates are simple functions called sequentially from `belief_updater.py`
- ✅ Threat assessment is NOT a separate model — it's a consumer of Composition Belief (weighted units) and Strategy Belief (predictive risk), reducing unnecessary abstraction
- ✅ Engagement calibration is data collection → analysis → tuning, not a belief model — avoids premature modeling

---

## Shadow Review

1. **Biggest assumption**: That pgmpy's `VariableElimination` runs fast enough for per-frame inference on our networks (7-10 nodes). It should be — small networks, sparse connectivity, 3-4 states per node. But it must be profiled on the first implementation. Plan: if >0.5ms, cache inference results and only re-query when evidence changes (the evidence only changes when a scout report comes in, which is at most once per second).

2. **Most likely failure/edge**: `data/` is empty locally — no JSONL telemetry files from dev sessions. But the cloud API has 200+ games. The training pipeline (`train_strategy_belief.py`) already fetches from the API, so Phase 2 training can proceed now. The API now has `strategy_category` populated for all 200 matches (cheese:81, macro:62, all_in:31, timing:26). Remaining gaps: (a) the rush_detect per-match events API endpoint returns empty, so Zerg timing features default to -1; (b) API sends `timing` instead of `timing_attack` — training script normalizes this.

3. **Smallest change to improve robustness**: Populate `_enemy_unit_last_seen` (10 LOC in `update_enemy_intel_tracking()`). This is the foundation for composition belief decay and is currently a ghost — declared but never written to. It costs nothing, breaks nothing, and unblocks Phase 1.
---

## Data Pipeline Readiness (2026-06-03)

### Potential Features (Not Yet Used by Training Script)

These fields are available in the API but not yet consumed by `train_strategy_belief.py`. Marked as **potential** because some are post-game aggregates the bot cannot access mid-game.

| Field | Type | Mid-game Available? | Usefulness |
|-------|------|---------------------|------------|
| `avg_collection_rate` | float | ⚠️ Maybe — check if bot's intel module collects per-frame income | **High** — primary SQ driver, could distinguish macro from cheese early |
| `avg_unspent` | float | ⚠️ Maybe — bot tracks `avg_unspent_minerals/vespene` in telemetry | **High** — low unspent = good spending, high unspent = economy collapse |
| `max_squads_atk` / `max_squads_atk_early` | int | No — post-game aggregate | Medium — proxy for aggression level |
| `max_squads_def` / `max_squads_def_early` | int | No — post-game aggregate | Medium — proxy for defensive posture |
| `max_squads_base` | int | No — post-game aggregate | Low |
| `economy_transitions` | int | No — post-game count | Medium — number of eco transitions suggests macro play |
| `critical_events` | float | No — post-game aggregate | Low — ambiguous signal |
| `recovery_events` | float | No — post-game aggregate | Medium — recovery from pressure suggests all_in was repelled |
| `stable_events` | float | No — post-game aggregate | Low — ambiguous |
| `real_sq` / `mineral_sq` / `gas_sq` | float | No — computed from replay snapshots post-game | **Training only** — ground truth for SQ, useful for validation |
| `chosen_opening` | str | No — post-game | Medium — bot's own opening choice |

### What Needs Fixing

1. **rush_detect per-match events endpoint** — Returns empty for tested matches. Without Zerg timing features, the BN can only use `duration_bin`, `rush_conf_bin`, and `enemy_race` to classify strategy. Fix: either (a) ensure the endpoint serves data for matches that have rush_detect events, or (b) add the timing fields to `match-level-full` as pre-aggregated columns (similar to how `avg_12pool_prob` is already included).

2. **`avg_12pool_prob` and `avg_speedling_prob` are NULL for 60% of games** — The rush detection ML model isn't running for all game versions. The `rush_detected` boolean is also NULL for many games. Fix: ensure rush detection runs for all games, or accept that these features are Zerg-only and handle NULLs in the BN.

3. ~~**Strategy label coverage**~~ — **Resolved.** API now has 200 matches with `strategy_category` populated (cheese:81, macro:62, all_in:31, timing:26). Training script normalizes `timing` → `timing_attack`.

4. **`nat_start` and expansion scouting** — Currently hardcoded to -1 in the training script. These are critical features for distinguishing macro (fast expansion) from cheese (no expansion). Fix: add these to the match-level-full endpoint, or populate them from the per-match rush_detect events endpoint once it's fixed.
