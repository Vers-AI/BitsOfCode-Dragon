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
- Integration testing with games (`enable_scout_voi: False` currently; needs live validation after enabling)
- Level-2 routing needs live testing with proxy/cannon/rush games

**Done (previous sessions — Phases A-E)**:
- ✅ Phase A: Strategy-aware threat thresholds — `STRATEGY_THREAT_MULTIPLIER` and `STRATEGY_THREAT_CLEAR_MULTIPLIER` in `threat_detection()` (reactions.py)
- ✅ Phase B: Strategy-aware composition nudging — `strategy_nudge_proportions()` using COUNTER_TABLE derivation with `STRATEGY_EXPECTED_UNITS` (macro.py)
- ✅ Phase C: Strategy-aware hunt targets — `get_strategy_hunt_targets()` modifying `get_hunt_target()` fallback (scouting.py)
- ✅ Phase D: Strategy-aware build runner scout routing — `get_strategy_scout_waypoints()` overriding YAML waypoints (scouting.py)
- ✅ BN model training — trained and saved to `bot/models/strategy_belief_model.pkl` (previous session)
- ✅ Opponent Belief — Dirichlet priors from API, saved to `bot/models/opponent_priors.json` (previous session)

**Done (this session — Phase 3)**:
- ✅ Part 1: Level-2-aware hunt targets — PROXY/CANNON/RUSH tables in `constants.py`, routing in `get_strategy_hunt_targets()` and `get_strategy_scout_waypoints()`
- ✅ Part 2: Staleness × relevance VOI — `bot/belief/scout_voi.py` with `get_voi_destinations()` and `update_location_sightings()`
- ✅ New accessor strings in `_resolve_hunt_accessor()` and `_resolve_scout_waypoint()`
- ✅ VOI fallback in `get_hunt_target()` between strategy tables and default expansion cycle
- ✅ `_location_last_seen` init in `bot.py`, `update_location_sightings()` call per step (guarded by `enable_scout_voi`)
- ✅ Telemetry: periodic `belief/scout_voi` events, match-end staleness snapshot, debug overlay
- ✅ Bug fix: CANNON_RUSH_SCOUT_WAYPOINTS used non-existent ARES strings — fixed to use resolver-compatible strings
- ✅ Bug fix: `update_location_sightings()` was running every frame even when `enable_scout_voi: False` — added config guard

---

### Phase 3: Scout VOI — Smart Destination Selection

**What**: Two-part enhancement to scouting destinations:
1. **Early-game**: Level-2-aware routing — proxy, cannon rush, and rush send scouts to different locations based on the specific cheese type detected
2. **Mid-late-game**: Staleness × relevance ranking — when the army is lost and strategy has converged, rank destinations by how stale they are and how decision-relevant they are

The type of scout (observer, hallucinated phoenix, worker probe) remains determined by stage and availability, as it is today.

**Current gap**: Scouting uses strategy-aware tables for the dominant strategy category (Phase C), but these lump all cheese into one route (`own_fourth → own_third → enemy_nat → enemy_spawn`). Proxy rax, cannon rush, and ling rush need very different scouting patterns. In mid-late game when the army is lost, scouts cycle expansions mechanically with no sense of which destination is most stale or decision-relevant.

**Important clarification**: VOI affects **where** scouts go, not **which type** of scout to use. Observer > hallucinated phoenix > worker probe selection stays in the existing `scouting.py` logic, which correctly prioritizes by capability and availability. VOI only changes the destination ranking within that scout type.

#### Part 1: Level-2-Aware Scouting (Early Game)

Strategy belief produces a `level2` label (e.g., `"proxy_rax"`, `"cannon_rush"`, `"12_pool"`) that distinguishes between cheese subtypes. Each subtype needs a different scouting pattern:

| Level-2 label | What we're looking for | Where to scout |
|---|---|---|
| `proxy_rax`, `proxy_gateway`, `proxy_hatch_spine` | Proxy buildings near our base | Perimeter sweep: own expansions → enemy expansions → back to own → map center |
| `cannon_rush` | Pylons/cannons behind our mineral lines | Our own territory: behind nat and main mineral lines |
| `12_pool`, `speedling`, `bunker_rush`, `worker_rush`, `marauder_push` | Rush coming from their base | Their natural → their main → their third → their main again |

**Proxy hunt targets** trace a square around the map, covering edge positions where proxy buildings are typically placed:

```python
PROXY_HUNT_TARGETS: list[str] = [
    "own_third",      # Up our side
    "own_fourth",
    "own_fifth",
    "own_sixth",
    "enemy_sixth",    # Across the top
    "enemy_fifth",
    "enemy_fourth",
    "enemy_third",    # Down their side
    "own_third",      # Back to start
    "map_center",     # Transit point on the way back
]
```

On small maps, `own_fifth` through `enemy_fifth` resolve to `None` and get filtered, so the path naturally shortens to the available expansions. The square pattern still works — just smaller.

**Cannon rush hunt targets** stay close to our own bases:

```python
CANNON_RUSH_HUNT_TARGETS: list[str] = [
    "own_third",          # Common proxy location near our base
    "own_nat_behind",     # Behind our natural mineral line (ARES get_behind_mineral_positions)
    "own_main_behind",    # Behind our main mineral line (ARES get_behind_mineral_positions)
    "own_third",          # Circle back — may have built while we were elsewhere
]
```

`own_nat_behind` and `own_main_behind` resolve to the center point of `get_behind_mineral_positions()` for our natural and main bases respectively — exactly where pylons and cannons go.

**Rush hunt targets** go straight to their base to confirm no expansion:

```python
RUSH_HUNT_TARGETS: list[str] = [
    "enemy_nat",      # Did they expand? Key question for rush vs macro
    "enemy_spawn",    # Their main — where production is
    "enemy_third",    # Hidden third base (2-base timing)
    "enemy_spawn",    # Loop back — they may have moved units out
]
```

Note: **Map center only appears in proxy targets**. Cannon rush is near our bases, rush is near their bases — neither needs a central transit point.

**Resolution logic** in `get_strategy_hunt_targets()`:
- If `prediction.level2` is proxy-related (`proxy_rax`, `proxy_gateway`, `proxy_hatch_spine`) → use `PROXY_HUNT_TARGETS`
- If `prediction.level2` is `cannon_rush` → use `CANNON_RUSH_HUNT_TARGETS`
- If `prediction.level2` is rush-related (`12_pool`, `speedling`, `bunker_rush`, `worker_rush`, `marauder_push`) → use `RUSH_HUNT_TARGETS`
- If `prediction.label` is `CHEESE` but level2 is unknown → use existing `STRATEGY_HUNT_TARGETS[CHEESE]` as fallback
- ALL_IN, TIMING, MACRO → use existing tables (unchanged)

**New accessor strings** for `_resolve_hunt_accessor()`:

| Accessor string | Resolution |
|---|---|
| `"own_fifth"` | `bot.mediator.get_own_expansions[4][0]` (IndexError → None) |
| `"own_sixth"` | `bot.mediator.get_own_expansions[5][0]` (IndexError → None) |
| `"own_nat_behind"` | `bot.mediator.get_behind_mineral_positions(bot.mediator.get_own_nat)[1]` (center point) |
| `"own_main_behind"` | `bot.mediator.get_behind_mineral_positions(bot.start_location)[1]` (center point) |
| `"enemy_fifth"` | `bot.mediator.get_enemy_expansions[4][0]` (IndexError → None) |
| `"enemy_sixth"` | `bot.mediator.get_enemy_expansions[5][0]` (IndexError → None) |
| `"map_center"` | `bot.game_info.map_center` |

**New accessor strings** for `_resolve_scout_waypoint()` (build runner scouts):

| ARES string | Resolution |
|---|---|
| `"OWN_FIFTH"` | `bot.mediator.get_own_expansions[4][0]` |
| `"OWN_SIXTH"` | `bot.mediator.get_own_expansions[5][0]` |
| `"OWN_NAT_BEHIND"` | `bot.mediator.get_behind_mineral_positions(bot.mediator.get_own_nat)[1]` |
| `"OWN_MAIN_BEHIND"` | `bot.mediator.get_behind_mineral_positions(bot.start_location)[1]` |
| `"ENEMY_FIFTH"` | `bot.mediator.get_enemy_expansions[4][0]` |
| `"ENEMY_SIXTH"` | `bot.mediator.get_enemy_expansions[5][0]` |
| `"MAP_CENTER"` | `bot.game_info.map_center` (already exists) |

**Cannon rush scout waypoints** (build runner):

```python
CANNON_RUSH_SCOUT_WAYPOINTS: list[str] = [
    "OWN_NAT_BEHIND", "OWN_MAIN_BEHIND", "THIRD", "NAT", "RAMP",
]
```

Note: `OWN_THIRD`, `OWN_NAT`, `OWN_RAMP`, `OWN_NAT_HG_SPOT` don't exist in ARES `BuildOrderTargetOptions`. The actual strings that `_resolve_scout_waypoint()` handles are `"THIRD"`, `"NAT"`, `"RAMP"` for own-base waypoints, plus the new `"OWN_NAT_BEHIND"` and `"OWN_MAIN_BEHIND"` for behind-mineral-line positions where cannons actually go.

#### Part 2: Staleness × Relevance Ranking (Mid-Late Game)

When strategy has converged (P(macro) > threshold or army is lost), cycle-expansion scouting is replaced by a relevance-weighted staleness system.

**Candidate locations and relevance weights**:

| Key | Relevance | Why |
|---|---|---|
| `enemy_nat` | 1.5 | Natural resolves "expand or commit" — most decision-relevant |
| `last_army_pos` | 1.4 | Where we last saw them |
| `enemy_third` | 1.3 | Third base = macro confirmation |
| `enemy_main` | 1.0 | Main base, always some value |
| `enemy_fourth` | 0.8 | Late-game relevance only |
| `enemy_ramp` | 0.7 | Ramp crossing = army movement indicator |

**Staleness tracking**: `bot._location_last_seen: dict[str, float]` maps each candidate location key to the game time it was last scouted. Updated once per step by checking if any friendly scout (observer, hallucinated phoenix, worker with SCOUTING or BUILD_RUNNER_SCOUT role) is within `VOI_VISION_RADIUS = 10.0` units of that location. Never-scouted locations get maximum staleness (`bot.time`).

**VOI score**: `staleness × relevance` for each candidate. Sorted descending. Most-stale, most-relevant location goes first. This naturally rotates scouts across locations over time.

**Composition uncertainty bonus** (secondary signal): Small additive bonus (`SCOUT_VOI_COMPOSITION_BONUS = 0.3`) to locations that would confirm expected-but-unseen unit types. If composition belief expects Immortals (from structure priors) but hasn't confirmed them, `enemy_nat` and `enemy_main` get a small boost because production structures are usually there.

**Integration in `get_hunt_target()`**:

```python
# Fallback: no cached army
strategy_targets = get_strategy_hunt_targets(bot)  # Level-2 aware (Part 1)
if strategy_targets:
    hunt_targets = strategy_targets  # Early game / cheese
else:
    voi_targets = get_voi_destinations(bot)  # Mid-late game, army lost (Part 2)
    if voi_targets:
        hunt_targets = voi_targets
    else:
        # Default: cycle enemy expansions 4th → 3rd → nat → main
        hunt_targets = [enemy_fourth, enemy_third, enemy_nat, enemy_spawn]
```

**Minimum-length guard**: If Level-2 routing produces fewer than 2 valid positions (e.g., small map where most expansions resolve to None), fall back to the existing `STRATEGY_HUNT_TARGETS[dominant]` table.

**Feature flag**: `config.yml` → `Belief.enable_scout_voi: false` (Phase 3 Part 2 only). Level-2 routing (Part 1) is gated by `Belief.enable_strategy` since it depends on strategy belief's Level-2 labels. When `enable_scout_voi` is disabled, `update_location_sightings()` does not run and `get_voi_destinations()` returns None, falling back to default expansion cycling.

**No new runtime dependencies.** No `scipy.stats.entropy` — relevance weights are fixed constants, staleness is game-time tracking.

**Decisions affected**:
- `get_strategy_hunt_targets()` → Level-2-aware routing with PROXY/CANNON/RUSH tables (Part 1)
- `get_strategy_scout_waypoints()` → Level-2-aware build runner scout routing for cannon rush (Part 1)
- `get_hunt_target()` → VOI fallback between strategy tables and default expansion cycle (Part 2)

**Files to create**:
- `bot/belief/scout_voi.py` — staleness × relevance ranking and location tracking (Part 2 only; ~100 LOC)

**Files to modify**:
- `bot/managers/scouting.py` — `get_strategy_hunt_targets()` enhanced with Level-2 routing; `get_strategy_scout_waypoints()` enhanced with Level-2 routing for cannon rush; `get_hunt_target()` adds VOI fallback; `_resolve_hunt_accessor()` adds 8 new accessor strings; `_resolve_scout_waypoint()` adds 6 new ARES strings
- `bot/constants.py` — Add PROXY_HUNT_TARGETS, CANNON_RUSH_HUNT_TARGETS, RUSH_HUNT_TARGETS, PROXY_LABELS, CANNON_LABELS, RUSH_LABELS, CANNON_RUSH_SCOUT_WAYPOINTS, VOI_MIN_HUNT_TARGETS (strategy section); SCOUT_VOI_RELEVANCE, VOI_VISION_RADIUS, SCOUT_VOI_COMPOSITION_BONUS (VOI section)
- `bot/bot.py` — Init `_location_last_seen = {}`, call `update_location_sightings(self)` in `on_step` (guarded by `enable_scout_voi` config)
- `bot/belief/__init__.py` — Export `get_voi_destinations`, `update_location_sightings`
- `bot/utilities/game_report.py` — Scout VOI staleness telemetry in periodic + match-end reports (gated by `enable_scout_voi`)
- `bot/utilities/debug.py` — Scout VOI staleness overlay line (gated by `enable_scout_voi`)
- `config.yml` — Add `enable_scout_voi: false` under `Belief:`

**Depends on**: Phase 2 (Strategy Belief) for Level-2 labels. Phase 1 (Composition Belief) for staleness × relevance when strategy has converged.

**LOC estimate**: ~100 in `bot/belief/scout_voi.py` (VOI only), ~50 in `bot/managers/scouting.py` (Level-2 routing + VOI fallback + new accessors), ~30 in `bot/constants.py`, ~5 in `bot/bot.py`, ~2 in `bot/belief/__init__.py`, ~15 in `bot/utilities/game_report.py`, ~10 in `bot/utilities/debug.py`. **Total: ~210 LOC**.

#### Implementation Status (Complete ✅)

**Architecture**: Two separate modules. Level-2 routing (Part 1) is inline in `scouting.py` — it reads `prediction.level2` and selects from constant lookup tables (PROXY/CANNON/RUSH). Staleness × relevance (Part 2) is in `bot/belief/scout_voi.py` — it tracks `_location_last_seen` and scores locations by `staleness × relevance`. Both are feature-gated: Level-2 by `enable_strategy`, VOI by `enable_scout_voi`.

**Files created**:
- `bot/belief/scout_voi.py` — `get_voi_destinations()` (staleness × relevance ranking), `update_location_sightings()` (per-step scout proximity check). ~100 LOC.

**Files modified**:
- `bot/managers/scouting.py` — `get_strategy_hunt_targets()` now inlines Level-2 routing (PROXY/CANNON/RUSH label matching before category fallback); `get_strategy_scout_waypoints()` now inlines Level-2 routing for cannon rush; `get_hunt_target()` adds VOI fallback between strategy targets and default expansion cycle; `_resolve_hunt_accessor()` adds 8 new accessors (own_fifth, own_sixth, own_nat_behind, own_main_behind, enemy_fifth, enemy_sixth, enemy_ramp, enemy_main, map_center); `_resolve_scout_waypoint()` adds 6 new ARES strings (OWN_NAT_BEHIND, OWN_MAIN_BEHIND, ENEMY_FIFTH, ENEMY_SIXTH, FIFTH, SIXTH already existed)
- `bot/constants.py` — Added PROXY_HUNT_TARGETS, CANNON_RUSH_HUNT_TARGETS, RUSH_HUNT_TARGETS, PROXY_LABELS, CANNON_LABELS, RUSH_LABELS, CANNON_RUSH_SCOUT_WAYPOINTS, VOI_MIN_HUNT_TARGETS (strategy section); SCOUT_VOI_RELEVANCE, VOI_VISION_RADIUS, SCOUT_VOI_COMPOSITION_BONUS (VOI section)
- `bot/bot.py` — `_location_last_seen: dict[str, float] = {}` init; `update_location_sightings(self)` call in `on_step` guarded by `enable_scout_voi`
- `bot/belief/__init__.py` — Exports `get_voi_destinations`, `update_location_sightings`
- `bot/utilities/game_report.py` — Periodic `belief/scout_voi` telemetry event + console staleness print + match-end `last_seen_{key}_final` snapshot
- `bot/utilities/debug.py` — VOI staleness overlay line showing top-5 locations
- `config.yml` — Added `enable_scout_voi: false` under `Belief:`

**Design decisions**:
- Level-2 routing (PROXY/CANNON/RUSH tables) lives in `scouting.py`, not `scout_voi.py` — it's strategy logic, not VOI logic
- `scout_voi.py` only contains staleness × relevance (Part 2) — clean separation of concerns
- `update_location_sightings()` is guarded by `enable_scout_voi` config flag — no per-frame overhead when disabled
- `CANNON_RUSH_SCOUT_WAYPOINTS` uses ARES-compatible strings that `_resolve_scout_waypoint()` actually handles: `"OWN_NAT_BEHIND"`, `"OWN_MAIN_BEHIND"`, `"THIRD"`, `"NAT"`, `"RAMP"` — not non-existent strings like `"OWN_THIRD"`, `"OWN_NAT"`
- `enemy_main` accessor is an alias for `enemy_spawn` in `_resolve_hunt_accessor()` — SCOUT_VOI_RELEVANCE uses `enemy_main` as key
- Map center only appears in PROXY_HUNT_TARGETS — cannon rush and rush don't need it

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
| `scipy.stats` | 1.17.1 | **Already installed** (transitive via sklearn) | Phases 1, 4 | Exponential (composition decay), Dirichlet (opponent) |
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
    # Phase 3: ScoutVOI — staleness tracking lives in bot._location_last_seen
    #          get_voi_destinations() and update_location_sightings() are standalone functions in scout_voi.py
    #          Level-2 routing (PROXY/CANNON/RUSH) lives in scouting.py, not here
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
get_voi_destinations(bot) -> list[tuple[Point2, float]]  # Staleness × relevance ranked destinations
update_location_sightings(bot) -> None  # Update _location_last_seen per step
# Level-2 routing: get_strategy_hunt_targets() uses PROXY/CANNON/RUSH tables

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

# scouting.py — destination selection by staleness × relevance (Phase 3)
targets = get_voi_destinations(bot)  # Returns [(Point2, score), ...] sorted by VOI
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
| `scouting.py:get_hunt_target()` | Scout VOI | Level-2 routing for cheese (Part 1); VOI fallback for mid-late game (Part 2) |
| `scouting.py:get_strategy_hunt_targets()` | Strategy Belief + Level-2 | Level-2-aware PROXY/CANNON/RUSH routing (Part 1, inline) |
| `scouting.py:get_strategy_scout_waypoints()` | Strategy Belief + Level-2 | Level-2-aware build runner scout routing for cannon rush (Part 1, inline) |
| `scouting.py:_resolve_hunt_accessor()` | Scout VOI | New accessors: own_fifth, own_sixth, own_nat_behind, own_main_behind, enemy_fifth, enemy_sixth, map_center, enemy_ramp, enemy_main |
| `scouting.py:_resolve_scout_waypoint()` | Scout VOI | New ARES strings: OWN_NAT_BEHIND, OWN_MAIN_BEHIND, ENEMY_FIFTH, ENEMY_SIXTH |
| `bot/utilities/game_report.py` | Scout VOI | Periodic staleness telemetry + match-end snapshot (gated by enable_scout_voi) |
| `bot/utilities/debug.py` | Scout VOI | Debug overlay line showing staleness per location (gated by enable_scout_voi) |

---

## Data Requirements

### Data Already Flowing (Telemetry)

| Belief Model | Telemetry Source | Fields Used |
|-------------|-----------------|-------------|
| Composition | `intel` periodic events + `on_unit_destroyed` callback | `scouted_enemy_units`, `scouted_enemy_structures`, `visible_enemy_count` |
| Strategy | `rush_detect` events + match record (cloud API) | All timing features, `ml_probs`, `ml_confidence`, `cheese_type` |
| Scout VOI | Composition Belief + Strategy Belief state at runtime + `bot._location_last_seen` staleness dict | Periodic telemetry (`belief/scout_voi`), match-end snapshot (`last_seen_{key}_final`), debug overlay |
| Opponent | Match record | `opponent_id`, `result`, `cheese_type`, `length`, `enemy_race`, `map` |

### New Telemetry Events Needed

| Event | Subsystem | Why | Fields |
|-------|-----------|-----|--------|
| Per-unit last-seen | `intel` | Foundation for composition belief decay | Internal state change, not a telemetry event. Populate `_enemy_unit_last_seen` dict. |
| Scout VOI staleness | `belief` | Track which locations are stale and how scouts rotate across them | `last_seen_{key}` for each candidate location (periodic, gated by `enable_scout_voi`) |
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
| 3 | +1 | 0 (functions, no class) | 0 | 0 | +2 (scouting, constants) | 3 | Within budget |
| 4 | +1 | +1 (OpponentBelief) | 0 | 0 | +4 (bot, strategy_belief, belief_updater, game_report, config, train script) | 5 | ✅ Complete |

Each phase is a separate task. Budget resets per phase.

---

## Over-Engineering Triggers (Self-Check)

- ✅ No class for logic with <3 methods and no state — each belief model manages ≥5 related state variables; Scout VOI uses functions, not a class
- ✅ No new config flags beyond the feature gates (which are the existing pattern from `config.yml`)
- ✅ No adapters/interfaces with only one implementation — each belief model has one implementation, no adapter pattern
- ✅ No new dependency replacing ≤10 LOC — pgmpy replaces 300+ LOC of hand-rolled rules in `rush_detection.py` and `reactions.py`; Phase 3 uses no new deps (no scipy.stats.entropy, just staleness × relevance with fixed weights)
- ✅ No pipelines/state machines where a loop + guard works — belief updates are simple functions called sequentially from `belief_updater.py`
- ✅ Threat assessment is NOT a separate model — it's a consumer of Composition Belief (weighted units) and Strategy Belief (predictive risk), reducing unnecessary abstraction
- ✅ Engagement calibration is data collection → analysis → tuning, not a belief model — avoids premature modeling

---

## Shadow Review

1. **Biggest assumption**: That pgmpy's `VariableElimination` runs fast enough for per-frame inference on our networks (7-10 nodes). It should be — small networks, sparse connectivity, 3-4 states per node. But it must be profiled on the first implementation. Plan: if >0.5ms, cache inference results and only re-query when evidence changes (the evidence only changes when a scout report comes in, which is at most once per second).

2. **Most likely failure/edge**: On small maps (e.g., Acropolis), `own_fifth` through `enemy_fifth` resolve to `None` in the proxy hunt target list, shrinking the perimeter sweep. The `VOI_MIN_HUNT_TARGETS = 2` guard catches the worst case. Also, `get_behind_mineral_positions()` returns ≥3 points — unusual mineral layouts could return fewer, but the `len(behind) > 1` guard handles that.

3. **Smallest change to improve robustness**: `CANNON_RUSH_SCOUT_WAYPOINTS` originally used non-existent ARES strings (`OWN_THIRD`, `OWN_NAT`, `OWN_RAMP`, `OWN_NAT_HG_SPOT`) — fixed to use strings the resolver actually handles (`OWN_NAT_BEHIND`, `OWN_MAIN_BEHIND`, `THIRD`, `NAT`, `RAMP`). `update_location_sightings()` was running every frame even when `enable_scout_voi: False` — fixed with a config guard.
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

---

## Phase 5: BN Model Improvements — Position & Timing Features

**What**: Add position-aware and timing-resolution features to the BN model to improve classification accuracy. The current naive Bayes model with 7 coarse variables cannot distinguish proxy production from standard play when the bot only observes "1 barracks, 1 base" — it needs to know WHERE the barracks is and HOW EARLY relative to standard timings.

**Problem**: Match 4829911 (PiG_Bot vs Hestia) exposed the core weakness. Hestia played standard mech Terran — barracks at 52s at their own base. The BN saw `Terran + rax=few + bases=one + factory=no + duration=medium` and classified `cheese/proxy_rax`. This is a reasonable guess with that information — 1 rax on 1 base IS what proxy rax looks like from limited intel. The BN simply doesn't have the features to tell the difference.

**Root cause**: The BN's 7 parent variables are all coarse existence/quantity features. No position data. No fine-grained timing. The model has ~3,456 CPD entries — tiny, but also unable to discriminate between superficially similar situations.

### Available Container Dependencies

Checked inside `aiarena/arenaclient-bot:v0.8.0` (the AI Arena bot container):

| Library | Version | Available? |
|---------|---------|------------|
| numpy | 2.0.2 | ✅ |
| scipy | 1.17.0 | ✅ |
| scikit-learn | 1.8.0 | ✅ |
| pandas | 2.3.3 | ✅ |
| torch (CPU) | 2.10.0+cpu | ✅ |
| tensorflow | 2.18.0 | ✅ |
| joblib | 1.5.3 | ✅ |
| pgmpy | — | ❌ NOT installed |
| xgboost | — | ❌ NOT installed |
| lightgbm | — | ❌ NOT installed |
| onnxruntime | — | ❌ NOT installed |

**Key insight**: scikit-learn 1.8.0 is available in the container. This means we can use any sklearn classifier (RandomForest, GradientBoosting, LogisticRegression, MLPClassifier) at runtime with zero new dependencies. pgmpy is NOT in the container, but we already solved that — runtime uses numpy .npz lookup, pgmpy is dev-only for training.

### Step 1: Add Position-Aware Features to BN

The bot already tracks proxy detection booleans in `strategy_detect.py`:
- `_barracks_near_our_base` (Terran)
- `_gateway_near_our_base` (Protoss)
- `_forge_near_our_base` / cannon near base (Protoss cannon rush)
- `_bunker_near_base` (Terran bunker rush)

These are set by `enemy_timings.py` but NOT fed into the BN as evidence variables. They need to flow from observation → BN evidence.

**New BN parent variables** (added to the 7 existing):

| Variable | States | Source | Why |
|---------|--------|--------|-----|
| `rax_near_base` | `yes / no / unknown` | `_barracks_near_our_base` | Proxy rax vs standard rax — THE missing feature for 4829911 |
| `gw_near_base` | `yes / no / unknown` | `_gateway_near_our_base` | Proxy gateway vs standard |
| `cannon_near_base` | `yes / no / unknown` | cannon rush detection | Cannon rush vs standard forge |
| `bunker_near_base` | `yes / no / unknown` | `_bunker_near_base` | Bunker rush vs standard |

These 4 variables would expand the CPD from ~3,456 to ~3,456 × 3^4 = ~279,936 entries. This is still small enough for numpy lookup. Training data needs to be sufficient — with 200+ games, each cell gets ~0-5 observations. Smoothing (Dirichlet alpha=1) handles sparse cells.

**Files to modify**:
- `bot/belief/strategy_belief.py` — Add 4 new evidence variables to `_collect_evidence()`
- `bot/belief/bn_inference.py` — Add 4 new parent variables to `_PARENT_VARS` and `_EXPECTED_STATES`
- `bot/intel/enemy_timings.py` — Ensure proxy booleans are set (most already are; verify cannon/bunker tracking)
- `scripts/train_strategy_belief.py` — Add 4 new features to `build_training_data()` and `discretize_features()`
- `scripts/export_bn_model.py` — Update export for new CPD shape

### Step 2: Add Timing-Resolution Features

The current BN bins timing into coarse categories. `rax_bin` is `none/few/many` (quantity, not timing). A barracks at 20s (proxy) and 52s (standard) both produce `rax_bin=few`. Adding timing bins discriminates these.

**New BN parent variables**:

| Variable | States | Source | Thresholds |
|---------|--------|--------|------------|
| `rax_timing` | `very_early / early / standard / late / none` | `_barracks_seen_time` | <25s=very_early (proxy), 25-45s=early (cheese), 45-90s=standard, >90s=late |
| `pool_timing` | `very_early / early / standard / late / none` | `_pool_seen_time` | <25s=very_early (12-pool), 25-40s=early (speedling), 40-80s=standard, >80s=late |
| `gw_timing` | `very_early / early / standard / late / none` | `_gateway_seen_time` | <20s=very_early (proxy), 20-40s=early, 40-80s=standard, >80s=late |
| `nat_timing` | `very_early / early / standard / late / none` | `_enemy_nat_started_at` | <60s=very_early (greedy), 60-120s=early (standard macro), 120-240s=late (all-in), >240s=none (all-in) |

**Threshold rationale**: Derived from Spawning Tool data and community build order analysis. A Terran barracks at 20s requires cutting workers — only viable as proxy. At 52s it's a standard 1-rax FE. The timing IS the signal.

With 4 position + 4 timing features added to the existing 7, the CPD grows to ~4 × 3^4 × 5^4 × (existing) — approximately 4M entries. This is large for numpy but still feasible (4M floats × 8 bytes = 32MB). If memory is a concern, we can:
- Use float32 instead of float64 (halves memory)
- Drop low-value variables (e.g., `bunker_near_base` only matters for Terran, could be conditional)
- Use a sparse representation

**Alternative**: If the CPD gets too large for naive Bayes, switch to a sklearn classifier (see Step 3).

### Step 3: Consider Upgrading from Naive Bayes

Naive Bayes assumes all features are independent given the class. This is wrong for SC2 — `rax_near_base` and `rax_timing` are correlated (proxy rax is both near AND very early). Naive Bayes double-counts this evidence.

**Option A: Keep naive Bayes with more features** (current architecture, just bigger)
- Pro: No code changes to inference path, just bigger CPD
- Pro: Still numpy lookup, no dependencies
- Con: Double-counts correlated evidence, may over/under-confidence
- Verdict: Good enough if we accept some miscalibration

**Option B: Switch to sklearn classifier** (scikit-learn 1.8.0 already in container)
- RandomForestClassifier or GradientBoostingClassifier
- Handles feature interactions natively
- Export as joblib pickle (joblib 1.5.3 in container)
- Replace `BNInference` class with `SklearnInference`
- Pro: Better accuracy, handles correlations, no independence assumption
- Pro: Feature importance scores for debugging
- Con: Different model architecture — need to rewrite inference path
- Con: No interpretability (RF is a black box vs BN's transparent CPDs)
- Verdict: Better long-term, but bigger change

**Option C: Small neural network** (torch 2.10.0 CPU in container)
- 7+8 input features → 16 hidden → 4 output (strategy categories)
- Train with PyTorch, export to TorchScript
- Pro: Captures any interaction pattern
- Pro: TorchScript is fast at runtime
- Con: Overkill for 4-class problem with <1000 training samples
- Con: More complex training pipeline, hyperparameter tuning
- Verdict: Only if data grows to 2000+ games and simpler models plateau

**Recommendation**: Start with Option A (more features in naive Bayes) for immediate improvement. If accuracy plateaus or double-counting causes issues, move to Option B (sklearn). Option C is future work.

### ~~Step 4: Mismatch Detector~~ — DONE (Implemented in telemetry API)

Implemented as two new API endpoints in `telemetry/pigbot/api.py`:
- `GET /api/mismatches` — returns matches where `bot_strategy_category != strategy_category`, with `mismatch_type` classification (`false_positive`, `false_negative`, `other_mismatch`), filterable by `false_positive_only` and `false_negative_only` query params
- `GET /api/mismatches/summary` — aggregated counts by `(bot_label, replay_label, enemy_race)`

**Key findings from initial data**:
- 526 total mismatches across ~700 matches with both labels populated
- 127 false positives (bot says cheese/all_in, replay says macro/timing) — dominated by `cannon_rush` and `proxy_gateway` false positives vs Terran mech
- 294 false negatives (bot says macro, replay says cheese/all_in/timing) — largest category, suggests the BN is under-classifying aggression
- Most common FP pattern: bot labels Terran mech as `cheese/cannon_rush` or `cheese/proxy_gateway` — same class of error as match 4829911

### Step 5: Training Data Quality

Several gaps in training data need closing:

1. **Missing telemetry files**: Match 4829911 had no `games/4829911.jsonl` because AI Arena didn't have a match log. The replay was processed but the match doesn't appear in the enriched endpoint (which joins telemetry + replay). Fix: ensure the puller downloads match logs even when they're not in the standard participation endpoint — try the result object's `arenaclient_log` URL as fallback.

2. **`nat_start` and `last_nat_scout_time`**: Hardcoded to -1 in training script. These are critical for distinguishing macro (fast expansion) from cheese (no expansion). Fix: add these to `match-level-full` endpoint as pre-aggregated columns from the match record.

3. **`derive_strategy_label()` proxy_rax heuristic**: Currently flags ANY Terran with 1 barracks + 1 base + rax < 180s as cheese, without checking position. This is the training-side equivalent of the bug we found. Fix: add position check — if `rax_near_base` is available in the training data, only flag as cheese when `rax_near_base = yes`.

### Implementation Order

| Step | Effort | Impact | Priority |
|------|--------|--------|----------|
| ~~4: Mismatch detector~~ | ~~Low~~ | ~~High~~ | ~~DONE~~ |
| 1: Position features in BN | Medium (4 new vars, update evidence collection + training) | High — directly fixes 4829911 class of errors | 1 |
| 2: Timing features in BN | Medium (4 new vars, new binning logic) | High — discriminates proxy timing from standard | 2 |
| 3: sklearn upgrade | High (rewrite inference path) | Medium — better but bigger change | 3 |
| 5: Training data quality | Medium (puller fix + endpoint update) | Medium — closes data gaps | 4 |

Steps 1 and 2 can be done together in a single training cycle. Step 4 (mismatch detector) is independent and can be done in parallel.

### Constraints

- **AI Arena container**: scikit-learn 1.8.0, numpy 2.0.2, scipy 1.17.0, torch 2.10.0 CPU all available. No new dependencies needed for Steps 1-4. Step 3 (sklearn upgrade) uses existing sklearn.
- **Runtime performance**: BN lookup stays sub-ms even with 15 parent variables (numpy fancy indexing). sklearn predict() on a RandomForest with 100 trees is ~0.1ms. Both well within the 0.5ms frame budget.
- **Training data volume**: Current API has ~200 games with strategy_category labels. Position features (rax_near_base etc.) need to be added to the training data extraction — they exist in the bot's telemetry events but may not be in the enriched endpoint yet. Need to add them to the event extraction in `build_training_data()`.
- **Competition safety**: All changes feature-gated. BN model falls back to existing 7-variable model if new model file is missing. Auto-TRUE guards remain available as fallback (even though currently disabled by config).

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Level-1 accuracy (bot vs replay) | Unknown — need mismatch detector | ≥85% after Step 1+2 |
| Cheese false positive rate (macro games tagged cheese) | Unknown — 4829911 is one known case | <10% after Step 1+2 |
| Cheese false negative rate (cheese games tagged macro) | Unknown | <15% after Step 1+2 |
| Mismatch count per 100 games | Unknown | <15 after Step 1+2, <8 after Step 3 |
| BN confidence on correct predictions | Unknown | >0.7 average |

### Connection to Auto-TRUE Guards

The auto-TRUE guards are currently disabled by config (`config.yml`). The intent is to train the BN to be the sole arbiter. The path to re-enabling guards (or not) depends on BN accuracy:

1. **After Steps 1+2**: Measure BN accuracy with position+timing features. If >90% on cheese detection, guards are redundant for cheese. Keep disabled.
2. **After Step 3 (if needed)**: If sklearn model hits >92% overall, guards can be permanently removed from the codebase.
3. **If BN plateaus <85%**: Re-enable guards for the specific scenarios where BN fails. Guards become targeted overrides, not blanket fallbacks.

The end state: BN as the sole strategy classifier, no deterministic guards needed. Whether that's achievable depends on data volume and feature quality. Steps 1-3 are the path to finding out.
