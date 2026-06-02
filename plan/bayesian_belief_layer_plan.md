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

TELEMETRY ──→ DuckDB ──→ Streamlit (monitoring)
                          └──→ Training scripts (model fitting)

REPLAY DATA ──→ DuckDB ──→ Ground truth for calibration
```

A **BeliefState** object sits between the observation pipeline and decision-making code. It maintains probability distributions over things the bot cannot directly observe. Decisions consume beliefs, not raw cached snapshots.

### Key Principle: Conjugate Updates Only

All belief updates use **closed-form conjugate priors** (Beta-Binomial, Dirichlet-Multinomial, Exponential decay). No MCMC, no iterative sampling, no matrix inversion. Each update is O(1) or O(n) where n = number of unit types/tags. Target: ≤0.5ms per frame for all belief updates combined.

### Key Principle: Threat Assessment Is a Consumer, Not a Model

`assess_threat()` is a damage estimator — it answers "how much force is near our bases?" Bayesian reasoning improves it by feeding probability-weighted unit lists (Composition Belief) and predictive strategy priors (Strategy Belief) into the existing function, not by replacing it with a separate Threat Belief model. See Phase 1 for details.

---

## The 4 Phases

### Phase 1: Composition Belief — Enemy Army Probability

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
- `bot/utilities/intel.py` — populate `_enemy_unit_last_seen` dict (ghost field, currently empty), wire composition belief into `get_enemy_intel_quality()`
- `bot/managers/reactions.py` — `assess_threat()` uses `composition.get_weighted_army()` for near-base threat calculation
- `bot/combat/combat.py` — pass weighted army to `can_win_fight()` and `handle_attack_toggles()`
- `bot/constants.py` — add `UNIT_HALF_LIFE` constant dict

**No new dependencies.** Uses `scipy.stats.expon` only.

**LOC estimate**: ~150 in `bot/belief/`, ~30 in integration points

**Validation**: Side-by-side comparison — emit composition belief freshness alongside existing intel freshness to telemetry. Confirm they agree on known cases (fresh units, stale units, edge cases).

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

**Data source**: Match record `cheese_type` (ground truth label), `rush_detect` events (features), `reactions` events (timing features for Terran/Protoss). All accessed via DuckDB.

**Decisions affected**:
- `early_threat_sensor()` → consumes `P(strategy = cheese_class)` instead of per-race booleans
- `cheese_reaction()` → threshold on `P_cheese > 0.6` instead of `if cannon_rush`
- `macro.py` → nudge composition based on `P(timing_attack)` vs `P(macro)`
- `_under_attack` flag → Strategy Belief informs whether a near-base threat is likely a committed push or just a probe

**Files to create**:
- `bot/belief/strategy_belief.py`
- `scripts/train_strategy_belief.py`

**Files to modify**:
- `bot/managers/reactions.py` — `early_threat_sensor()` consumes `P_cheese` instead of per-race booleans
- `bot/utilities/rush_detection.py` — kept as fallback, `get_enemy_ling_rushed_v2()` becomes optional path
- `bot/utilities/intel.py` — `get_enemy_cannon_rushed()` becomes optional path

**New dependency**: `poetry add pgmpy` (~2.4MB pure Python wheel)

**Data prerequisite**: Need 50+ games vs each race. Must:
1. Create `data/games/` directory (handled by telemetry on first write)
2. Run 50+ games vs Zerg, 50+ vs Terran, 50+ vs Protoss
3. Ingest telemetry JSONL into DuckDB (existing pipeline)
4. Verify data quality via Streamlit dashboard (class balance, feature distributions)
5. Run `train_strategy_belief.py` (queries DuckDB directly) to fit BN parameters
6. Save model to `bot/models/strategy_belief_model.pkl`

**Backward compatibility**: Existing `rush_detector_model.pkl` is kept. Strategy belief model falls back to rule-based guards (auto-TRUE) if the BN model file is missing. No regression if the model isn't ready.

**LOC estimate**: ~180 in `bot/belief/strategy_belief.py`, ~50 in training script, ~30 in integration points

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

**Model**: Simple Dirichlet concentration parameters per opponent. The categories match Strategy Belief's parent categories (not a separate taxonomy):

```
Opponent Belief categories → Strategy Belief priors:
├── P(cheesy)     → inflates P(cheese) in Strategy Belief
├── P(aggressive) → inflates P(timing_attack) and P(all_in)
└── P(macro)      → inflates P(macro)
```

Updated after each game from match record (result, cheese_type, game length, economy metrics). Prior for unknown opponent: uninformative `[1, 1, 1]`. Persists to `data/opponent_profiles.json`.

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

**Library**: `scipy.stats.dirichlet` (already installed)

**Data source**: Match records (`result`, `cheese_type`, `length`, `opponent_id`, `enemy_race`). ARES DataManager already writes per-opponent JSON to `data/<opponent_id>-<race>.json`.

**Decisions affected**:
- Strategy Belief prior → known cheesy opponent inflates `P(cheese)`
- Composition Belief prior → known aggressive opponent inflates `P(combat_units over workers)`
- Scouting priority → unknown opponent gets more early scouts (higher VOI for information)

**Files to create**:
- `bot/belief/opponent_belief.py`
- `data/opponent_profiles.json` (created on first write, schema-versioned)

**Files to modify**:
- `bot/bot.py` — load opponent profile at game start, save at game end
- `bot/belief/strategy_belief.py` — adjust prior based on opponent profile

**No new dependencies.** Uses `scipy.stats.dirichlet`.

**Persistence**: `data/opponent_profiles.json` with schema versioning, rolling cap (max 200 opponents), safe fallback on corruption.

**Competition safety**: Opponent belief defaults to uninformative prior `[1, 1, 1]` if no data or opponent is unknown. No per-frame I/O. Profile is loaded once at game start and written once at game end.

**LOC estimate**: ~80 in `bot/belief/opponent_belief.py`, ~40 for persistence + validation

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
| `pgmpy` | 1.1.2 | **New dependency** | Phase 2 | `DiscreteBayesianNetwork`, `VariableElimination`, `MaximumLikelihoodEstimator` |

**`pgmpy` is the only new runtime dependency.** It is pure Python (~2.4MB wheel), requires Python ≥3.10 (compatible with our ≥3.11 constraint), has no heavy transitive deps beyond numpy/scipy which we already have, and is sklearn-compatible. Variable elimination on our small networks (7-10 nodes) runs in <1ms.

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
    strategy: StrategyBelief         # Phase 2
    scout_voi: ScoutVOI              # Phase 3
    opponent: OpponentBelief         # Phase 4
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

# Phase 4: Opponent Belief
opponent.P_cheesy -> float                              # P(opponent tends to cheese)
opponent.P_aggressive -> float                           # P(opponent tends to be aggressive)
opponent.P_macro -> float                                # P(opponent tends to play macro)
opponent.games_played -> int                             # How many games we have vs this opponent
opponent.adjusted_prior() -> dict                        # Strategy Belief prior dict
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
|-------------------|-----------------|--------|
| `bot.py:296-300` | Composition Belief | Replace binary `age < 30s` filter with weighted unit list |
| `bot.py:on_unit_destroyed` | Composition Belief | Remove destroyed units with `P=0.0` immediately |
| `intel.py:get_enemy_intel_quality()` | Composition Belief | Freshness score becomes one component of composition belief |
| `intel.py:update_enemy_intel_tracking()` | Composition Belief | Populate `_enemy_unit_last_seen` dict (ghost field, currently empty) |
| `reactions.py:assess_threat()` | Composition Belief | Use `composition.get_weighted_army()` for near-base threat calculation |
| `reactions.py:threat_detection()` | Strategy Belief | `P(imminent_push)` informs `_under_attack` hysteresis |
| `combat.py:handle_attack_toggles()` | Composition + Strategy | Weighted army in combat sim; strategy-informed risk thresholds |
| `combat.py:can_win_fight()` | Composition Belief | Receives weighted army instead of binary-filtered army |
| `rush_detection.py:get_enemy_ling_rushed_v2()` | Strategy Belief | Superseded by `strategy_belief.py` (kept as fallback during transition) |
| `intel.py:get_enemy_cannon_rushed()` | Strategy Belief | Superseded by `P(cannon_rush)` from BN |
| `scouting.py:get_hunt_target()` | Scout VOI | Rank destinations by information gain |
| `scouting.py:control_observers()` | Scout VOI | Prioritize highest-VOI destination (observer assignment logic unchanged) |

---

## Data Requirements

### Data Already Flowing (Telemetry)

| Belief Model | Telemetry Source | Fields Used |
|-------------|-----------------|-------------|
| Composition | `intel` periodic events + `on_unit_destroyed` callback | `scouted_enemy_units`, `scouted_enemy_structures`, `visible_enemy_count` |
| Strategy | `rush_detect` events + match record (via DuckDB) | All timing features, `ml_probs`, `ml_confidence`, `cheese_type` |
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
| Strategy (Zerg) | 50+ vs Zerg (already have some) | Existing + new telemetry via DuckDB |
| Strategy (Terran/Protoss) | 50+ vs each race | New games needed |
| Scout VOI | 0 (derived from other beliefs) | N/A |
| Opponent profiles | Accumulates over time (0 min, improves with games) | Match records |

**Critical**: `data/` directory is currently empty. The `data/games/` directory doesn't exist yet. No games have been run locally with telemetry enabled. Phase 2 requires collecting 50+ games per race before the BN can be trained.

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
  belief/                          ← new package
    __init__.py                    ← Re-exports: create_belief_state, BeliefState
    belief_state.py                ← BeliefState dataclass (read-only snapshot consumed by decisions)
    belief_updater.py              ← Ingests observations, produces new BeliefState each frame
    composition_belief.py          ← Phase 1: enemy composition with soft decay + structure priors
    strategy_belief.py             ← Phase 2: multi-race strategy classifier (pgmpy BN + rule guards)
    scout_voi.py                   ← Phase 3: value-of-information for scout destination ranking
    opponent_belief.py             ← Phase 4: cross-game opponent style priors
  models/
    rush_detector_model.pkl        ← Existing: Zerg rush LogisticRegression (kept as fallback during transition)
    strategy_belief_model.pkl      ← New: pgmpy BN structure + fitted parameters (Phase 2)
queries/
  views.py                         ← Shared DuckDB view definitions + connection helper
scripts/
  train_strategy_belief.py         ← Train pgmpy BN using queries/views.py
  train_rush_model.py              ← Existing: update to use queries/views.py instead of deleted JSONL file
data/
  games/                           ← Per-game JSONL telemetry (created on first write)
  opponent_profiles.json           ← Cross-game opponent profiles (Phase 4 persistence)
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
| Competition-safe defaults | If belief models fail to load or produce NaN, fall back to existing heuristics. Feature-gated behind `config.yml` flags, all defaulting to **off** in competition builds. |

### Feature Flags (config.yml)

```yaml
belief:
  enable_composition: false    # Phase 1
  enable_strategy: false       # Phase 2
  enable_scout_voi: false      # Phase 3
  enable_opponent: false       # Phase 4
```

Each flag defaults to `false`. Beliefs are disabled in competition until validated. When disabled, existing heuristics run unchanged. Flags are documented in config and in the telemetry plan.

---

## Data Pipeline

### Existing Infrastructure: DuckDB + Streamlit

Telemetry data flows through an established pipeline:

```
Game Telemetry (JSONL / stdout TELEM)
         │
         ▼
    DuckDB database          ← central data store, accumulates across all games
         │
         ├──→ Streamlit dashboard   ← live monitoring, ad-hoc queries
         │
         └──→ Training scripts        ← query DuckDB directly for model training
```

DuckDB queries JSONL files directly via `read_json_auto('data/games/*.jsonl')` — no persisted database file, no import step, no server. Training scripts share view definitions via `queries/views.py`. This is a significant advantage:

- **No ETL lag** — training scripts always read the latest data from the JSONL files
- **No stale CSVs** — DuckDB queries are live; no "forget to regenerate the training set" bugs
- **No database file to maintain** — the JSONL files ARE the database; DuckDB just reads them in-place
- **SQL joins are trivial** — match records + events joined by `match_id` in a single query
- **Streamlit provides visibility** — monitor data quality, class balance, and distribution drift before training

### Shared View Definitions: `queries/views.py`

All training scripts share common DuckDB view definitions. This avoids duplicating the SQL in each script:

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

Training scripts create these views via DuckDB SQL. Example for strategy belief:

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

The existing training script reads from `data/rush_detection_log.jsonl` which no longer exists. It must be rewritten to use `queries/views.py` instead. This is the **smallest change to unblock model retraining** — the DuckDB view already produces the same feature columns the old format provided.

### Updated: `scripts/train_strategy_belief.py`

1. Import `get_connection` from `queries.views`
2. Query `strategy_training` view (pre-created by `get_connection`)
3. Define BN structure (arcs from SC2 domain knowledge)
4. Fit parameters with `MaximumLikelihoodEstimator`
5. Validate accuracy against held-out test set
6. Save model to `bot/models/strategy_belief_model.pkl`

### Updated: `scripts/train_rush_model.py`

1. Import `get_connection` from `queries.views`
2. Query the same `strategy_training` view (backward-compat feature columns)
3. Train LogisticRegression (existing approach)
4. Save model to `bot/models/rush_detector_model.pkl`

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
| 4 | +2 | +1 (OpponentBelief) | 0 | 0 | +1 (bot, strategy) | 2 | Within budget |

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

2. **Most likely failure/edge**: `data/` is empty — no games have been run locally with telemetry enabled. The training pipeline for Phase 2 (Strategy Belief) cannot ship without collecting 50+ games per race. The first thing to do is run games, confirm telemetry JSONL files are being written to `data/games/`, verify the views in `queries/views.py` return data by querying DuckDB directly, and then train models.

3. **Smallest change to improve robustness**: Populate `_enemy_unit_last_seen` (10 LOC in `update_enemy_intel_tracking()`). This is the foundation for composition belief decay and is currently a ghost — declared but never written to. It costs nothing, breaks nothing, and unblocks Phase 1.