# Telemetry Plan

Structured logging for the SC2 bot (AI Arena ladder). Every match produces queryable JSONL data.

## Questions This Data Answers

1. Is the bot improving over time / across versions?
2. Which matchups and maps are weak?
3. What decisions led to a specific loss?

---

## Schemas

All records carry a `schema_version` field. Increment this when the schema changes in a way that breaks downstream parsing. Current version: `1`.

### Match Record — one per game, written at game end

The match record is the **index card** for finding and grouping games. It holds game-level conclusions and summaries — single values that characterize the whole game. If a value changes during play, it belongs in an event record instead.

| Field                    | Type   | Required | Description                                           |
|--------------------------|--------|----------|-------------------------------------------------------|
| `schema_version`         | number | yes      | Schema version, currently `1`                        |
| `ts`                     | string | yes      | ISO timestamp                                         |
| `version`                | string | yes      | Bot semver, e.g. `"0.9.0"`                           |
| `env`                    | string | yes      | `"local"` / `"ci"` / `"ladder"`                      |
| `match_id`               | string | yes      | UUID generated at game start — stable join key         |
| `arena_match_id`         | number | no       | AI Arena match ID — filled by puller post-match; `null` during game |
| `opponent_id`            | string | no       | AI Arena opponent user ID (from `--OpponentId` CLI arg) |
| **Context**              |        |          |                                                       |
| `bot_race`               | string | no       | Our race, e.g. `"Protoss"`                            |
| `enemy_race`             | string | no       | Opponent race, e.g. `"Zerg"`                          |
| `map`                    | string | no       | Map name                                               |
| `chosen_opening`         | string | no       | Bot's build order choice (key decision input)         |
| `rush_time_seconds`      | number | no       | Rush distance estimate in seconds (derive tiers in analysis) |
| **Result**               |        |          |                                                       |
| `result`                 | string | no       | `"win"` / `"loss"` / `"tie"` / `"undecided"` / `"incomplete"` |
| `length`                 | number | no       | Game duration in seconds                               |
| **Bot's game-level conclusions** | |    |                                                       |
| `cheese_type`            | string | no       | Opponent strategy classification, e.g. `"12_pool"`, `"cannon_rush"`, `"none"` |
| `commenced_attack`       | bool   | no       | Did the bot ever go on offense?                        |
| `used_cheese_response`   | bool   | no       | Did the bot activate cheese defense?                   |
| **Performance (from PerformanceMonitor)** | | |                                                    |
| `sq`                     | number | no       | Combined spending quotient                             |
| `mineral_sq`             | number | no       | Mineral spending quotient                               |
| `gas_sq`                 | number | no       | Gas spending quotient                                  |
| `efficiency_rating`      | string | no       | Bot's label: `"good"` / `"ok"` / `"poor"` etc.        |
| `avg_unspent_minerals`   | number | no       | Average unspent minerals over the game                  |
| `avg_unspent_vespene`    | number | no       | Average unspent vespene over the game                   |
| `avg_income_minerals`    | number | no       | Average mineral income rate over the game               |
| `avg_income_vespene`     | number | no       | Average vespene income rate over the game                |
| `idle_worker_time`       | number | no       | Total idle worker time in seconds                       |
| `idle_production_time`   | number | no       | Total idle production time in seconds                   |

### Event Record — many per game, written during play

| Field              | Type   | Required | Description                                              |
|--------------------|--------|----------|----------------------------------------------------------|
| `schema_version`   | number | yes      | Schema version, currently `1`                            |
| `ts`               | number | yes      | Game time in seconds                                     |
| `match_id`         | string | yes      | UUID — links events to the match record                 |
| `subsystem`        | string | yes      | Origin module (see subsystem schemas below)              |
| `action`           | string | yes      | What the subsystem did                                  |
| `reason`           | string | yes      | Short label for why                                      |

Additional fields are flattened at the top level — no nested `state` object. Each subsystem emits its own set of fields defined below.

**Design principle:** Event records capture the **bot's POV** — decisions, perceptions, and classifications. Raw game state (resource counts, unit compositions, supply) is reconstructable from replay and should be omitted unless the bot's *perception* differs from reality (e.g., scouted enemy counts vs. actual counts).

**Fields intentionally omitted** (reconstructable from replay): `minerals`, `vespene`, `supply_used`, `supply_cap`, `income_rate_mineral`, `income_rate_vespene`, `army_size`, own army composition, base count, worker count.

**Composition format:** Unit/structure counts use JSON objects, not strings. DuckDB parses these natively. Example: `{"Stalker": 12, "Sentry": 3}` not `"12xStalker,3xSentry"`.

**Field-level JSON is allowed:** The "no nested objects" constraint means no wrapping `state` object around all fields. Individual fields may be JSON objects (composition, probabilities) — DuckDB handles these via `read_json_auto`.

**Missing fields:** If a field cannot be computed (e.g., `can_win_fight` throws an exception), omit the field silently. Do not write `null` or error values. The absence of a field is itself informative.

---

### Event Categories

Events fall into three categories. Each has different emission rules:

**1. Decision events** — emitted at the moment a decision is made. These are the most valuable events.

```
subsystem="combat", action="engage", reason="army_superior", can_win_fight=True, ...
```

**2. State transitions** — emitted only when a value changes. Never repeated at the same value. The `reason` describes what triggered the change.

```
subsystem="combat", action="state_change", reason="attack_commenced", commenced_attack=True
subsystem="reactions", action="phase_transition", reason="early_to_mid", game_phase=1
```

State transition fields: `commenced_attack`, `under_attack`, `game_phase`, `economy_state`, `cheese_type`. These are binary flags or labels that flip infrequently. Emitting them every 30s is pure noise — you already know the state from the most recent transition.

**3. Periodic snapshots** — emitted at a fixed interval (every 30 game seconds) with only continuously-varying quantities. No state flags.

```
subsystem="combat", action="periodic", reason="timer",
  can_win_fight=True, attack_target="Point2(50.3, 112.7)",
  role_attacking=24, role_defending=4, role_base_defender=2,
  defender_composition={"Stalker": 3, "Sentry": 1},
  squads_atk=2, squads_def=1, squads_base=1
```

Snapshot fields: `can_win_fight`, `attack_target`, `role_attacking`, `role_defending`, `role_base_defender`, `defender_composition`, `squads_atk`, `squads_def`, `squads_base`, `scouted_enemy_units`, `scouted_enemy_structures`, `visible_enemy_count`.

**Why 30s?** The game report already uses 30s. Now that state flags are separated into transition events, every snapshot carries genuinely new information — no redundant `commenced_attack=true` repeated 15 times. Volume is negligible (~20 events per game, ~2KB JSONL). Until the replay pipeline exists, these snapshots are the only temporal record of the bot's state, so 30s resolution is worth keeping.

---

### Subsystem Schemas

Each subsystem defines its own `action` vocabulary and optional fields. The required fields (`schema_version`, `ts`, `subsystem`, `action`, `reason`) are universal.

#### `"combat"` — Army control, engagement decisions

| Action              | Category   | Reason examples                          | Optional fields                                                                 |
|---------------------|------------|------------------------------------------|----------------------------------------------------------------------------------|
| `"engage"`          | Decision   | `"army_superior"`, `"timing_window"`     | `can_win_fight`, `attack_target`, `role_attacking`, `role_defending`, `role_base_defender`, `defender_composition`, `squads_atk`, `squads_def`, `squads_base` |
| `"retreat"`         | Decision   | `"outnumbered"`, `"threat_detected"`      | `can_win_fight`                                                                  |
| `"state_change"`    | Transition | `"attack_commenced"`, `"under_attack"`, `"attack_ended"`, `"threat_cleared"` | `commenced_attack`, `under_attack` |
| `"periodic"`        | Snapshot   | `"timer"`                                | `can_win_fight`, `attack_target`, `role_attacking`, `role_defending`, `role_base_defender`, `defender_composition`, `squads_atk`, `squads_def`, `squads_base` |

| Field                    | Type   | Description                                              |
|--------------------------|--------|----------------------------------------------------------|
| `can_win_fight`          | bool   | Bot's combat simulation result                           |
| `commenced_attack`       | bool   | Whether bot has switched to offensive posture            |
| `under_attack`           | bool   | Bot's threat assessment                                  |
| `attack_target`          | string | Bot's current attack target (position or structure name) |
| `role_attacking`         | number | Units assigned to ATTACKING role                         |
| `role_defending`         | number | Units assigned to DEFENDING role                         |
| `role_base_defender`     | number | Units assigned to BASE_DEFENDER role                     |
| `defender_composition`   | object | JSON: unit type → count for defending units              |
| `squads_atk`             | number | Number of attacking squads                              |
| `squads_def`             | number | Number of defending squads                               |
| `squads_base`            | number | Number of base defender squads                           |

#### `"reactions"` — Threat detection, cheese responses

| Action              | Category   | Reason examples                    | Optional fields                          |
|---------------------|------------|------------------------------------|------------------------------------------|
| `"cheese_detected"` | Decision  | `"worker_rush"`, `"cannon_rush"`   | `cheese_type`, `under_attack`            |
| `"defender_alloc"`  | Decision  | `"harass_response"`, `"all_in"`    | `role_defending`, `defender_composition` |
| `"phase_transition"`| Transition | `"early_to_mid"`, `"mid_to_late"`  | `game_phase`                              |

| Field                    | Type   | Description                                              |
|--------------------------|--------|----------------------------------------------------------|
| `cheese_type`            | string | Detected cheese strategy (see match record enum)         |
| `under_attack`           | bool   | Bot's threat assessment                                  |
| `game_phase`             | number | Bot's phase classification (0=Early, 1=Mid, 2=Late)      |
| `role_defending`         | number | Units assigned to DEFENDING role                         |
| `defender_composition`   | object | JSON: unit type → count for defending units              |

#### `"cheese_detect"` — Cheese/all-in classification and scouting

| Action              | Category   | Reason examples                    | Optional fields                          |
|---------------------|------------|------------------------------------|------------------------------------------|
| `"classification"`  | Transition | `"score_updated"`, `"label_changed"` | `cheese_label`, `score_12p`, `score_speed`, `cheese_detected`, `auto_true_fired`, `cheese_source` |
| `"ml_update"`       | Transition | `"model_evaluated"`               | `ml_probs`, `ml_confidence`              |
| `"scout_report"`    | Decision   | `"nat_scouted"`, `"pool_seen"`    | `nat_present_on_last_scout`, `last_nat_scout_time`, `enemy_nat_started_at`, `pool_seen_state`, `pool_seen_time`, `speed_research_started`, `speed_research_time`, `extractor_seen_time`, `queen_started_time`, `first_ling_seen_time`, `first_ling_contact_nat_time`, `ling_has_speed`, `gas_workers_count` |

| Field                        | Type   | Description                                              |
|------------------------------|--------|----------------------------------------------------------|
| `cheese_label`               | string | Current classification: `"12_pool"`, `"speedling"`, `"macro"`, `"none"` |
| `score_12p`                  | number | 12-pool heuristic score                                  |
| `score_speed`                | number | Speedling heuristic score                               |
| `cheese_detected`            | bool   | Whether cheese/all-in is confirmed                        |
| `auto_true_fired`            | bool   | Whether auto-override triggered                          |
| `cheese_source`              | string | What triggered detection                                 |
| `ml_probs`                   | object | JSON: class → probability (e.g. `{"12_pool": 0.7, "speedling": 0.2, "macro": 0.1}`) |
| `ml_confidence`              | number | ML model confidence (0–1)                                |
| `nat_present_on_last_scout`  | bool   | Whether bot believed enemy natural was present           |
| `last_nat_scout_time`        | number | Game time of last enemy natural scout                    |
| `enemy_nat_started_at`       | number | Game time bot detected enemy natural started             |
| `pool_seen_state`            | string | Pool state as bot saw it                                  |
| `pool_seen_time`             | number | Game time bot first saw pool                              |
| `speed_research_started`     | bool   | Whether bot detected speed research                      |
| `speed_research_time`        | number | Game time bot detected speed research                    |
| `extractor_seen_time`        | number | Game time bot first saw extractor (-1 if not seen)       |
| `queen_started_time`         | number | Game time bot first saw queen (-1 if not seen)            |
| `first_ling_seen_time`       | number | Game time bot first saw zergling (-1 if not seen)        |
| `first_ling_contact_nat_time`| number | Game time lings first reached our natural (-1 if not seen) |
| `ling_has_speed`             | bool   | Whether Metabolic Boost was detected on lings            |
| `gas_workers_count`          | number | Number of drones mining gas at enemy extractor           |

#### `"economy"` — Economy state transitions

| Action              | Category   | Reason examples                    | Optional fields                          |
|---------------------|------------|------------------------------------|------------------------------------------|
| `"state_transition"`| Transition | `"saturated"`, `"worker_rush_recovery"` | `economy_state`                          |

| Field              | Type   | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| `economy_state`    | string | Bot's economy classification label                       |

#### `"intel"` — Scouted enemy information

| Action              | Category   | Reason examples                    | Optional fields                          |
|---------------------|------------|------------------------------------|------------------------------------------|
| `"scout_update"`    | Snapshot   | `"periodic"`                       | `scouted_enemy_units`, `scouted_enemy_structures`, `visible_enemy_count` |

| Field                        | type   | Description                                              |
|------------------------------|--------|----------------------------------------------------------|
| `scouted_enemy_units`        | object | JSON: unit type → count of what bot believes it sees      |
| `scouted_enemy_structures`  | object | JSON: structure type → count of what bot believes it sees |
| `visible_enemy_count`       | number | Total visible enemy units (quick filter)                 |

#### Other subsystems — to be defined

The following subsystems will emit events but their action/field vocabularies are not yet specified. They follow the same pattern: required fields + subsystem-specific optional fields. **Rule: no event from an undefined subsystem without updating this plan first.**

| subsystem            | Source module                          | Will emit events for (future definition)             |
|----------------------|----------------------------------------|------------------------------------------------------|
| `"unit_micro"`       | `bot/combat/unit_micro.py`             | Per-unit micro decisions (blink, storm, FF, dodge)   |
| `"formation"`        | `bot/combat/formation.py`              | Concave formation start/stop, fan-out execution      |
| `"target_scoring"`   | `bot/combat/target_scoring.py`         | Weighted target selection, counter-table hits        |
| `"group_snipe"`      | `bot/combat/group_snipe.py`            | Blink snipe commit/execute, focus fire               |
| `"group_chase"`      | `bot/combat/group_chase.py`            | Chase commit/execute, retreat detection              |
| `"force_field"`      | `bot/combat/force_field.py`             | FF split/ramp/choke block decisions                  |
| `"macro"`            | `bot/managers/macro.py`                | Production decisions, worker caps, gas adjustments  |
| `"scouting"`         | `bot/managers/scouting.py`             | Observer assignments, worker/hallucination scouts    |
| `"structure_manager"`| `bot/managers/structure_manager.py`    | Chrono boost, recharge, mass recall                  |
| `"nova"`             | `bot/utilities/nova_manager.py`         | Disruptor nova registration, exclusion zones         |
| `"wall"`             | `bot/utilities/natural_wall_manager.py` | Wall generation, placement decisions                  |

New subsystems can be added freely — define the action vocabulary and optional fields when implementing.

---

## Sampling Policy

Event records are sampled by environment. Match records are never sampled (always written).

```python
TELEMETRY_SAMPLE_RATE = {"local": 0.1, "ci": 0.5, "ladder": 1.0}
```

In `log_event()`: if `random.random() > sample_rate[env]`, skip silently.

---

## Output Convention

- **stdout** → telemetry lines, each prefixed with `TELEM `
- **stderr** → errors and tracebacks only, never mixed
- **ladder games** → stdout with `TELEM ` prefix (captured by bot controller into `stdout.log`)
- **local games** → `data/games/<match_id>.jsonl` (raw JSONL, no prefix; terminal stays clean)

Example:

```python
log_event(subsystem="combat", action="engage", reason="army_superior",
          can_win_fight=True,
          role_attacking=24, role_defending=4,
          defender_composition={"Stalker": 3, "Sentry": 1})
# → TELEM {"schema_version":1,"ts":124.3,"version":"0.9.0","env":"ladder","match_id":"a1b2c3d4-...","subsystem":"combat","action":"engage","reason":"army_superior","can_win_fight":true,"role_attacking":24,"role_defending":4,"defender_composition":{"Stalker":3,"Sentry":1}}
```

---

## Versioning

- `BOT_VERSION` constant lives in `__init__.py`, follows semver (`MAJOR.MINOR.PATCH`).
- `SCHEMA_VERSION` constant lives in `telemetry.py`, currently `1`.
- The writer stamps both onto every record automatically.
- Bot version is bumped manually on meaningful behavior changes; tagged as `git tag v0.9.0`.
- Schema version is bumped only when record fields are added/renamed/removed — not tied to bot version.

---

## Required Fields

`schema_version`, `version`, `env`, `match_id` must always be present on both match and event records. Without them, downstream analysis cannot separate runs.

`arena_match_id` and `opponent_id` are match-record-only fields. `arena_match_id` is `null` during the game and filled by the puller post-match. `opponent_id` is captured from `--OpponentId` on ladder, `null` locally.

---

## Migration of Existing Code

### `game_report.py` → telemetry events

#### `print_startup_report()` → match record fields

Startup data is static game context — it belongs in the match record, not as a separate event.

| Startup field            | Match record field        | Notes                                          |
|--------------------------|---------------------------|------------------------------------------------|
| `map_name`               | `map`                     | Already in schema                              |
| `enemy_race`              | `enemy_race`               | Direct from `bot.enemy_race.name`              |
| —                          | `bot_race`                | **New** — our race (always Protoss currently)  |
| `chosen_opening`          | `chosen_opening`          | **New** — bot's build decision                  |
| `rush_time_seconds`       | `rush_time_seconds`       | **New** — derive tiers in analysis, don't bake thresholds into schema |
| `natural_expansion`       | —                         | Position data, not a grouping key. Derivable from map. |
| `enemy_natural`           | —                         | Same — position, not useful for analysis.       |

#### `print_periodic_intel_report()` → event records

Each 30-second snapshot from the game report becomes multiple targeted events. Fields are split into **bot POV** (keep) and **game's-eye** (drop — reconstructable from replay). Bot POV fields are further split by event category: **Transition** (emit on change only), **Snapshot** (emit every 30s), and **Decision** (emit at decision point).

**Bot POV — emit as event fields:**

| Periodic field            | Event field               | Subsystem   | Category     | Notes                                    |
|---------------------------|---------------------------|-------------|--------------|------------------------------------------|
| `_commenced_attack`       | `commenced_attack`        | `combat`    | Transition   | Emit only on false→true change           |
| `_under_attack`           | `under_attack`            | `combat`    | Transition   | Emit only on change                       |
| `game_state` (0/1/2)     | `game_phase`              | `reactions` | Transition   | Emit only on change                       |
| `can_win_fight`           | `can_win_fight`           | `combat`    | Snapshot     | Continuously varying                      |
| `current_attack_target`   | `attack_target`           | `combat`    | Snapshot     | Continuously varying                      |
| Role counts (ATK/DEF/BASE) | `role_attacking`, etc.  | `combat`    | Snapshot     | Continuously varying                      |
| Defender composition      | `defender_composition`    | `combat`    | Snapshot     | Continuously varying, JSON object         |
| Squad counts              | `squads_atk`, etc.       | `combat`    | Snapshot     | Continuously varying                      |
| `economy_state` label    | `economy_state`           | `economy`   | Transition   | Emit only on change                       |
| Cheese detection fields     | Various (see cheese_detect schema) | `cheese_detect` | Transition/Decision | See cheese_detect schema       |
| Scouted enemy units       | `scouted_enemy_units`    | `intel`     | Snapshot     | JSON object: unit type → count            |
| Scouted enemy structures  | `scouted_enemy_structures` | `intel`   | Snapshot     | JSON object: structure type → count       |

**Game's-eye — do NOT emit (reconstructable from replay):**

| Periodic field            | Why skip                                   |
|---------------------------|--------------------------------------------|
| Own army composition      | Replay has exact unit counts               |
| `minerals` / `vespene`    | Replay has resource data                   |
| `supply_used` / `supply_cap` | Replay has supply data                  |
| `income_rate_*`           | Replay has income rates                    |
| `bases` count             | Replay has structure data                  |
| `workers` / `gathering`   | Replay has worker data                     |

**Borderline — keep with clear labeling:**

| Periodic field            | Event field                  | Notes                                    |
|---------------------------|------------------------------|------------------------------------------|
| Scouted enemy units       | `scouted_enemy_units`        | JSON object: what the bot *believed* it saw |
| Scouted enemy structures  | `scouted_enemy_structures`   | JSON object: what the bot *believed* it saw |

#### `print_end_game_report()` → match record fields

End-game data is game-level summary — belongs in the match record.

| End-game field              | Match record field          | Notes                                |
|-----------------------------|-----------------------------|--------------------------------------|
| `game_result`               | `result`                    | Already in schema                    |
| `game_time`                 | `length`                    | Already in schema                    |
| `sq`                        | `sq`                        | **New** — combined spending quotient |
| `mineral_sq` / `gas_sq`     | `mineral_sq` / `gas_sq`     | **New** — per-resource SQ            |
| `efficiency_rating`         | `efficiency_rating`         | **New** — bot's label                |
| `avg_unspent_minerals`      | `avg_unspent_minerals`      | **New**                              |
| `avg_unspent_vespene`       | `avg_unspent_vespene`       | **New**                              |
| `avg_income_minerals`       | `avg_income_minerals`       | **New**                              |
| `avg_income_vespene`        | `avg_income_vespene`        | **New**                              |
| `idle_worker_time`          | `idle_worker_time`          | **New**                              |
| `idle_production_time`      | `idle_production_time`      | **New**                              |

#### `get_replay_tags_to_send()` → match record field

All replay tags collapse into a single `cheese_type` field on the match record.

| Tag value                  | `cheese_type` value         |
|----------------------------|-----------------------------|
| `Rush_12_pool`             | `"12_pool"`                 |
| `Rush_speedling`           | `"speedling"`               |
| `WorkerRush`               | `"worker_rush"`             |
| `CannonRush`               | `"cannon_rush"`             |
| `MarineRush`               | `"marine_rush"`             |
| `MarauderRush`             | `"marauder_rush"`           |
| `ProxyZealot`              | `"proxy_zealot"`            |
| `FourGate`                 | `"four_gate"`               |
| `RoachRush`                | `"roach_rush"`              |
| `RavagerRush`              | `"ravager_rush"`            |
| (none)                     | `"none"`                    |

If multiple tags fire (unlikely but possible), use the first detected. The `cheese_type` field is the bot's final classification of opponent strategy.

`print_*` functions become thin wrappers — they still print human-readable output for dev debugging, but they also emit a `TELEM` line alongside. The `TELEM` output is the canonical data source; the printed output is secondary.

### `cheese_detection.py` — match record + event records (replaces old `log_rush_detection_result()`)

The hand-rolled JSONL writer (formerly `log_rush_detection_result()` in `rush_detection.py`, writing to `data/rush_detection_log.jsonl`) is **replaced** by telemetry. The training script (`scripts/train_rush_model.py`) currently reads from `data/rush_detection_log.jsonl` — it will be updated to read from the `TELEM` pipeline instead.

**Why this works:** All 13 ML feature columns are now available in telemetry events:

| Training Feature | Telemetry Source | Event |
|---|---|---|
| `pool_start` | `pool_seen_time` | `cheese_detect` `scout_report` |
| `nat_start` | `enemy_nat_started_at` | `cheese_detect` `scout_report` |
| `last_nat_scout_time` | `last_nat_scout_time` | `cheese_detect` `scout_report` |
| `nat_present_on_last_scout` | `nat_present_on_last_scout` | `cheese_detect` `scout_report` |
| `gas_time` | `extractor_seen_time` | `cheese_detect` `scout_report` |
| `queen_time` | `queen_started_time` | `cheese_detect` `scout_report` |
| `ling_seen` | `first_ling_seen_time` | `cheese_detect` `scout_report` |
| `ling_contact` | `first_ling_contact_nat_time` | `cheese_detect` `scout_report` |
| `speed_start` | `speed_research_time` | `cheese_detect` `scout_report` |
| `ling_has_speed` | `ling_has_speed` | `cheese_detect` `scout_report` |
| `gas_workers` | `gas_workers_count` | `cheese_detect` `scout_report` |
| `score_12p` | `score_12p` | `cheese_detect` `classification` |
| `score_speed` | `score_speed` | `cheese_detect` `classification` |
| `cheese_label` | `cheese_label` | `cheese_detect` `classification` |
| `auto_true_fired` | `auto_true_fired` | `cheese_detect` `classification` |
| `result` | `result` | Match record |
| `game_time_seconds` | `length` | Match record |
| `map_name` | `map` | Match record |
| `enemy_race` | `enemy_race` | Match record |
| `rush_distance_seconds` | `rush_time_seconds` | Match record |

**Migration steps:**
1. Add `log_event()` calls to `_track_enemy_timings()` for `scout_report` events (when timing data updates).
2. Add `log_event()` calls to `get_enemy_ling_rushed_v2()` for `classification` and `ml_update` events (when scores/labels change).
3. Add `log_match()` call to `on_end()` for match record fields (replacing `log_cheese_detection_result()`).
4. Remove `log_cheese_detection_result()` function from `cheese_detection.py`.
5. Delete `data/cheese_detection_log.jsonl` — all data flows through `TELEM` now.

**Training pipeline impact:** The current `log_cheese_detection_result()` writes one record per game with all 13 ML features in a single row. With telemetry, features are spread across multiple events at different timestamps. The training script (`scripts/train_rush_model.py`) must be updated to:
1. Filter TELEM lines for `subsystem="cheese_detect"` by `match_id`
2. Get the **final** `classification` event (for `cheese_label`, `score_12p`, `score_speed`, `auto_true_fired`)
3. Get the **latest** `scout_report` event before classification (for all timing features)
4. Join with the match record (for `result`, `enemy_race`, `map`, `rush_time_seconds`)

This is a standard DuckDB join — not complex, but it's a real change from the current single-line-per-game format.

---

## Data Pipeline (Out of Scope)

### Puller + Enrichment Pipeline (future)

A puller script (separate from the bot) runs post-match and does three things:

1. **Download logs** — fetch `stdout.log` and `stderr.log` from AI Arena, filter for `TELEM`-prefixed lines, strip the prefix, write into `games/<match_id>.jsonl`.
2. **Enrich `arena_match_id`** — call `https://aiarena.net/api/match-participations/?bot=<BOT_ID>&ordering=-created` with the user's API token, find the most recent match matching `opponent_id` + `map` + timestamp, and write the real AI Arena `match_id` into every record's `arena_match_id` field.
3. **Download replay** — fetch the `.SC2Replay` file for later replay parsing (game's-eye data: resources, supply, exact army composition, etc.).

The enrichment step is the only place `arena_match_id` gets filled. During the game, it's `null`. This keeps the bot simple (no API calls during gameplay) while ensuring every record can be joined to AI Arena data later.

DuckDB queries the `games/` folder directly via `read_json_auto('games/*.jsonl')`. No import step, no server.

---

## Scope

**In scope:**

- `telemetry.py` module with `log_event()` and `log_match()` helpers
- `BOT_VERSION` constant in `bot/__init__.py` (already exists as `"0.9.0"`)
- `SCHEMA_VERSION` constant in `telemetry.py`
- Sampling policy by environment
- Replace `game_report.py` as the data pipeline — `print_*` functions become thin wrappers that print human-readable output **and** emit `TELEM` lines. The `TELEM` output is the canonical data source; the printed output is for dev convenience only.
- Migrate `cheese_detection.py:log_cheese_detection_result()` into `log_match()` + `cheese_detect` events
- Deprecate `data/cheese_detection_log.jsonl` — all data flows through `TELEM` now

**Out of scope (for now, but prepared for):**
- Puller script that downloads logs from AI Arena and enriches `arena_match_id`
- Replay download and parsing for game's-eye data
- DuckDB queries and dashboards
- structlog dependency

**Preparation for future enrichment:** The schema already includes `arena_match_id` (null during game, filled post-match) and `opponent_id` (from `--OpponentId`). The puller will use the AI Arena API (`/api/match-participations/?bot=<BOT_ID>&ordering=-created`) to match on `opponent_id` + `map` + timestamp and write the real AI Arena match ID into every record.

---

## Implementation Decisions

### match_id source

AI Arena does **not** pass a match ID to bots — only `--OpponentId` (opponent's user ID). So `match_id` is **always a UUID** generated at game start. The real AI Arena match ID (`arena_match_id`) is filled post-match by the puller.

### env detection

Check `--LadderServer` in `sys.argv` (same check `run.py` already uses to distinguish ladder vs local). No env var or config change needed:

```python
def _detect_env() -> str:
    if "--LadderServer" in sys.argv:
        return "ladder"
    return "local"   # CI can override via env var later
```

This mirrors the existing pattern in `run.py` line 64: `if "--LadderServer" in sys.argv`. The `ci` env is reserved for future CI runs — currently only `local` and `ladder` are produced.

### BOT_VERSION initial value

Already exists in `bot/__init__.py` as `BOT_VERSION = "0.9.0"`. The telemetry module imports it from there. Bumped manually on meaningful behavior changes, tagged as `git tag v0.9.0`.

### Transition tracking

Module-level state in `telemetry.py`. SC2 bots run one game per process — there's no concurrency concern. A simple dict tracks previous values:

```python
# In telemetry.py
_prev: dict = {}   # e.g. {"commenced_attack": False, "game_phase": 0, ...}

def log_transition(subsystem: str, field: str, new_value) -> bool:
    """Return True if value changed (caller should emit the event)."""
    if _prev.get(field) == new_value:
        return False
    _prev[field] = new_value
    return True
```

Reset `_prev` at game start (called from `on_start` or `log_match`).

---

## Constraints

- JSONL only — no nested formats, no binary, no in-bot databases.
- stdout for telemetry, stderr for errors — never mix.
- Telemetry writes must be cheap and never block gameplay logic.
- `match_id` is a UUID generated at game start — it's the stable join key for all events in a match. Never changes.
- `arena_match_id` is the AI Arena integer match ID — **not available during the game**. The bot controller passes `--OpponentId` (opponent's user ID) but not the match ID. The puller enriches this field post-match by querying `https://aiarena.net/api/match-participations/?bot=<BOT_ID>&ordering=-created` and matching on `opponent_id` + `map` + timestamp.
- `opponent_id` is captured from the `--OpponentId` CLI arg on ladder. It's the AI Arena user ID of the opponent, and is the primary key for correlating with the API.
- All optional fields are flattened at the top level of the record — no nested `state` object.