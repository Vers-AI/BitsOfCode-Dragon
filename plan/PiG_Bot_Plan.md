---
project: PiG_Bot
version: 1.0
author: Drekken
goal: "Develop a human-like Protoss bot following PiG's B2GM 2023 double-robo macro style."
framework: ARES
language: Python
status: active
priority: high
modules:
  - Combat
  - Tactics
  - Macro
  - Build Manager
  - Scouting
  - Logging & Data Storage
  - Analysis & Learning
  - Replay Parsing
dependencies:
  - python-sc2
  - ARES
  - pandas
  - matplotlib
  - sc2reader
tags:
  - StarCraft II
  - AI
  - Bot Development
  - PiG_Bot
  - Windsurf
---

# PiG_Bot – Project Plan

## Objectives
- Build a Protoss SC2 bot that plays like a human (Bronze → GM) using PiG’s B2GM robo-centric style.
- Implement clear modules for Macro, Combat, Scouting, Build Management, and Reactive Defense.
- Support difficulty control, pre-defined build switches, and basic micro (stutter-step, disruptors).
- Log per-match data for analysis and future lightweight adaptation.

## Scope: What the Bot Should Do
- Human-like play patterns; difficulty control; adaptive reactions.
- Behaviors: build Bronze→GM macro style; scout and interpret; defend cheese (zergling/cannon/probe rush);
  disruptor usage; basic stutter-step; respond to air; stay near 3 bases/66 probes; pre-defined build switches;
  fight only when favored; two builds per matchup (PvP 4-Gate All-in / 2-Gate Expand; PvZ Standard Robo / Double SG Phoenix; PvT Standard Macro / Proxy Rax Response).

---

# Tasks

## Combat ✅ (from original **Plan → Combat**)
- [x] Integrate Combat simulator `can_win_fight`  
  - [x] Only attack if enemy combat score == 0
- [x] Create Probe Rush response
- [x] Enhance Probe Rush response with `WorkerKiteBack` and `DEFENDING` role (replicate cannon rush response)
- [x] Test Cannon Rush Response
- [x] Fix Cannon Rush Response
- [x] Enhance Observers  
  - [x] Army-observer path distance  
  - [x] Multi-state (Follow/Reveal)  
  - [x] Move to unseen enemies  
  - [x] Add additional observers  
  - [x] Surveillance mode  
  - [x] Split into SCOUT & FOLLOW roles
- [x] Send Zealot into the wall hole
  - [x] Fix units blocked by gatekeeper zealot
  - [x] Rally units inside, not outside
  - [x] Dodge ravager/disruptor balls & nukes
  - [x] React to Liberator zones
  - [x] React to invisible enemies
- [x] Add Sentry logic

### Tactics
- [x] Break army into squads
- [x] Per-squad attack targets & combat simulators
- [x] Anti-air squad (only AA units; proportional to threat)

#### MindMe's Weighted Scoring System
- [x] Phase 1a — Create `target_scoring.py`: `score_target()` + `select_target()` + weight constants
- [x] Phase 1b — Hybrid unit type values: `UNIT_DATA army_value` base + `TACTICAL_BONUS` overrides (spellcasters, siege, enablers)
- [x] Phase 1c — Wire into `unit_micro.py`: replace `cy_pick_enemy_target` + `priority_targets` with `select_target()`
- [x] Phase 1d — Wire into `combat.py`: remove old `get_priority_targets` logic, use `select_target` for fallback targeting
- [x] Phase 1e — Update `__init__.py` exports
- [x] **TEST** — Run games, watch replays, verify targeting behavior
- [x] Phase 3 — Wire formation movement (`cy_adjust_moving_formation`) for pre-combat grouping
- [x] Phase 4 — Feed counter table into production decisions (what to build vs enemy comp)
- [ ] Tune weights from replay observations (`DISTANCE_WEIGHT`, `HEALTH_WEIGHT`, `TYPE_VALUE_SCALE`, `TACTICAL_BONUS` entries)

### Gatekeeper System (PvZ Wall Logic) ✅
- [x] Issue Hold Position
- [x] Move to zealot hole in wall  
  - [x] Identify the hole
- [x] If no zealot in role, add zealot Gatekeeper role
- [x] Create Gatekeeper role
- [x] Ensure only active in PvZ
- [x] Only 1 zealot at a time
- [x] Hold position unless attacking
- [x] Release role when attacking, **not** when defending


## Macro
- [x] Build-order recovery if buildings get destroyed (consult Rasper)
- [ ] Adjust PvP strategy to 1-base (instead of natural)
- [x] Fluid weighted composition vs enemy comp & resource 
- [x] fix the resource dump on tech before 3rd is finished 

## Build Manager
- [x] Implement PvT Standard Macro build
- [ ] Implement PvT Proxy Rax Response build
- [ ] Implement PvZ Standard Robo Opener
- [ ] Implement PvZ Double Stargate Phoenix (vs Mutas)
- [ ] Implement PvP 4-Gate All-in
- [x] Implement PvP 2-Gate Expand
- [x] Add Repear wall positions
- [ ] Build selector by opponent race + scout info
- [x] Connect selected build to `BuildOrderRunner` at game start
- [ ] Mid-game build switch logic (e.g., Spire → Stargate)
- [x] Adjust transition from build order to fluid composition to wait till amount of macro to sustain

## Data & Information
- [ ] `scouted_spire()` and other scout triggers
- [x] Fallback logic if scout dies early
- [x] Create a Reaction System
- [x] Observers ensure mid-game critical tech scouting
- [x] Add trigger observer speed upgrades when intel is stale (via `_intel_urgency` system)
- [x] Add trigger observer production priority when intel is stale
- [x] Add trigger a unit scout if information is stale and no observer exists
- [x] Add Sentry hallucination scout when intel is stale (requires Sentry logic first)
- [x] Add build shield batteries if `under_attack`


### Logging & Data Storage
- [ ] Log per-game JSON to `/logs/`  
  - opponent_race, map_name, game_result, build_order_used, build_loss_flag, enemy_units_seen, biggest_fight_outcome
- [ ] Use pandas-friendly JSONL formatting
- [ ] `on_end()` post-game hook triggers logging

### Analysis & Learning (Post-Match)
- [ ] Analyze performance across matchups (pandas)
- [ ] Chart winrates over time (matplotlib)
- [ ] Visualize build performance by race/opponent/map
- [ ] Lightweight build adaptation (e.g., avoid low-winrate builds)

### Replay Parsing (Optional)
- [ ] Test `sc2reader` on bot & human replays
- [ ] Extract build order + APM
- [ ] Compare build execution vs human timing
- [ ] Store replay summaries alongside JSON logs
- [x] Add Tags for replays

## Misc / TODO
- [x] Change Gamestate from text → INT
- [x] Optimal worker check includes minerals + gas
- [x] Review macro cycle (floating minerals during build order)
- [x] Worker-rush returns to mining afterward
- [ ] Add cheese detection: structure near our town hall / mediator `get_enemy_PF_rushed`
- [x] Adjust `Worker_rush` detection range
- [x] Remove Target Unit; use Target Unit **position**
- [x] Check if regrouping is too aggressive in `regroup_army`
- [x] Check threat sensitivity—Marines/Marauders/Reapers
- [x] Review Battle_Sim sensitivity + `all_out_attack`
- [x] Add `Did_Enemy_rush` hook
- [x] Remove Overcharge
- [x] Test Energy Recharge
- [x] Test new battle-sim conditions
- [x] Fix battle lockout bouncing
- [x] Cap workers at a set limit (80; 8 on gold base)
- [x] Add upgrades logic
- [x] Inner/outer radius for unit grouping; break when 30% engaged
- [x] Add Logs for every 30 seconds
- [x] Put debug in game
- [x] Fix Gatekeeper off in non-PvZ
- [ ] Use action request to rescue stuck units
- [ ] Custom gas stealer logic — replace ARES gas steal preventer
  - Root cause: ARES `_handle_gas_steal` deadlocks the build runner when an enemy
    worker oscillates near our geysers. The preventer assigns/removes a guard worker
    frame-to-frame as the enemy probe moves in/out of the 12-tile radius, so
    `worker.build_gas()` never fires. Meanwhile `do_step()` has already set
    `current_step_started = True` and delegated to the preventer, blocking all
    subsequent build steps until the 800-mineral fail-safe force-completes the build.
  - Workaround in place: `ShouldHandleGasSteal: False` on all builds.
  - Goal: implement our own gas steal prevention in `bot/managers/` that:
    1. Only activates when we actually need gas (build order has a GAS step coming)
    2. Assigns a guard worker that stays committed (no churn on enemy range exit)
    3. Builds the assimilator via the standard `do_step()` path, not a separate
       delegation that can deadlock
    4. Releases the guard worker once the assimilator starts or the threat clears

---

## Data To Log (schema)
- `opponent_race`, `map_name`, `game_result`, `build_order_used`, `build_loss_flag`, `time_build_switched`,
  `enemy_opening_seen`, `enemy_first_army_unit`, `enemy_air_tech_seen`, `biggest_fight_outcome`, `units_lost_total`,
  `enemy_units_seen`, `scouted_before_4min`, `reacted_to_cheese`, `num_expansions_taken`.

## Dependencies
- `python-sc2`, `ARES`, `pandas`, `matplotlib`, `JSON` (logs), `sc2reader` (optional)

## Future Goals (After MVP)
- Difficulty selection; mode types (macro/harass/reactive); adaptive build switching via scoring/scouting; 4+ builds per MU; advanced micro (FF, blink, AoE splits); terrain logic for more maps; play other races; disable speed mining.

## Notes / Observations
- Could rally at the natural entrance instead of natural base.
- 17 marines didn’t trip threat.
- Drawn to injured units? Investigate targeting.
- Units rally toward expansion; may be better to rally 2 units in front of natural toward center.
