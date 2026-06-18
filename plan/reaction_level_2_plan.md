# Reaction Level 2 — PiGBot Strategy Responses

Derived from PiG's B2GM 2023 guide. These are **what the bot should do** once a strategy is detected. Detection is handled by the Bayesian belief layer (Level 1). This document defines the reaction policies only.

---

## Architecture: ReactionManager

All reaction state and routing is centralized in `ReactionManager` (`bot/managers/reactions.py`).

### Two-layer routing

| Layer | Purpose | Example |
|-------|---------|---------|
| **Category** (CHEESE/ALL_IN/TIMING/MACRO) | Macro/build influence | Switch build order, cap probes, cancel Nexus |
| **Strategy** (level2 label) | Tactical handler selection | `worker_rush` → `defend_worker_rush()`, `cannon_rush` → `defend_cannon_rush()` |

### Consumer API (`bot.reaction_manager.*`)

| Property | Type | Meaning |
|----------|------|---------|
| `is_active` | `bool` | Any reaction currently running? |
| `is_cheese_response` | `bool` | Cheese active AND not transitioned? (controls army comp, probe cap, gas) |
| `is_early_defensive` | `bool` | Should combat hold army back? (same as `is_cheese_response` for now) |
| `keep_workers_safe` | `bool` | Should Mining keep gas workers safe? (False during worker rush) |
| `active_reaction_name` | `str \| None` | Level2 label of active reaction (e.g., `"worker_rush"`) |
| `reaction_start_time` | `float` | Game time when reaction was activated (-1 if inactive) |
| `category_config` | `CategoryConfig` | Build/probe_cap/army_comp/cancel_nexus/hold_army/stop_gas_below |

### CategoryConfig (in `bot/constants.py`)

| Category | build | probe_cap | army_comp | cancel_nexus | hold_army | stop_gas_below |
|----------|-------|-----------|-----------|--------------|-----------|----------------|
| CHEESE | `Cheese_Reaction_Build` | 20 | CHEESE_DEFENSE_ARMY | True | True | 21 |
| ALL_IN | None | 44 | None | False | True | 0 |
| TIMING_ATTACK | None | None | None | False | False | 0 |
| MACRO | None | None | None | False | False | 0 |

### Cheese transition logic

Cheese mode transitions to standard army when **any** of:
1. Game reached mid-game (`game_state >= 1`, ~6:00 timer)
2. Sustained threat-free window: no enemy combat units near our bases for `CHEESE_THREAT_CLEAR_GRACE` seconds (30s)
3. Army has commenced attacking (`_commenced_attack = True`)

The old economy-based check was removed — it created a circular dependency where cheese mode kept workers low, which kept `economy_state` "reduced", which blocked the transition.

### Detection → Reaction flow

```
Strategy Belief (BN model) ──→ StrategyPrediction ──→ ReactionManager.update()
                                                              │
Rule-based fallback ──→ detect_cheese() ──→ _prediction_from_rules() ──┘
                                                              │
                                                    _route_prediction()
                                                              │
                                              ┌──── CategoryConfig (build switch, cancel Nexus, etc.)
                                              └──── Handler (defend_worker_rush, defend_cannon_rush, etc.)
```

---

## Implementation Checklist

### ✅ Done — Core Architecture

- [x] `ReactionManager` class in `bot/managers/reactions.py`
  - [x] Handler registry (`register_handlers()`)
  - [x] Consumer API properties (`is_active`, `is_cheese_response`, `is_early_defensive`, `keep_workers_safe`, `active_reaction_name`, `reaction_start_time`, `category_config`)
  - [x] Core lifecycle: `update()`, `execute()`, `deactivate()`, `force_deactivate()`
  - [x] Internal routing: `_route_prediction()`, `_resolve_handler()`, `_get_handler()`
  - [x] Deactivation: `_check_deactivation()` with strategy-specific + category-default checks
  - [x] Rule-based fallback: `_prediction_from_rules()` reads detection labels
  - [x] Threat-clear grace period: `_enemy_combat_near_bases()` + `CHEESE_THREAT_CLEAR_GRACE`
  - [x] `_should_deactivate_worker_rush()` and `_should_deactivate_cannon_rush()`

- [x] `CategoryConfig` dataclass + `REACTION_CATEGORY_CONFIGS` in `bot/constants.py`
- [x] `CHEESE_THREAT_CLEAR_GRACE = 30.0` constant in `bot/constants.py`

### ✅ Done — Wiring & Consumer Updates

- [x] `bot/bot.py`: Removed 8 old flags, added `self.reaction_manager = ReactionManager()`, replaced dispatch block with `update()` + `execute()`, updated `Mining(keep_safe=...)`, added `register_handlers()` in `on_start()`
- [x] `bot/managers/macro.py`: Reads `reaction_manager.is_cheese_response` (5 locations), `category_config.probe_cap` for worker cap
- [x] `bot/combat/combat.py`: Reads `reaction_manager.is_early_defensive`
- [x] `bot/utilities/game_report.py`: Reads `active_reaction_name`, `reaction_start_time`, `is_cheese_response`
- [x] `bot/utilities/debug.py`: Reads `reaction_manager.is_cheese_response`
- [x] `bot/belief/strategy_belief.py`: Reads `reaction_manager.active_reaction_name` for worker rush check
- [x] `bot/intel/strategy_detect.py`: Removed `_worker_rush_detected_time` and `_not_worker_rush` side effects from `detect_worker_rush()`

### ✅ Done — Handler Refactoring

- [x] `defend_worker_rush()`: Pure per-frame logic, no flag management
- [x] `defend_cannon_rush()`: Pure per-frame logic, no flag management
- [x] `cheese_reaction()`: No-op (build switch handled by CategoryConfig)
- [x] `early_threat_sensor()`: Removed — logic absorbed into `ReactionManager.update()`

### ✅ Done — Dead Code Removal

- [x] Removed from `bot.py`: `_used_cheese_response`, `_transitioned_from_cheese`, `_not_worker_rush`, `_worker_rush_detected_time`, `_cannon_rush_response`, `_cannon_rush_active`, `_cannon_rush_completed`, `_cannon_rush_cleanup_timer`
- [x] Removed `_get_economy_state()` helper (economy-based transition check replaced with threat-clear + attack-commenced)
- [x] Removed economy-based transition check from `_check_deactivation()` (was circular: cheese → low workers → "reduced" economy → blocked transition)

### ✅ Done — Bug Fixes (post-refactoring)

- [x] **Expansion banking fix**: `handle_macro()` now adds `ExpansionController(prioritize=True)` at the top of the MacroPlan when banking for a Nexus. Previously, `BuildWorkers`/`AutoSupply`/`GasBuildingController` would spend minerals before the Nexus could be placed. Also skips `SpawnController` while banking.
- [x] **Cheese probe cap**: `CategoryConfig.probe_cap` is now consumed in `handle_macro()` — `worker_limit = min(worker_limit, cheese_probe_cap)` when `is_cheese_response` is True. Previously, `BuildWorkers` would jump to the standard profile cap, ignoring the cheese cap.
- [x] **Cheese_Reaction_Build**: Removed duplicate `0 GATEWAY @ nat_wall` step (was causing a second Gateway to be built)
- [x] **ConstantWorkerProductionTill**: Changed from 30 to 20 to match `CategoryConfig.probe_cap`

### ⬜ Remaining — Policy Implementation

- [ ] **Policy 3: Proxy Defense** (`cheese/proxy_rax`, `cheese/proxy_gate`) — No handler registered yet. Currently falls back to `cheese_reaction()` (no-op). Needs:
  - Small reaction: battery + chrono stalker/adept
  - Big reaction: scout proxy locations, immortal first, triple chrono gateway
  - Handler function: `defend_proxy()`
  - Register in `ReactionManager._handlers[StrategyCategory.CHEESE]["proxy_rax"]` and `["proxy_gate"]`

- [ ] **Policy 4: All-In Defense** — `CategoryConfig` exists (probe_cap=44, hold_army=True) but no handler. Needs:
  - Handler function: `defend_all_in()`
  - Wall-off logic for ravager/ling
  - Battery overcharge for chargelot all-in
  - Register in `ReactionManager._handlers[StrategyCategory.ALL_IN]`

- [ ] **Policy 5: Air Switch** — Not in `REACTION_CATEGORY_CONFIGS` yet. Needs:
  - New `StrategyCategory.AIR` or handle via composition belief → tactical handler
  - Stargate production, phoenix count, scout-then-commit logic

### ⬜ Remaining — Minor Improvements

- [ ] **`stop_gas_below` not consumed**: `CategoryConfig.stop_gas_below` is defined but `get_optimal_gas_workers()` still hardcodes `21`. Should read `bot.reaction_manager.category_config.stop_gas_below` so ALL_IN's `stop_gas_below=0` works when implemented.
- [ ] **`is_early_defensive` = `is_cheese_response`**: Currently identical. Should diverge when ALL_IN handler is added (ALL_IN should hold army but not use CHEESE_DEFENSE_ARMY).
- [ ] **`force_deactivate()`**: Listed in plan but not yet implemented. Should be callable from outside (e.g., when game transitions to late-game macro).

---

## Policy 1: Cheese Response (12_pool, proxy structures, any early all-in cheese)

**Trigger labels:** `cheese/12_pool`, `cheese/proxy_rax`, `cheese/proxy_gate`, `cheese/worker_rush`, any `cheese/*` not caught by specific policies

**PiG's reaction:**
1. Stop probes at 20, don't build Nexus
2. 2nd pylon → 2nd gate → Core
3. Pump 2 zealots, then stalkers/adepts
4. Battery on natural
5. 3rd pylon
6. Nexus + resume probes once money is stable
7. Resume standard build order

**Implementation status:** ✅ Active. `CategoryConfig.CHEESE` switches to `Cheese_Reaction_Build`, caps probes at 20, cancels Nexus, uses `CHEESE_DEFENSE_ARMY`. Transition to standard army when threat clears for 30s, game reaches mid-game, or attack commences.

---

## Policy 2: Cannon Rush

**Trigger label:** `cheese/cannon_rush`

**PiG's reaction:**
1. If no enemy Forge visible → ignore the pylons (they're just vision, not cannon rush)
2. Pull workers to kill in priority order: complete/near-complete Cannons → Probes → Pylons
3. Battery on natural

**Implementation status:** ✅ Active. `defend_cannon_rush()` handles worker pulling and target priority. Deactivates when no enemy probes/cannons/pylons near base.

---

## Policy 3: Proxy Defense (proxy rax, proxy gate)

**Trigger labels:** `cheese/proxy_rax`, `cheese/proxy_gate`

**Small reaction (1 proxy building, opponent still expanding):**
1. Make a battery on natural
2. Continue build as normal
3. Chrono a stalker or adept

**Big reaction (multiple proxy buildings or no expansion):**
1. Send probe to scout proxy locations
2. Go immortal first (skip observer)
3. Triple chrono on gateway
4. Still expand on time

**Implementation status:** ⬌ Not implemented. Falls back to `cheese_reaction()` (no-op). Needs handler registration and implementation.

---

## Policy 4: All-In Defense (2-base push, ravager/ling, chargelot all-in, BC rush)

**Trigger labels:** `all_in/one_base_all_in`, `all_in/two_base_all_in`, `all_in/battlecruiser_rush`, `all_in/roach_rush`

**PiG's reaction:**
1. Stop probing at 2-base saturation (44 probes max, at most 10 on 3rd base)
2. Mass gateway units non-stop
3. Get charge and gateways up
4. Use colossus to engage: force siege → pull back → poke forward → engage when unsieged
5. Use observers to track army movement
6. Charge is critical — closes distance AND reduces tank damage
7. For ravager/ling specifically: full wall-off with buildings + battery, immortal over observer
8. For chargelot all-in specifically: save 50 energy for battery overcharge, hold-position probes to block zealots, chrono robos hard for colossus

**Implementation status:** ⬌ `CategoryConfig.ALL_IN` exists (probe_cap=44, hold_army=True) but no handler. Needs `defend_all_in()` implementation.

---

## Policy 5: Air Switch (mutalisks, skytoss, BCs, turtle into air)

**Trigger labels:** Detected via composition belief (air units + air tech buildings), not via strategy label

**PiG's reaction:**
1. 2 stargates → phoenix production
2. Battery in each base
3. If scouted early (spire at ≤50% completion): double chrono both stargates → 4 phoenix before mutas arrive (4 phoenix beat 8 mutas)
4. Then scout again — are they continuing air or switching to ground?
5. **Continuing air:** 3rd stargate, fleet beacon, air attack + armor upgrades, mass phoenix with range
6. **Switching to ground:** Cut phoenix production, focus on colossus/disruptor/standard robo army, use remaining phoenix for harass and picking up units
7. **Vs BCs specifically:** Void rays (but vulnerable to mines and yamato), or tempests with observer vision for hit-and-run, or blink all-in to kill them before they scale
8. **Vs turtle into air:** Option 1 — blink all-in to kill them; Option 2 — macro up (3rd+4th base, 80 probes, 2 stargates, air upgrades)

**Implementation status:** ⬌ Not in `REACTION_CATEGORY_CONFIGS` yet. Needs category definition and handler.

---

## Coverage Check

| Area | Policy | B2GM Source | Implementation |
|------|--------|-------------|----------------|
| Cheese / early all-in | Policy 1 | Standard Cheese Reaction section | ✅ Active |
| Cannon rush | Policy 2 | "Vs Cannon Rush" section | ✅ Active |
| Proxy rax/gate | Policy 3 | "Response to proxy rax" + "Big reaction / Small reaction" | ⬌ No handler |
| All-in (2-base push, ravager/ling, chargelot, BC) | Policy 4 | Multiple game lessons | ⬌ Config only, no handler |
| Air (mutas, skytoss, BCs, turtle) | Policy 5 | "MUTALISKS" section | ⬌ Not started |