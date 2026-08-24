# Combat Intel Consolidation Plan

**Created:** 2026-08-17
**Status:** Draft (not yet implemented)
**Related:** `plan/expansion_improvement_plan.md` (race-specific expansion signals, depends on this), `plan/bayesian_belief_layer_plan.md` (enemy perception layer)

## Problem This Solves

The bot's **own-side battlefield posture** is scattered across ~7 independent flags on the `bot` instance, owned by 3 different subsystems, with no unified read API:

| Flag | Written by | File:line | Read by (consumers) |
|------|-----------|-----------|---------------------|
| `_under_attack` | `threat_detection` (primary) + `ReactionManager` (force-set) | `reactions.py:1147-1154`, `:253`, `:325` | combat (`:1812`), macro (`:836`, `:1666`), scouting (`:309`, `:331`), reactions (`:385`), debug (`:77`), game_report (`:255`, `:396`) |
| `_commenced_attack` | `handle_attack_toggles` via `_set_attack_state` | `combat.py:1801` | combat (`:702`, `:748`, `:927`, `:1845`), macro (`:865`), reactions (`:378`), debug (`:77`), game_report (`:248`, `:395`, `:757`) |
| `_main_army_defending` | `threat_detection` | `reactions.py:1215` | (currently no external readers — written but unused outside reactions) |
| `_defender_threat_position` | `threat_detection` | `reactions.py:1212` | combat (`:1812`, `:1814`, `:2109`), debug (`:885`) |
| `_enemy_army_ever_seen` | `update_enemy_intel_tracking` | `intel_quality.py:121`, `:126` | macro (`:870`), game_report |
| `_last_enemy_army_visible_time` | `update_enemy_intel_tracking` | `intel_quality.py:122` | macro (`:877`) |
| `_intel_urgency` | `update_enemy_intel_tracking` | `intel_quality.py:138`, `:140` | macro (`:888`), scouting (urgency thresholds) |

**Consequences:**

1. **`_has_map_control()`** (macro.py:812-895) — the closest thing to a battlefield-state query — reads 5 of these flags from 3 different owners plus calls `assess_threat()`. It's an 84-line ad-hoc aggregation sitting in macro.py because that's where its only consumer (`expansion_checker`) lives. It answers one yes/no question ("safe to expand?") rather than exposing reusable state.

2. **No reusable posture query.** A consumer that wants "are we currently on offense or defense?" must independently read `_commenced_attack`, `_under_attack`, and `_main_army_defending` — three flags from two modules — and reconcile them. Each consumer does this ad-hoc or avoids the question.

3. **Ownership is split.** `threat_detection` (reactions.py) owns `_under_attack` + `_defender_threat_position` + `_main_army_defending`. `handle_attack_toggles` (combat.py) owns `_commenced_attack`. `update_enemy_intel_tracking` (intel_quality.py) owns the three intel flags. Moving any single flag breaks the others; there's no cohesive boundary.

4. **`expansion_improvement_plan.md` (race-specific signals)** is blocked on this consolidation — it adds more queries ("race-aware all-in detection", "enemy army supply comparison") that need the same battlefield posture as `_has_map_control`, and they shouldn't each re-derive it.

## What This Plan Is — and Isn't

**Is:** Consolidate the 7 own-side posture flags into a single `CombatIntel` object living in `bot/intel/combat_intel.py`. Provide a read API for derived battlefield queries (`has_map_control()`, `is_on_offense()`, `is_under_pressure()`, etc.). Move the flag *updates* into the object so it owns its state. Existing subsystems (reactions, combat, intel_quality) become *feeders* that call `combat_intel.update_*()` instead of writing `bot._*` directly.

**Isn't:**
- Not touching the **enemy perception** side — that's the Belief layer (`bot/belief/`), which is a separate, already-in-progress consolidation. `CombatIntel` will *consume* `bot.belief_state` where needed but won't own it.
- Not touching `ReactionManager` — it owns *reaction* state (active category, strategy, transition), which is a different concern. `CombatIntel` owns *posture* state (under attack? attacking? defending? intel freshness?). `ReactionManager` may read `CombatIntel` but they remain separate objects.
- Not touching `enemy_timings.py` or `strategy_detect.py` — those are observation/classification systems, not posture.
- Not changing any **gameplay logic** — this is a refactor that preserves identical behavior. The same thresholds, the same hysteresis, the same decision outcomes. The only change is *where* the state lives and *how* it's read.

## Architecture Overview

```
BEFORE:
  reactions.py threat_detection  ──writes──→  bot._under_attack
                                              bot._defender_threat_position
                                              bot._main_army_defending
  combat.py handle_attack_toggles ──writes──→ bot._commenced_attack
  intel_quality.py update_tracking──writes──→ bot._enemy_army_ever_seen
                                              bot._last_enemy_army_visible_time
                                              bot._intel_urgency

  Consumers read bot._* flags directly (each does its own ad-hoc aggregation)

AFTER:
  reactions.py threat_detection  ──calls──→ combat_intel.update_threat_state(...)
  combat.py handle_attack_toggles ──calls──→ combat_intel.update_attack_state(...)
  intel_quality.py update_tracking──calls──→ combat_intel.update_intel_state(...)

  bot.combat_intel (CombatIntel instance)
    ├── .under_attack: bool
    ├── .commenced_attack: bool
    ├── .main_army_defending: bool
    ├── .defender_threat_position: Point2 | None
    ├── .enemy_army_ever_seen: bool
    ├── .last_enemy_army_visible_time: float
    ├── .intel_urgency: float
    ├── .has_map_control() -> bool          # derived query (moved from macro.py)
    ├── .is_on_offense() -> bool            # derived query (new, replaces ad-hoc reads)
    ├── .is_under_pressure() -> bool        # derived query (new)
    └── .can_expand_natural_under_attack() -> bool  # moved from macro.py

  Consumers read bot.combat_intel.* (single object, typed access)
```

### Key Principle: State Owner, Not Logic Owner

`CombatIntel` **owns the state** and the **derived queries** over that state. It does **not** own the *decision logic* that sets the state — `threat_detection` still decides *when* we're under attack (the hysteresis, the thresholds); it just calls `combat_intel.set_under_attack(True)` instead of `bot._under_attack = True`. Similarly, `handle_attack_toggles` still decides *when* to commence attack; it calls `combat_intel.set_attack_state(True)`. This keeps the decision logic in its current module (close to the combat/threat context it needs) while consolidating the *state* in one place.

### Key Principle: Backward-Compatible Flag Shim

To avoid a big-bang migration of ~25 read sites across 6 files, Phase 1 keeps `bot._under_attack` etc. as **read-through properties** that delegate to `combat_intel`. This means existing `if bot._under_attack:` reads keep working unchanged. Phases 2-3 migrate the reads to `bot.combat_intel.under_attack` one module at a time. Phase 4 removes the shims.

### Key Principle: No Behavior Change

Every threshold, every hysteresis band, every boolean outcome must be **identical** before and after. This is verifiable by running the same games with the same seeds and diffing the debug overlay. If any decision flips, the refactor is wrong.

---

## The 4 Phases

### Phase 1: Create `CombatIntel` — State + Updates + Flag Shims ✅ TARGET

**What:** Create the `CombatIntel` class, wire it as the single write target, and add backward-compatible `bot._*` properties that delegate to it. Zero changes to read sites.

**Files to create:**
- `bot/intel/combat_intel.py` — the `CombatIntel` class

**Files to modify:**
- `bot/bot.py` — instantiate `self.combat_intel = CombatIntel()` in `__init__`; replace the 7 `self._*` attribute declarations with `@property` shims that read/write through to `self.combat_intel`
- `bot/managers/reactions.py` — `threat_detection()` writes via `bot.combat_intel.set_under_attack()` / `set_defender_threat_position()` / `set_main_army_defending()` instead of `bot._under_attack = ...` etc. Also `ReactionManager._route_prediction()` (`:325`) and `deactivate()` (`:253`) write via `combat_intel`.
- `bot/combat/combat.py` — `_set_attack_state()` closure (`:1792-1808`) calls `bot.combat_intel.set_attack_state()` instead of `bot._commenced_attack = new_value`. Telemetry accumulators (`_attack_initiation_count`, `_total_attack_time`, `_current_attack_start`) stay on bot — they're telemetry, not posture.
- `bot/intel/intel_quality.py` — `update_enemy_intel_tracking()` writes via `bot.combat_intel.set_intel_state()` instead of `bot._enemy_army_ever_seen = ...` etc.
- `bot/intel/__init__.py` — re-export `CombatIntel`

**`CombatIntel` class sketch:**

```python
@dataclass
class CombatIntel:
    """Owns the bot's own-side battlefield posture state.

    Fed by three subsystems:
      - threat_detection (reactions.py) → under_attack, defender_threat_position, main_army_defending
      - handle_attack_toggles (combat.py) → commenced_attack
      - update_enemy_intel_tracking (intel_quality.py) → enemy_army_ever_seen, last_enemy_army_visible_time, intel_urgency

    Provides derived queries (has_map_control, is_on_offense, etc.) so consumers
    don't each re-derive posture from raw flags.
    """
    under_attack: bool = False
    commenced_attack: bool = False
    main_army_defending: bool = False
    defender_threat_position: Optional[Point2] = None
    enemy_army_ever_seen: bool = False
    last_enemy_army_visible_time: float = 0.0
    intel_urgency: float = 0.0

    # ── Update methods (called by the owning subsystems) ──
    def set_under_attack(self, value: bool) -> None: ...
    def set_defender_threat_position(self, pos: Optional[Point2]) -> None: ...
    def set_main_army_defending(self, value: bool) -> None: ...
    def set_attack_state(self, value: bool) -> None: ...
    def set_intel_state(self, ever_seen: bool, last_visible_time: float, urgency: float) -> None: ...

    # ── Derived queries (read by consumers) ──
    def has_map_control(self, bot) -> bool: ...
    def can_expand_natural_under_attack(self, bot) -> bool: ...
    def is_on_offense(self) -> bool: ...
    def is_under_pressure(self) -> bool: ...
```

**Flag shim pattern in `bot.py`:**

```python
@property
def _under_attack(self) -> bool:
    return self.combat_intel.under_attack

@_under_attack.setter
def _under_attack(self, value: bool) -> None:
    self.combat_intel.under_attack = value
```

This means **all existing `bot._under_attack` reads and writes work unchanged** — they just route through `combat_intel` under the hood. The 7 flags become thin properties. Once all writers are migrated to `combat_intel.set_*()` calls, the setters become no-ops (or can raise to catch stragglers in dev).

**LOC estimate:** ~120 in `combat_intel.py`, ~50 in `bot.py` (7 property pairs), ~30 across the 3 feeder modules. Total ~200.

**Risk:** Low. The flag shims guarantee zero behavior change. The only risk is a property shadowing mistake in `bot.py` `__init__` (e.g., if `self.combat_intel` isn't set before a shim is accessed). Order `__init__` carefully: `self.combat_intel = CombatIntel()` must come before any code that touches the shimmed flags.

**Validation:** Run 3 test games with identical seeds pre/post refactor. Diff the debug overlay text. Zero differences expected.

---

### Phase 2: Move `_has_map_control()` + `_can_expand_natural_under_attack()` into `CombatIntel`

**What:** Relocate the two map-control functions from `macro.py` into `CombatIntel` as derived query methods. Update `expansion_checker` to call `bot.combat_intel.has_map_control(bot)`.

**Why after Phase 1:** Phase 1 makes `CombatIntel` the state owner. Phase 2 makes it the *query* owner. The functions currently read 5 flags — after Phase 1, those flags are `combat_intel` attributes, so the methods become simple internal reads (no `bot.` prefix needed for the posture flags; only `bot` is passed for ARES mediator access like `get_own_nat`, `get_ground_enemy_near_bases`, and `assess_threat`).

**Files to modify:**
- `bot/intel/combat_intel.py` — add `has_map_control(bot)` and `can_expand_natural_under_attack(bot)` methods (moved from macro.py, logic unchanged)
- `bot/managers/macro.py` — delete `_has_map_control()` (`:812-895`) and `_can_expand_natural_under_attack()` (`:787-809`); replace call site at `:924` with `bot.combat_intel.has_map_control(bot)`; replace call at `:837` and `:1666` with `bot.combat_intel.can_expand_natural_under_attack(bot)`
- `bot/managers/macro.py` — remove now-unused imports: `assess_threat`, `THREAT_BLOCK_EXPANSION_LEVEL`, `EXPANSION_INTEL_URGENCY_BLOCK`, `EXPANSION_PASSIVE_ENEMY_TIME` (move to `combat_intel.py` imports)

**LOC change:** ~-90 in macro.py, ~+90 in combat_intel.py. Net ~0.

**Risk:** Low. Pure relocation. The functions get `self.` access to the 5 flags instead of `bot._*` access. The `assess_threat()` call and ARES mediator access still go through `bot`.

**Validation:** Same seed-diff approach. Expansion timing must be identical.

---

### Phase 3: Migrate Read Sites to `bot.combat_intel.*`

**What:** Replace all `bot._under_attack`, `bot._commenced_attack`, etc. reads with `bot.combat_intel.under_attack`, `bot.combat_intel.commenced_attack`, etc. Remove the flag shims from `bot.py`.

**Why after Phase 2:** Phases 1-2 prove the object works. Phase 3 cleans up the backward-compat layer. This is the tedious but mechanical phase.

**Read sites to migrate (by module):**

| Module | Flag reads | Lines |
|--------|-----------|-------|
| `combat/combat.py` | `_commenced_attack` (×5), `_under_attack` (×1), `_defender_threat_position` (×3) | `:702`, `:748`, `:927`, `:1812`, `:1814`, `:1845`, `:2109` |
| `managers/macro.py` | `_under_attack` (×2 — already migrated to `combat_intel.has_map_control` in Phase 2, but `:1666` `blocked_by_attack` still reads directly) | `:1666` |
| `managers/scouting.py` | `_under_attack` (×2) | `:309`, `:331` |
| `managers/reactions.py` | `_under_attack` (×2 — internal reads in `ReactionManager`), `_commenced_attack` (×1) | `:385`, `:378` |
| `utilities/debug.py` | `_commenced_attack` (×1), `_under_attack` (×1), `_defender_threat_position` (×1) | `:77`, `:885` |
| `utilities/game_report.py` | `_commenced_attack` (×3), `_under_attack` (×2) | `:248`, `:255`, `:395`, `:396`, `:757` |
| `bot/bot.py` | `_commenced_attack` (comment at `:437`) | `:437` |

**Files to modify:** 7 files, ~20 read sites. Each is a mechanical `bot._flag` → `bot.combat_intel.flag` rename.

**Files to modify (cleanup):**
- `bot/bot.py` — remove the 7 `@property` shim pairs from Phase 1

**LOC change:** ~-70 (shim removal), ~+0 (reads are same length). Net ~-70 (cleanup).

**Risk:** Medium. This is the most touch-heavy phase. Missing a read site would cause an `AttributeError` at runtime (which is good — it fails loud, not silent). The risk is a read site hidden behind a `getattr(bot, '_flag', default)` (which wouldn't crash, just silently return the default). Grep for `getattr.*_under_attack` / `getattr.*_commenced_attack` before starting.

**Validation:** Full test suite + seed-diff games. Every `getattr` usage must be migrated to `getattr(bot.combat_intel, 'flag', default)` or direct access.

---

### Phase 4: Add New Derived Queries + Enable `expansion_improvement_plan.md`

**What:** Now that `CombatIntel` is the single posture read API, add the new derived queries that were previously blocked, and wire the race-specific expansion signals from `expansion_improvement_plan.md`.

**New derived queries:**

```python
def is_on_offense(self) -> bool:
    """Army is attacking and not under attack at home."""
    return self.commenced_attack and not self.under_attack

def is_under_pressure(self) -> bool:
    """Under attack or defending — don't expand, don't cut army."""
    return self.under_attack or self.main_army_defending

def is_intel_stale(self) -> bool:
    """Intel urgency above the expansion-block threshold."""
    return self.intel_urgency >= EXPANSION_INTEL_URGENCY_BLOCK

def time_since_enemy_seen(self) -> float:
    """Seconds since last direct vision of enemy army."""
    return bot.time - self.last_enemy_army_visible_time
```

**Race-specific expansion signals** (from `expansion_improvement_plan.md`):
- Race-aware expansion mirroring (#1)
- Race-aware all-in detection (#2)
- Enemy worker economy comparison (#3)
- Enemy army supply comparison (#4)

These become additional gates/positive signals inside `has_map_control()`, which is now a method on `CombatIntel`. They need `bot` for `bot.enemy_race`, `bot._enemy_worker_count`, `bot._enemy_army_supply`, etc. — the prerequisite is `track_enemy_timings` running every frame (see `expansion_improvement_plan.md` Prerequisite section).

**Files to modify:**
- `bot/intel/combat_intel.py` — add derived queries + race-specific gates in `has_map_control()`
- `bot/bot.py` — move `track_enemy_timings` call outside the build-order-only block (prerequisite from expansion plan)
- `bot/constants.py` — add race-specific thresholds (e.g., `EXPANSION_MIRROR_ZERG_DEFICIT = 2`)

**LOC estimate:** ~60 in `combat_intel.py`, ~5 in `bot.py`, ~10 in `constants.py`.

**Risk:** Medium. The race-specific signals are *new gameplay logic*, not a refactor. Each should be feature-gated or tested in isolation. This phase crosses from "refactor" into "gameplay change" — treat each signal as its own mini-proposal with tuning.

**Validation:** Playtest vs each race. The race-specific thresholds need tuning games, not just seed-diffs.

---

## Module Layout

```
bot/intel/
  __init__.py                 # Re-exports CombatIntel + existing intel API
  combat_intel.py             # NEW: CombatIntel class (posture state + derived queries)
  enemy_timings.py            # UNCHANGED: raw observation recording
  strategy_detect.py          # UNCHANGED: cheese/strategy classification
  intel_quality.py            # MODIFIED: writes via combat_intel.set_intel_state()
```

`combat_intel.py` imports:
- `from bot.constants import THREAT_BLOCK_EXPANSION_LEVEL, EXPANSION_INTEL_URGENCY_BLOCK, EXPANSION_PASSIVE_ENEMY_TIME`
- `from bot.managers.reactions import assess_threat` (for `has_map_control` Gate 2)
- `from ares.consts import UnitTypeId` (for `can_expand_natural_under_attack` battery check)
- `from cython_extensions import cy_distance_to` (for battery distance check)

**Circular import check:** `combat_intel.py` → `reactions.py` (for `assess_threat`). `reactions.py` → `combat_intel.py`? No — `reactions.py` calls `bot.combat_intel.set_*()` at runtime, not at import time. The import is one-directional: `combat_intel` imports `assess_threat` from `reactions`. If this causes a cycle (reactions imports something that imports combat_intel), use a lazy import inside `has_map_control()` — same pattern as `constants.py:_get_economy_state()`.

---

## Impact Map

### Phase 1
- **Touched files:** `combat_intel.py` (new), `bot.py`, `reactions.py`, `combat.py`, `intel_quality.py`, `intel/__init__.py`
- **External callers/interfaces affected:** None — flag shims preserve the `bot._*` API
- **Risk of regression:** Low — shims are transparent

### Phase 2
- **Touched files:** `combat_intel.py`, `macro.py`
- **External callers/interfaces affected:** `expansion_checker` calls `bot.combat_intel.has_map_control()` instead of local function
- **Risk of regression:** Low — pure relocation

### Phase 3
- **Touched files:** `combat.py`, `macro.py`, `scouting.py`, `reactions.py`, `debug.py`, `game_report.py`, `bot.py`
- **External callers/interfaces affected:** All `bot._*` posture flag reads change to `bot.combat_intel.*`
- **Risk of regression:** Medium — many touch points; mitigated by loud `AttributeError` on miss

### Phase 4
- **Touched files:** `combat_intel.py`, `bot.py`, `constants.py`
- **External callers/interfaces affected:** `has_map_control()` gains race-specific gates
- **Risk of regression:** Medium — new gameplay logic, needs playtest tuning

---

## Implementation Priority

1. **Phase 1** (state + shims) — safest, highest foundation value. Do this first.
2. **Phase 2** (move map control) — immediate payoff: macro.py shrinks, `CombatIntel` becomes useful.
3. **Phase 3** (migrate reads) — mechanical cleanup. Can be done incrementally, one module per PR.
4. **Phase 4** (new queries + race signals) — gameplay change, separate from refactor. Only after Phases 1-3 are stable.

Phases 1-2 are a clean, self-contained refactor with zero behavior change. Phase 3 is cleanup that can be deferred. Phase 4 is new feature work that *enabled by* this consolidation but not part of it.

---

## Shadow Review

1. **Biggest assumption:** that the 7 flags are the *complete* set of own-side posture state. If there are other posture-adjacent flags (e.g., `_cloaked_threat_positions` at `reactions.py:1175`, or the blink-snipe/chase/focus state machines in combat.py), they may belong in `CombatIntel` too. This plan deliberately scopes to the 7 flags that `_has_map_control` reads + the 2 defense flags that track "is the army defending" — the minimal cohesive set. The snipe/chase/focus state machines are *micro* state, not *posture* state, and should stay in combat.py.

2. **Most likely failure/edge case:** `ReactionManager._route_prediction()` at `reactions.py:325` force-sets `bot._under_attack = True` outside the normal hysteresis. If this write path is missed in Phase 1, the shim will catch it (the setter delegates to `combat_intel`), but if the shim is removed in Phase 3 before this site is migrated, `ReactionManager` will silently fail to set the flag. Must audit all 3 write paths (`threat_detection`, `ReactionManager`, `intel_quality`) in Phase 1.

3. **Smallest change to improve robustness (≤10 LOC):** Add a `CombatIntel.snapshot() -> dict` method that returns all 7 fields as a dict, and log it to telemetry once per 30s. This gives post-game visibility into posture state transitions — currently impossible because the flags are scattered. Costs 5 LOC in `combat_intel.py`, 3 in `game_report.py`.