# Sentry Guardian Shield — Influence Map Placement

Use a friendly-density influence map to move each Sentry to the spot in the army that maximizes Guardian Shield coverage. Active shields zero out the influence in their radius, so additional Sentries naturally seek the next densest uncovered pocket and the squad converges on full coverage.

### Deployment Rules

1. **Range is the trigger.** GS is only considered when the squad faces a ranged enemy (existing check, unchanged).
2. **No redundant deployment.** A Sentry deploys *only* if there is uncovered friendly density it can cover. If the squad is already at maximum coverage (every pocket already under an active shield), no Sentry deploys — the energy stays banked.
3. **One Sentry per uncovered pocket.** Each active shield zeros its radius on the density grid; the next Sentry only deploys if a non-zero pocket remains. When the grid is fully zeroed, the loop stops.

The influence map is purely a *placement + coverage* mechanism — it decides where each Sentry goes and which ones are needed to reach full coverage, never casting a Sentry into an already-covered area.

---

## Background

### Current State

Guardian Shield (GS) is handled by `_compute_guardian_shield_assignments()` in `bot/combat/combat.py:122` and executed in `micro_sentry()` at `bot/combat/unit_micro.py:900`.

- **Cast trigger:** any squad with ranged enemies present → cast. This is correct and stays as-is. Range detection already exists (`combat.py:147` — `u.ground_range > MELEE_RANGE_THRESHOLD` over the nearby-enemy set from `get_units_in_range`).
- **Redundancy guard:** the greedy loop stops when every squad unit is within `GUARDIAN_SHIELD_RADIUS` of an active shield — i.e. full coverage. No Sentry is approved once coverage is maxed. This behavior is preserved (the influence-map loop terminates on the same condition, expressed as "density grid fully zeroed").
- **Sentry selection:** greedy set-cover. Already-shielded Sentries are free coverage sources; then the candidate closest to the center-of-mass of *uncovered* squad units is approved, repeating until full coverage or candidates exhausted.
- **Sentry movement:** `micro_sentry()` follows `ranged_center` (center of ranged units) during combat, else `squad_position`. All Sentries follow the *same* point, so multiple Sentries stack on the ranged line even when the squad is spread across a concave.
- **Overlap guard:** `GUARDIAN_SHIELD_OVERLAP_DISTANCE = 8.0` exists in `constants.py` but is *not* consulted by the current assignment function — overlap is only prevented implicitly via the greedy set-cover.

### The Gap

The cast trigger is right. The *placement* is what's missing:

| Concern | Current | Desired |
|---|---|---|
| Cast trigger | Ranged enemy present → cast | **Unchanged** — ranged enemy present → cast |
| Redundant deployment | Loop stops at full coverage | **Unchanged** — Sentry deploys only if an uncovered pocket exists; max coverage → no deployment |
| Sentry positioning | All follow one `ranged_center` point | Each Sentry heads for the densest *uncovered* pocket in the squad |
| Multi-Sentry spread | Implicit via set-cover center-of-mass | Explicit via influence map: shielded areas zeroed, next Sentry picks a new peak |
| Coverage convergence | Greedy stops at "every unit in radius of a shield" | Same coverage goal, but driven by density peaks so Sentries spread to where the units actually are |

### Existing Influence-Map Patterns in the Bot

Two precedents to reuse:

1. **Disruptor Nova targeting** (`bot/utilities/use_disruptor_nova.py:62`, `nova_manager.py:289`)
   - Reads `bot.mediator.get_tactical_ground_grid` (ARES-built grid of friendly army value, negative-weighted: `grid_manager.py:241`).
   - Scans a bounded circle around the Disruptor for the **max** grid value.
   - `NovaManager.get_exclusion_mask()` builds a boolean mask of already-targeted radii and **zeros out** those cells before the next Disruptor picks a target — exactly the "zero out after deploy" pattern we want for GS.

2. **ARES ground grid** (`grid_manager.py`, `tutorials/influence_and_pathing.md`)
   - `map_data.get_pyastar_grid()` returns a clean pathing grid (1.0 = walkable, 0.0 = blocked).
   - `map_data.add_cost(position, radius, grid, weight)` adds a circular cost blob — used to paint friendly density onto a custom grid.
   - Custom grids drop straight into ARES pathing behaviors (`PathUnitToTarget`, `KeepUnitSafe`).

### Reusable Scoring

`bot/combat/target_scoring.py` and `ares/dicts/unit_data.py:UNIT_DATA["army_value"]` already give every unit type a numeric value. The same `army_value` that scores enemy targets can score *our own* units for density — no new value table needed. `combat.py:560` already sums `army_value` over enemies for the engagement gate; we mirror that for friendlies.

---

## Proposed Design

### 1. Cast Trigger — Unchanged

GS is deployed whenever the squad faces a ranged enemy. The existing check stays:

```python
has_ranged_enemies = any(u.ground_range > MELEE_RANGE_THRESHOLD for u in enemies)
if not has_ranged_enemies:
    return set()   # no ranged threat → no GS
```

No density floor, no value gate. **Range is the only trigger.** The influence map decides *placement* and *which Sentries are needed for full coverage* — never *whether* to cast.

### 2. Friendly Density Grid (per squad, per frame it has Sentries)

A small, local influence map built only for squads containing Sentries — not a global grid.

```
clean = map_data.get_pyastar_grid()            # 1.0 walkable, 0.0 blocked
for u in squad_units:
    if u.type_id in GS_IGNORE_TYPES:           # workers, observers, hallucinations
        continue
    value = UNIT_DATA[u.type_id]["army_value"] # reuse existing value table
    clean = map_data.add_cost(u.position, GS_INFLUENCE_RADIUS, clean, weight=value)
```

- **Why local, not global:** GS only matters for the squad the Sentries are in. Building one ~army-sized grid per Sentry-bearing squad is far cheaper than a map-wide grid, and avoids polluting other subsystems.
- **Why `army_value`:** it already encodes "how much is this unit worth protecting" and is the same metric the engagement gate uses — consistent decision-making. The density value is used only to *rank* pockets against each other (find the densest), not to gate the cast.
- **`GS_IGNORE_TYPES`:** workers, observers, hallucinations, and Sentries themselves don't benefit from GS meaningfully (workers shouldn't be in combat; Sentries don't shoot). Keeps the density signal clean.
- **Perf note:** one `add_cost` call per squad unit is O(n) in squad size; `add_cost` internally paints a numpy circle. For a 40-unit squad with a 4.5-radius brush this is ~40 small numpy ops — well under a millisecond, and only runs for Sentry-bearing squads.

### 3. Active-Shield Zeroing

After painting friendly density, zero out areas already covered by an active (or just-approved-this-frame) shield:

```
for s in sentries:
    if s.has_buff(BuffId.GUARDIANSHIELD) or s.tag in approved_this_frame:
        mask = circular_mask(s.position, GUARDIAN_SHIELD_RADIUS, grid.shape)
        density_grid[mask] = 0.0
```

This is the `NovaManager.get_exclusion_mask` pattern, applied additively. After zeroing, the highest peak in `density_grid` is the densest *uncovered* pocket — which is where the next Sentry should go. The loop naturally terminates when the grid is fully zeroed (full coverage achieved).

### 4. Sentry-by-Sentry Assignment (replaces greedy set-cover)

```
approved = set()
peaks: dict[int, Point2] = {}
remaining_candidates = [s for s in sentries if s.energy >= GS_COST and not s.has_buff(GS)]
density = build_density_grid(squad_units)

# Already-shielded Sentries zero their area first (free coverage)
for s in sentries:
    if s.has_buff(BuffId.GUARDIANSHIELD):
        density[circular_mask(s.position, GS_RADIUS)] = 0.0

while remaining_candidates and density.max() > 0:
    peak_pos = argmax_position(density)
    best_sentry = min(remaining_candidates, key=lambda s: dist_sq(s.position, peak_pos))
    approved.add(best_sentry.tag)
    peaks[best_sentry.tag] = peak_pos
    # Zero out the area this Sentry will cover
    density[circular_mask(peak_pos, GS_RADIUS)] = 0.0
    remaining_candidates.remove(best_sentry)
```

- Each iteration picks the densest *remaining* uncovered pocket and assigns the closest capable Sentry to it.
- Loop runs until either no candidates remain **or the density grid is fully zeroed** (full coverage) — same coverage goal as the greedy set-cover, but driven by where the units actually cluster. This is the direct analog of the Disruptor Nova `get_exclusion_mask` flow: a Nova doesn't fire if no unexcluded enemy clump remains, and a Sentry doesn't deploy if no uncovered friendly pocket remains.
- Returns both the approved set (for the cast decision) and a `{tag: peak_pos}` mapping (for movement).

### 5. Sentry Movement Change in `micro_sentry()`

Currently `micro_sentry()` (`unit_micro.py:989`) follows `ranged_center` (or `squad_position`). Add an optional `gs_target: Point2 | None` parameter:

- If the Sentry was **approved this frame** and has a `peak_pos`, path to `peak_pos` via `PathUnitToTarget` with `success_at_distance = GUARDIAN_SHIELD_RADIUS` so it stops once in coverage range.
- Otherwise, keep the existing `ranged_center` / `squad_position` follow behavior.

This keeps the safe-follow behavior for non-casting Sentries while letting casting Sentries peel off to their assigned pocket. The `KeepUnitSafe` dodge layers stay first in the maneuver chain — safety always wins over placement.

---

## Files to Modify

| File | Change |
|---|---|
| `bot/combat/combat.py` | Replace `_compute_guardian_shield_assignments()` body with the influence-map version. Build density grid, zero already-shielded areas, iterate peak-pick-and-zero. Return `{tag}` plus a `{tag: peak_pos}` mapping for movement. Cast trigger (ranged enemy check) unchanged. |
| `bot/combat/unit_micro.py` | `micro_sentry()` accepts `gs_target: Point2 \| None`. When set, `PathUnitToTarget` targets it instead of `ranged_center`. |
| `bot/constants.py` | Add `GS_INFLUENCE_RADIUS`, `GS_IGNORE_TYPES`. |

No new files, no new dependencies, no ARES src changes.

## New Constants

```python
# ===== GUARDIAN SHIELD INFLUENCE MAP =====
GS_INFLUENCE_RADIUS = 4.5
"""Radius each friendly unit paints on the GS density grid.
Matches GUARDIAN_SHIELD_RADIUS — a pocket is "dense" at the scale GS covers."""

GS_IGNORE_TYPES = {
    UnitTypeId.PROBE,
    UnitTypeId.OBSERVER,
    UnitTypeId.HALLUCINATION_PHOENIX,
    UnitTypeId.HALLUCINATION_ADEPT,
    UnitTypeId.SENTRY,
    # ... other hallucinations
}
"""Unit types excluded from the GS density grid — they don't benefit from GS
or shouldn't be in combat anyway. Keeps the density signal clean."""
```

---

## Phased Implementation

### Phase 1: Density Grid + Active-Shield Zeroing (no movement change)

**Goal:** use the influence map to pick *which* Sentry casts, without changing Sentry movement.

- Build the per-squad density grid in `_compute_guardian_shield_assignments()`.
- Zero already-shielded areas.
- Replace greedy set-cover with peak-pick-and-zero loop.
- Keep `micro_sentry()` movement exactly as-is (all Sentries still follow `ranged_center`).

**Risk:** low. Only *which* Sentry casts changes; movement is untouched. Replay-testable in isolation.

### Phase 2: Sentry Movement to Assigned Peak

**Goal:** casting Sentries path to their assigned `peak_pos` so they spread to cover different pockets.

- Add `gs_target` param to `micro_sentry()`.
- `PathUnitToTarget` to `gs_target` with `success_at_distance = GUARDIAN_SHIELD_RADIUS`.
- Non-casting Sentries keep current follow behavior.

**Risk:** medium. Sentries may briefly separate from the army to reach their pocket. `KeepUnitSafe` layers must remain first in the maneuver chain so a Sentry never walks into danger to reach a peak. Replay-test for Sentries getting picked off while repositioning.

---

## Testing

- **Phase 1 unit test:** two clumps of friendly units, two Sentries, ranged enemies present → assert both Sentries approved, each assigned to a different clump's peak (tags differ, peaks differ). Single clump, one Sentry → assert one approved, peak at the clump.
- **Phase 2 unit test:** approved Sentry with `gs_target` issues `PathUnitToTarget` to that target; non-approved Sentry falls back to `ranged_center`.
- **Replay checks:** watch for (a) multiple Sentries visibly splitting to cover a concave rather than stacking on ranged center, (b) full squad coverage achieved with minimum Sentries, (c) no Sentry walking to death during repositioning, (d) GS still firing against any ranged enemy (trigger unchanged).

---

## Open Questions

1. **Should the density grid weight low-HP units higher?** ARES's `tactical_ground_grid` multiplies influence by 1.5 when `health <= 100` (`grid_manager.py:238`). Mirroring this for GS would prioritize protecting damaged units — but GS reduces incoming ranged damage, so protecting *healthy* high-value clumps (Stalkers, Immortals) may be more valuable than chasing low-HP units that may die anyway. **Proposal:** start without the HP multiplier; add later if replays show Sentries consistently heading to the wrong pocket.

2. **Should we path the Sentry to `peak_pos` or to a point between its current position and `peak_pos`?** A Sentry on the far side of the squad repositioning through the enemy front line is a liability. **Proposal:** `PathUnitToTarget` uses `get_ground_grid` (enemy influence), so it will route around danger — but we should still cap the reposition distance and fall back to `ranged_center` follow if the path is too long.

---

## Shadow Review

1. **Biggest assumption:** that `army_value` is a good proxy for "where do units cluster" — since we're using it for *placement* (not a cast gate), the exact weight matters less than the *spatial* distribution it produces. Even a uniform weight per unit would find the same pockets; `army_value` just breaks ties toward higher-value clusters.
2. **Most likely failure:** Sentries repositioning into danger during Phase 2. `KeepUnitSafe` is first in the chain, but `PathUnitToTarget` can still issue a move command that `KeepUnitSafe` then overrides — net effect may be a Sentry that stutters toward danger and back. Needs replay validation before competition.
3. **Smallest robustness fix (≤10 LOC):** cap reposition distance — if `cy_distance_to(sentry.position, peak_pos) > GS_MAX_REPOSITION_DISTANCE`, fall back to `ranged_center` follow for this Sentry this frame. Prevents cross-army suicide runs.