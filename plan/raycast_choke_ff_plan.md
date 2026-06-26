# Raycast Choke Detection & Force Field Plan

A phased plan to connect choke detection to Force Field placement, with an upstream Cython contribution as the endgame.

---

## Background

### Current State

**Force Fields** (`bot/combat/force_field.py`) only target two ramps:
- `bot.main_base_ramp` — our main ramp
- `bot.mediator.get_enemy_ramp` — enemy main ramp

No map chokes are considered for FF placement. When enemies push through a map choke (not a ramp), the FF system either attempts a generic perpendicular split through the enemy center or does nothing.

**Choke detection** (`bot/utilities/choke_grid.py` + `bot/combat/combat.py`) uses `map_analyzer`'s precomputed `map_chokes` API. It detects chokes and decides whether to engage or hold, but does **not** place Force Fields at them. The engagement policy (hold/push based on who funnels) is tactically sound but doesn't exploit the choke offensively with FFs.

**Wall-off generation** (`bot/utilities/natural_wall_manager.py`, ~1100 lines) has fallen into disrepair. It fights the `map_analyzer` API at every step — using PCA on walkable pixels for direction, bounding-box clustering for width (±50% error), and expanding-circle brute force for building placement.

### The Gap

| Feature | Choke System | FF System | Wall-Off System |
|---------|:-----------:|:--------:|:---------------:|
| Detects map chokes | ✅ | ❌ | ✅ (poorly) |
| Detects ramps | ✅ | ✅ (main only) | ✅ |
| Places FFs at chokes | ❌ | ❌ | N/A |
| Places FFs at ramps | ❌ | ✅ (single FF) | N/A |
| Measures choke width | ±50% error | N/A | ±50% error |
| Knows choke orientation | ❌ | N/A | ❌ (PCA hack) |
| Detects dynamic building chokes | ❌ | ❌ | ❌ |
| Engagement hold/pull-back | ✅ | ❌ | N/A |

### Limitations of `map_analyzer` Choke API

| Limitation | Impact |
|---|---|
| `Polygon.width` is ±50% inaccurate | Computed from outer perimeter extremes, not narrowest point |
| Ramps absorb overlapping RawChokes | Genuine narrow passages near ramps lose precise `min_length` |
| Vision blockers mixed in | `map_chokes` includes `VisionBlockerArea` (bushes/fog) — not filtered in `create_narrow_choke_points` |
| `center` is integer-rounded | 1-tile offset on a 3-tile choke could miss optimal block point |
| `side_a`/`side_b` types inconsistent | `RawChoke` → `Point2`, `VisionBlockerArea` → `tuple[int,int]` |
| No directionality | No passage orientation exposed for FF/wall alignment |
| `is_choke_between` samples only 8 points | 2-tile choke between samples 3.75 tiles apart can be missed |
| Static only | Cannot detect enemy wall-offs or cannon-rush gaps built mid-game |

---

## The Raycasting Solution

Instead of relying solely on `map_analyzer`'s precomputed chokes, cast rays through the **pathing grid** at runtime to find narrow passages — places where a ray crosses a thin strip of walkable tiles bordered by unpathable terrain.

### What Raycasting Fixes

| Current Limitation | Raycasting Solution |
|---|---|
| `width` is ±50% inaccurate | Perpendicular ray march gives exact narrowest width at each tile |
| 8-sample `is_choke_between` misses narrow chokes | Ray marches every tile — no gaps between samples |
| `center` is integer-rounded | Float-precision positions — exact narrowest point |
| No directionality | Ray direction *is* the passage axis; perpendicular is the FF block orientation |
| Vision blockers mixed in | Pathing grid doesn't know about bushes/fog — no false positives |
| Ramps absorb overlapping RawChokes | Raycasting sees terrain directly, ignores classification |
| No dynamic building detection | Pathing grid includes building footprints — detects enemy wall-offs |

### The Double-Edged Sword: Building Detection

The pathing grid includes building footprints. This is both a feature and a risk:

- **Good:** Detects enemy wall-offs, bunker gaps, cannon-rush gaps — `map_analyzer` is blind to these
- **Bad:** Our own buildings create false chokes. A pylon wall at our natural shows up as a choke every time our army paths near it

**Solution:** Use two grids:
- `map_data.get_pyastar_grid()` — terrain-only, for static choke detection (no building false positives)
- Live `bot.mediator.get_ground_grid` — for dynamic building/wall-off detection (includes buildings)

### Does It Have to Be Every Frame?

**No.** Three tiers of frequency:

| Data | Frequency | Why |
|---|---|---|
| Static terrain chokes | **Once** (`on_start`) | Cliffs, ramps, map geometry never change |
| Dynamic building chokes | **Every ~2 seconds** (44 frames) | Structures built/destroyed over seconds, not frames |
| Enemy position near choke | **Every frame** (cheap dict lookup) | Enemy moves fast, but choke location doesn't |

---

## Available Cython Functions

### Existing Functions for Batched Raycasting (No Compilation Needed)

| Function | Module | What It Does | Raycasting Relevance |
|---|---|---|---|
| `cy_in_pathing_grid_ma` | `general_utils` | O(1) grid lookup: `weight >= 1.0 and weight != INFINITY` | Core building block — each ray step is one call |
| `cy_point_below_value` | `numpy_helper` | O(1) grid value threshold check | Check "walkable AND not dangerous" |
| `cy_all_points_below_max_value` | `numpy_helper` | Batch check: are ALL points in a list below threshold? | **Key** — pass pre-computed ray points, check in one C call |
| `cy_all_points_have_value` | `numpy_helper` | Batch check: do ALL points have a specific value? | Same batch approach |
| `cy_points_with_value` | `numpy_helper` | Filter a list of points to those matching a value | Filter ray points to "walkable only" |
| `cy_last_index_with_value` | `numpy_helper` | Walk a list of points, return last index where grid matches value | Find where walkability ends |
| `cy_translate_point_along_line` | `geometry` | Move a point along a line defined by slope | Generate ray points (one at a time) |
| `cy_distance_to` / `cy_distance_to_squared` | `geometry` | Distance math | Width measurement |
| `cy_towards` | `geometry` | Move from point A toward point B by distance N | Ray stepping |
| `cy_flood_fill_grid` | `map_analysis` | Flood fill from a point on terrain+pathing grids | Detect enclosed regions / choke connectivity |
| `cy_dijkstra` | `dijkstra` | Multi-target Dijkstra on a cost grid | Find actual path bottlenecks |

### The Key Discovery: Batched Calls

Instead of calling `cy_in_pathing_grid_ma` 30 times per ray (30 Python→C round trips), pre-compute ray points in Python and pass the entire list to `cy_all_points_below_max_value` in a **single C call**. The C function iterates internally with `@boundscheck(False) @wraparound(False)` — no Python overhead per point.

| Approach | Per Ray | Per Squad/Frame | At 22 FPS |
|---|---|---|---|
| Naive (1 call per tile) | ~60 calls | ~300 calls | ~6,600 calls/sec |
| **Batched (`cy_all_points_below_max_value`)** | **~3 calls** | **~15 calls** | **~330 calls/sec** |
| Hybrid (only when `is_choke_between` hits) | ~3 calls | ~3 calls (rare) | **~66 calls/sec** |

### The Missing Function: `cy_ray_march_pathable`

The batched approach works but has a limitation: `cy_all_points_below_max_value` returns a boolean (all walkable or not), not the *index* where walkability stops. Finding the exact boundary requires binary search (log₂(10) ≈ 4 calls per ray direction).

A dedicated `cy_ray_march_pathable` function would do it in **1 call per ray direction** — march from a start point along a direction, return the first unpathable step index:

```cython
# ~25 lines of Cython, follows existing cy_in_pathing_grid_ma pattern
@boundscheck(False)
@wraparound(False)
cpdef int cy_ray_march_pathable(
    cnp.ndarray[cnp.npy_float32, ndim=2] grid,
    (double, double) start_pos,
    (double, double) direction,  # normalized (dx, dy)
    int max_steps,
):
    """March from start_pos along direction, return first unpathable step index.

    Returns:
        0 if start_pos itself is unpathable.
        Index (1-based) of first unpathable tile.
        max_steps if entire ray is pathable.
    """
    cdef:
        int i
        unsigned int ix, iy
        double weight

    for i in range(1, max_steps + 1):
        ix = <unsigned int>(start_pos[0] + direction[0] * i)
        iy = <unsigned int>(start_pos[1] + direction[1] * i)
        if ix >= grid.shape[0] or iy >= grid.shape[1]:
            return i
        weight = grid[ix, iy]
        if weight < 1.0 or weight == INFINITY:
            return i
    return max_steps
```

---

## Phased Implementation

### Phase 1: Hybrid Choke-FF System (Existing Cython — Zero New Dependencies)

**Goal:** Connect choke detection to FF placement using existing batched Cython functions. No compilation, no fork, works today.

**Cost:** ~3-15 C calls per choke detection (only when `is_choke_between` already detected a choke). Negligible per-frame cost.

#### 1a. Refine Choke Position + Orientation

When `is_choke_between()` returns a hit (cheap dict lookup), refine the approximate choke tile into an exact narrowest point + passage orientation using batched raycasting.

```
is_choke_between() dict lookup (existing, ~free)
  → if hit: raycast perpendicular rays at the choke tile
    → pre-compute ray point lists in Python (cheap)
    → batch-check with cy_all_points_below_max_value (1 C call per direction)
    → binary search for exact boundary (4 calls per direction)
    → result: exact narrowest point + passage orientation
  → cache per choke_id with timestamp
```

**Files to modify:**
- `bot/utilities/choke_grid.py` — add `refine_choke_with_raycast()` function
- `bot/constants.py` — add raycast-related constants (max ray steps, binary search depth, cache TTL)

**New constants:**
```python
# ===== RAYCAST CHOKE REFINEMENT =====
RAYCAST_MAX_WIDTH = 15.0       # Max tiles to march per perpendicular ray
RAYCAST_STEP_SIZE = 1.0        # Tile step size for ray marching
RAYCAST_CACHE_TTL_FRAMES = 44  # Re-refine static chokes every ~2 seconds
RAYCAST_BINARY_SEARCH_DEPTH = 4 # log2(15) ≈ 4 calls per ray via binary search
```

#### 1b. Add `compute_ff_choke_block` to Force Field System

New function in `bot/combat/force_field.py` — places FFs at detected chokes using the refined position + orientation from 1a.

**Priority chain in `combat.py`:**
```
1. Ramp block (existing)     — single FF at ramp center
2. Choke block (NEW)          — FF chain at refined choke point, perpendicular to passage
3. Army split (existing)      — FF chain through enemy center, perpendicular to engagement
```

**Algorithm:**
1. `is_choke_between()` detects a choke between squad and enemy center
2. `refine_choke_with_raycast()` finds exact narrowest point + passage direction
3. Check if enemy center is crossing through the choke (within `FF_CHOKE_BLOCK_RADIUS`)
4. Place FF chain perpendicular to the passage direction, centered on the narrowest point
5. Same energy pooling + greedy assignment as existing `compute_ff_split`

**Files to modify:**
- `bot/combat/force_field.py` — add `compute_ff_choke_block()` function
- `bot/combat/combat.py` — add choke block to the FF priority chain (between ramp block and army split)
- `bot/constants.py` — add `FF_CHOKE_BLOCK_RADIUS`, `FF_CHOKE_BLOCK_MIN_VALUE`, `FF_CHOKE_MIN_WIDTH`

**New constants:**
```python
# ===== FORCE FIELD CHOKE BLOCK =====
FF_CHOKE_BLOCK_RADIUS = 5.0    # Max distance from enemy center to choke point to trigger block
FF_CHOKE_BLOCK_MIN_VALUE = 6.0 # Minimum enemy army_value to justify choke block FF
FF_CHOKE_MIN_WIDTH = 8.0       # Only block chokes narrower than this (wider chokes need too many FFs)
FF_CHOKE_MAX_WIDTH = 3.0        # Chokes wider than this need multiple FFs; narrower = single FF
```

#### 1c. Dynamic Building Choke Detection

Use the live `bot.mediator.get_ground_grid` (includes building footprints) to detect enemy wall-offs and cannon-rush gaps.

**Frequency:** Every ~2 seconds (44 frames), cached per squad_id with timestamp.

**Filter:** Only detect building chokes when attacking (`bot._commenced_attack`), not when defending (our own buildings would create false positives).

**Files to modify:**
- `bot/utilities/choke_grid.py` — add `detect_dynamic_choke()` function using live grid
- `bot/combat/combat.py` — call dynamic choke detection in the squad loop, feed to FF priority chain

#### 1d. Integration with Existing Choke Engagement Policy

The existing choke engagement policy (hold/push based on who funnels) should be aware of FF availability. If we have sentries with energy, holding at a choke becomes more attractive because we can FF it.

**Files to modify:**
- `bot/combat/combat.py` — in the choke policy section, check sentry energy before deciding hold/push

### Phase 2: Upstream Cython Contribution

**Goal:** Write `cy_ray_march_pathable` and contribute it to `AresSC2/cython-extensions-sc2`. Eliminates the binary search overhead and makes raycasting 4× cheaper.

**Why contribute upstream:** Once merged, the `cython-extensions-sc2` team handles compilation for all platforms (Windows, Linux, macOS), pre-built release zips for ladder/competition, and Python version compatibility. **Zero maintenance burden on us.**

#### 2a. Write the Function

**File:** New `raycast.pyx` in the `cython-extensions-sc2` repo (or add to `numpy_helper.pyx`)

```cython
# ~25 lines, follows existing cy_in_pathing_grid_ma pattern
@boundscheck(False)
@wraparound(False)
cpdef int cy_ray_march_pathable(
    cnp.ndarray[cnp.npy_float32, ndim=2] grid,
    (double, double) start_pos,
    (double, double) direction,
    int max_steps,
):
    # ... (see code above)
```

#### 2b. Add Wrapper + Validator (Boilerplate)

Follow the existing pattern in `cython_extensions/type_checking/`:
- `wrappers.py` — add safe wrapper
- `validators.py` — add validation function
- `__init__.py` — add to `__all__`

~30 lines of boilerplate, copy-paste from an existing function.

#### 2c. Submit PR

1. Fork `AresSC2/cython-extensions-sc2`
2. Add function + wrapper + validator
3. Add a notebook in `notebooks/` demonstrating usage (they have a notebook-based development workflow)
4. Submit PR

**Likelihood of merge:** High. The function:
- Fits their existing pattern perfectly (grid query function)
- Is genuinely useful for any SC2 bot doing spatial queries
- Is small, focused, well-documented
- Fills a real gap (no existing function does directional ray marching)
- The repo is actively maintained (v0.16.0 on GitHub)

#### 2d. Compilation Requirements

To compile locally for development/testing:

| Requirement | Status on Current Machine | Action Needed |
|---|---|---|
| `setuptools` | ✅ 80.9.0 | None |
| `numpy` | ✅ 2.3.5 | None |
| `Cython` (compiler) | ❌ Not installed | `pip install Cython` |
| C compiler (MSVC) | ❌ Not found | Install "Desktop development with C++" in VS Installer, or MinGW |

The `cython_extensions` package uses a bootstrap compilation system — all `.pyx` files compile into a single `bootstrap.cp311-win_amd64.pyd`. You can't drop in a new `.pyx` without recompiling the entire package.

Build command (from their README): `poetry build` after `poetry install --with dev,test,docs,semver`.

### Phase 3: Upgrade to Upstream Function

**Goal:** Replace binary search calls with the dedicated `cy_ray_march_pathable` once it's released upstream.

**When:** After Tom Kerr merges the PR and a new version is released.

**How:**
1. `pip install cython-extensions-sc2 --upgrade`
2. Replace `refine_choke_with_raycast()` binary search with `cy_ray_march_pathable` calls
3. Download new pre-built zip from releases for competition shipping

**Effort:** ~5-minute refactor. Swap ~13 C calls for ~3 C calls per choke detection.

---

## Wall-Off System Improvements

The same raycasting infrastructure that powers choke FFs also revitalizes the wall-off system. `natural_wall_manager.py` currently does in Python what should be done in C.

### What Raycasting Replaces in Wall-Offs

| Current Code | Lines | Problem | Raycasting Replacement |
|---|---|---|---|
| `calculate_choke_width()` | ~40 | Clusters choke points, takes bounding box — ±50% error | 2 C calls: march left + right from center |
| `find_choke_along_path()` | ~60 | PCA on walkable pixels in 25×25 window | March along path, measure width at each tile |
| `measure_width_at_slice()` | ~30 | Projects pixels onto normal, takes range — ±50% | Direct perpendicular ray march |
| `find_narrowest_slice()` | ~30 | 20 slices, each doing pixel projection | March along passage, track min width |
| `validate_wall_connectivity()` | ~40 | numpy dilation + flood fill | Ray through wall — should be blocked |
| `snap_to_buildable_tile()` | ~30 | Expanding square search over placement grid | Raycast to nearest buildable tile |
| `find_valid_building_position()` | ~30 | Expanding circle with `can_place_structure` per tile | Raycast along choke tangent for valid positions |

### Wall-Off Improvements (Phase 1 — Existing Cython)

Using batched `cy_all_points_below_max_value`:

1. **Choke width:** Pre-compute perpendicular ray points, batch-check, binary search for boundary
2. **Narrowest point:** March along passage, measure width at each tile (batched)
3. **Wall validation:** Cast rays through wall positions — if any ray passes through, the wall has a hole
4. **Building placement:** Raycast along choke tangent to find valid buildable positions

### Wall-Off Improvements (Phase 2 — With `cy_ray_march_pathable`)

With the dedicated function, all of the above become single C calls per ray. The wall-off system shrinks from ~1100 lines to ~400 lines, with exact measurements instead of ±50% approximations.

---

## Architecture Summary

```
on_start (once):
  ├─ map_analyzer chokes → narrow_choke_points dict (existing)
  └─ [NEW] Raycast each map_analyzer choke → refine exact narrowest
     point + passage orientation → cache as refined_choke_points

every ~2 seconds (44 frames):
  └─ [NEW] Raycast squad→enemy line for dynamic building chokes
     → only if squad is attacking (not defending)
     → cache result per squad_id with timestamp

every frame (cheap):
  ├─ is_choke_between() dict lookup (existing, ~free)
  ├─ If choke detected: use cached refined position + orientation
  │  → feed to FF placement with correct perpendicular axis
  └─ Choke engagement policy (existing) — now FF-aware
```

### FF Priority Chain (Final)

```
1. Ramp block (existing)      — single FF at ramp center
2. Choke block (NEW, Phase 1) — FF chain at refined choke, perpendicular to passage
3. Army split (existing)      — FF chain through enemy center, perpendicular to engagement
```

---

## Impact Map

### Files Modified

| File | Phase | Changes |
|---|---|---|
| `bot/constants.py` | 1a, 1b | Raycast constants, FF choke block constants |
| `bot/utilities/choke_grid.py` | 1a, 1c | `refine_choke_with_raycast()`, `detect_dynamic_choke()` |
| `bot/combat/force_field.py` | 1b | `compute_ff_choke_block()` |
| `bot/combat/combat.py` | 1b, 1c, 1d | FF priority chain, dynamic choke detection, FF-aware engagement policy |
| `bot/utilities/natural_wall_manager.py` | 1 (wall-offs) | Replace PCA/bounding-box with batched raycasting |

### External Callers/Interfaces Affected

- `bot/combat/__init__.py` — export `compute_ff_choke_block`
- `bot/combat/unit_micro.py` — no changes (FF execution already handles multi-position assignments)

### Risk of Regression

**Low.** All changes are additive:
- Existing ramp block and army split logic unchanged
- Choke block is a new priority tier between existing tiers
- Choke engagement policy gets an additional input (sentry energy) but existing logic preserved
- Wall-off changes are in a standalone utility that runs at game start, not in the game loop

### Performance Impact

| Operation | Current Cost | Phase 1 Cost | Phase 2 Cost |
|---|---|---|---|
| Choke detection (per squad per frame) | 8 dict lookups (~free) | +13 C calls when choke detected (rare) | +3 C calls when choke detected |
| Dynamic building choke (per squad) | N/A | ~13 C calls every 44 frames | ~3 C calls every 44 frames |
| Wall-off generation (on_start) | ~1100 lines Python, PCA, flood-fill | ~400 lines, batched C calls | ~200 lines, single C calls |

All well within frame-time budget. `cy_find_units_center_mass` alone does O(n²) distance checks every frame — the raycasting cost is negligible by comparison.

---

## Shadow Review

1. **Biggest assumption:** That `map_analyzer` chokes are accurate enough for broad-phase detection, with raycasting only needed for narrow-phase refinement. If `map_analyzer` misses chokes entirely (e.g., on unusual map geometries), the hybrid approach won't find them.

2. **Most likely failure/edge case:** Dynamic building choke detection on the live grid. Our own buildings, ally buildings, and destructible rocks all appear in the pathing grid. The "only when attacking" filter helps, but neutral destructible rocks could create false chokes on some maps.

3. **Smallest change to improve robustness (≤10 LOC):** Add a `is_neutral_structure` filter to `detect_dynamic_choke()` — skip tiles blocked by neutral destructible rocks when raycasting the live grid. One `if` check per ray step, or a pre-filtered grid copy.