# AdvancedMicro Plan

---

# What Micro Actually Means

> 🧠 **Micro = maximising the value of your units.** Every micro move boils down to two things: keep your units alive longer, and get more damage out of them.

## The Two Objectives

1. **Maximise survivability** — units that live longer deal more total damage
2. **Maximise utility** — usually means damage output, but includes any value a unit provides

## The Tension

Survivability and utility often conflict in the moment. Bio under Psi-Storm: stand and shoot (utility) or move out (survivability)? The answer is always situational. Favor survivability for high-value units that can escape. Favor utility when escape is unrealistic.

## The Dual Mandate

You're always playing both sides: **maximise YOUR units' value** while **minimising your OPPONENT's units' value**. Force Fields cut armies in half. Terrain chokes reduce enemy surface area. Every micro decision serves one or both of these.

## The Fundamentals

- **Concave > Convex** — more units in attack range = more simultaneous damage. Form a line, A-move, the concave forms itself. Surrounds are just big concaves, it is always better to surround your enemy than attacking from just one direction in an engagement if the situation permits. Ramp control forces the enemy into a convex. you don't have to literally form a concave with the units, because all you have to do is to form a line with the units. You can A-move forward once you have formed a line, and the units will form a concave by themselves. simply spread out into small groups to form a line perpendicular to the attack direction
- **Convex has its uses** — against melee, a tight convex reduces the surface area they can touch. Terrain + ranged units in a ball can exploit this.
- **Kiting (backward) & Stutter Stepping (forward)** — use the attack cooldown window to reposition. Kiting keeps melee units at distance. Stutter stepping forward prevents targets from escaping.
- **Target Fire** — the smaller the fight, the more it matters. 5v4 is way bigger than 50v49. In small fights: focus fire one unit at a time. In big fights: select small groups to snipe high-value targets without breaking your concave.
- **Unit Positioning** — tanks in front, damage dealers behind. Some comps handle this naturally via range differences. Others (Marine/Marauder) need manual adjustment.
- **Micro Priority** — not all units benefit equally from micro. Focus your APM on the units that gain the most from active control. Let A-move units do their job.

## The Foundation

> 🎯 A concave is the foundation. Target fire, splits, kiting — none of it works without a decent concave first.

---

# Behaviors

There behaviors and decisions will go through a layer system

- **Army Layer:** This layer reads the overall fight state and makes decisions that affect the whole army.
- Squad Layer: regional clusters of units
- **Unit Layer:** Each units decision

## Prioritization - What to Target

solution in mind
- cost matrix (look to disruptors to see if the calculus)

1. For each unit, score each possible action by its expected delta to total army survivability + total army damage output
2. Pick the action with the highest combined score
3. Weight survivability vs utility based on fight state (winning = favor utility, losing = favor survivability)

Results:
- [x] Focus fire highest DPS threat (kill Colossus before Zealot)
- [x] Focus fire splash damage units (Disruptor, Colossus, High Templar)
- [x] Snipe lowest HP unit to remove DPS from field fastest
- [x] Target enabler units (Warp Prism keeping reinforcements flowing, Observer providing detection)
- [x] Priority shift mid-fight as key threats die
- [x] Ignore high armor/HP tanks when squishier damage dealers are available
- [x] Target the unit your composition has bonus damage against (Stalkers focus Armored)

## Positioning - Where to Be

### Concave Formation System

> The core idea: form a **line perpendicular to the enemy direction** before engaging, then A-move in. The game engine naturally forms the concave. This is a **one-shot pre-engagement setup** — once units are in position and fighting, normal micro takes over.

#### When to trigger
- About to engage a large bulk of the enemy army (not small skirmishes)

> **Note:** Chokes and ramps are a separate consideration. If we have a choke/ramp-aware
> army movement system (we already have `is_near_choke_or_ramp` in combat.py), it would
> naturally bypass or adapt the formation logic upstream. The formation system itself
> doesn't need to know about terrain — it just won't be invoked in those situations.

#### Phase A: Intra-Squad Line Spread (start here)
Each squad independently fans its own units into a line before engagement.
All calculations and move commands happen **within each squad individually** — no cross-squad coordination needed.

**Approach: Three-Group Fan-Out** (inspired by printf's concave tutorial)
Rather than assigning individual positions to every unit, split each squad into 3 sub-groups
and issue only 3 move commands — much simpler and avoids per-unit oscillation.

**Applies to: ground ranged units only** (Stalkers, Immortals, Colossus, Archons, etc.)
Melee units (Zealots) are excluded — they continue straight toward the enemy. Fanning them
out laterally would hurt them since they need to close distance and make contact ASAP.

**How it works:**
1. Detect engagement is imminent (enemy army within trigger range, bulk threshold met)
2. For each squad: separate melee vs ranged ground units
3. Compute the approach vector (squad center → enemy center)
4. Compute the perpendicular axis (left/right relative to approach)
5. Sort **ranged units only** by their lateral offset on that perpendicular axis
6. Split ranged units into 3 sub-groups: left flank, center, right flank
7. Issue move commands:
   - Left flank → point offset LEFT of approach line (fan out)
   - Right flank → point offset RIGHT of approach line (fan out)
   - Center → continues straight toward enemy
   - Melee units → continue straight toward enemy (no fan-out)
8. Once spread is achieved (ranged units roughly on the line), A-move entire squad toward enemy
9. Concave forms naturally from the spread. Transition to normal unit micro on contact.

**Why three groups instead of per-unit positions:**
- Only 3 move commands per squad (cheap, no oscillation)
- Mirrors how human players actually form concaves (box-select flanks, fan out)
- Sub-groups stay cohesive — units within each third move together
- Much simpler state management than tracking N individual formation slots

**Key design decisions:**
- Squad-level operation — uses existing ARES squad system
- Fan-out width scales with squad size (more units = wider spread)
- Melee units (Zealots) excluded from fan-out — they A-move straight at the enemy
- No stutter-step maintenance — value is in the initial shape at contact
- State machine per squad: APPROACHING → SPREADING → ENGAGING → FIGHTING (normal micro)
- Can fan out while still advancing (no need to retreat first for a bot)

**Challenges:**
- Trigger distance tuning (too early = slow approach, too late = no spread time)
- Terrain-aware fan-out (validate pathability of left/right fan positions)
- Squad size threshold (don't bother for small squads, maybe < 6 units)
- Transition timing: when is "spread enough" to commit the A-move?

**Implementation checklist:**
- [ ] Detect "bulk engagement imminent" condition (distance + army size thresholds)
- [ ] Compute approach vector and perpendicular axis per squad
- [ ] Split squad into 3 sub-groups by lateral position
- [ ] Compute fan-out target points for left/right flanks
- [ ] Validate fan-out positions are pathable (grid check)
- [ ] Issue 3 group move commands (can use ARES give_same_action per sub-group)
- [ ] Detect "spread achieved" and transition to A-move
- [ ] Transition to normal unit micro once in weapon range / contact
- [ ] Skip formation near chokes/ramps (reuse existing detection)
- [ ] Squad size minimum threshold

#### Phase B: Cross-Squad Coordination (after Phase A works)
Squads reposition relative to each other to form one unified wide front.

**How it works:**
1. Main squad holds or advances slowly
2. Flanking squads spread out laterally to widen the overall front
3. All squads engage together once the wide front is formed

**Key design decisions:**
- Builds on Phase A (each squad still does its own internal spread)
- Adds inter-squad positioning: squads are assigned lateral offsets from center
- Main squad anchors the center, other squads fill left/right flanks
- Requires timing coordination so squads engage together

**Challenges:**
- Cross-squad timing (don't let one squad engage alone while others are still positioning)
- Terrain may not allow the desired wide front
- Deciding when the front is "good enough" to commit

**Implementation checklist:**
- [ ] Assign squads lateral positions relative to enemy direction
- [ ] Main squad anchors center, others spread to flanks
- [ ] Engagement timing gate (all squads ready before any commits)
- [ ] Terrain validation for squad-level positions

---

### Choke Policy ✅ Phase 1 Complete

> The core insight: **chokes reduce enemy engagement surface area.** The value of fighting at a choke depends on how much enemy DPS the choke denies. Ramps are a subset of chokes — chokes where you also lack vision up.

#### Design Principles

- **Don't push through chokes. Make the enemy come through them.** Position on your side of the choke, let the enemy funnel through into your concave.
- **The lure behavior emerges naturally from the combat sim.** If the choke modifier makes `can_engage` harder to satisfy, the squad approaches, gets told "no," pulls back. Enemy AI chases through the choke. Now enemy is on your side — no choke between you — normal engagement.
- **Squad-level decision**, not army-level. Each squad independently evaluates whether a choke is between it and its enemy group.
- **Ramps = chokes + vision penalty.** `is_blind_ramp_attack` is a special case of the choke system with an extra "can't see" factor. Once choke policy works, ramp logic can be subsumed into it.

#### Phase 1 — Implemented ✅

**Choke map (`create_narrow_choke_points`):**
- Builds a `dict[Point2, float]` (tile → width) once at game start from `map_analyzer` chokes
- Pre-filters chokes wider than `CHOKE_MAX_WIDTH = 10.0` tiles — wide open areas don't bottleneck armies
- Width measured via `md_pl_choke.min_length` (RawChoke), side_a ↔ side_b distance (MDRamp), or `Polygon.width` fallback

**Choke detection (`is_choke_between`):**
- Samples `CHOKE_SAMPLE_POINTS` points along the squad→enemy line, O(1) dict lookup per point
- Returns `(width, choke_tile)` — the matched tile position is used to anchor the retreat
- Returns `(0.0, None)` if no choke on the line

**Army width (`effective_army_width`):**
- Uses `2 × mean distance from center mass` — mean is stable against outlier kiting units
- Small squads (≤2 units) return minimum `CHOKE_MIN_ARMY_WIDTH = 2.0`

**Funnel decision logic (per squad, per frame):**
- `choke_enemies` = ground, non-flying, non-worker enemies only (structures, air, workers, overlords filtered)
- `we_funnel = our_width > choke_width`, `they_funnel = enemy_width > choke_width`
- `PASS:decisive_win` — if combat sim returns `VICTORY_DECISIVE_OR_BETTER`, push through regardless of geometry
- `HOLD:we_funnel` — we compress, they don't → hold
- `HOLD:lure` — they compress, we don't → lure them through
- `HOLD:melee_ratio` — both funnel → hold only if `enemy_melee_dps / total_dps >= CHOKE_MELEE_DPS_THRESHOLD`
- `PASS:low_melee` / `PASS:both_fit` — choke doesn't matter, engage normally

**Group retreat (`PathGroupToTarget`):**
- Issues single MOVE command to entire squad via ARES group behavior
- Retreat target anchored to `choke_tile` (not squad position) — prevents retreating past multiple chokes
- `retreat_target = choke_tile + retreat_dir * CHOKE_RETREAT_DIST (7.0 tiles)`
- Individual unit micro gated during group retreat (`choke_active` flag)

**Current constants:**
- `CHOKE_MAX_WIDTH = 10.0` tiles
- `CHOKE_MIN_ARMY_WIDTH = 2.0` tiles
- `CHOKE_RETREAT_DIST = 7.0` tiles, `success_at_distance = 3.0`
- `CHOKE_MELEE_DPS_THRESHOLD = 0.5`

**Checklist:**
- [x] `create_narrow_choke_points()` — tile → width dict at game start
- [x] `is_choke_between()` — returns `(width, choke_tile)` tuple
- [x] `effective_army_width()` — mean-based diameter, outlier-stable
- [x] Enemy filtering — ground combat only (`choke_enemies`)
- [x] Funnel decision logic with 4 cases + decisive victory override
- [x] `PASS:decisive_win` override when combat sim is decisive
- [x] Group retreat via `PathGroupToTarget` anchored to choke tile
- [x] Ranged and melee micro gated during group retreat
- [x] Debug overlay — all decisions rendered in-game
- [x] Test: retreat behavior stable, no multi-choke cascade, decisive wins push through

#### Phase 2 — Backlog

- [ ] **Per-squad choke cooldown** — after a HOLD→PASS transition, suppress re-evaluation for N frames to prevent oscillation near choke edges
- [ ] **Minimum enemy DPS gate** — skip HOLD if `sum(u.ground_dps for u in choke_enemies) < threshold`; prevents queens/workers from holding back a decisive army
- [ ] **Bounce fix (t-value directionality)** — `is_choke_between` currently walks from squad toward enemy; if squad has already crossed the choke tile, t=0 samples can match a choke behind the squad, causing spurious holds
- [ ] **Ranged firing position estimate** — count how many enemy ranged units can draw LoS through the choke width; currently only melee ratio is used
- [ ] **Unit value asymmetry modifier** — `(enemy_count / own_count) * (own_avg_value / enemy_avg_value)` for outnumbered-but-expensive situations
- [ ] **Cross-squad coordination at chokes** — with Phase B concave, coordinate multiple squads holding the same choke
- [ ] Ramps subsumed into choke policy (replace `is_blind_ramp_attack` with choke + vision penalty)

---

#### Other Positioning Goals

challenges (general):
- take the entire army's relative position advantage to the enemy's position into consideration while in battle
- only applying at a certain army/squad size threshold
- to keep ideal positions (top of ramp or choke with a concave) vs attacking into the enemy and being in a convex
- preventing oscillation

Results:
- [ ] Concave formation to maximize simultaneous fire from ranged units (Phase A + B above)
- [ ] Not attacking up ramps without vision and enemies at the top
- [x] Pre-spread against AoE (Storm, Disruptor Nova, Siege Tank splash)
- [ ] High ground advantage before engagement
- [ ] convex positioning to restrict enemy surface area
- [ ] Retreat damaged units behind healthy ones (shield cycling)
- [x] Regroup after Blink to avoid staggered arrivals (blink targets concave edge, keeping stalker in formation)
- [ ] Spacing between High Templar to prevent double-Storm or EMP hitting both
- [x] Pull back during ability cooldowns (Disruptor retreats after firing Nova)
- [ ] surround with Zealots to pin enemy army
- [ ] Use terrain to restrict enemy concave (fight in corridors when outnumbered)

## Tactics - What to Use

solution in mind
- stutter stepping foward for range units when there is an advantage as a group

Challenges:
- deciding between objectives of maximizing and minizing damage

results:

*Movement maneuvers (ranged units, primarily Stalkers):*
- [x] Stutter step backward (kiting): attack, move away during cooldown, attack again. Used against melee or shorter-range units closing distance
- [ ] Stutter step forward: attack, move toward during cooldown, attack again. Used to chase retreating units or keep them in range
- [x] Stand and fire (no movement between shots, maximum DPS when positioning is already optimal)
- [x] Splitting: spreading units apart reactively to minimize incoming AoE damage mid-fight

*Blink (Stalker):*
- [x] Blink injured Stalker to the side of the army to save it (shield regen) — concave-aware blink along formation line
- [x] Blink back from dodgeable ranged damage [Cyclone lock-on, Widow Mine, Fungal Growth] — detects buffs (LOCKON) and projectile units (WIDOWMINEWEAPON, FUNGALGROWTHMISSILE)
- [x] Blink sniping (Snipe-A: walk-in, blink-out) — see detailed plan below
- [ ] Blink sniping (Snipe-B: blink-in, walk-out) — see detailed plan below
- [ ] Blink chasing (finish retreating enemies) — see detailed plan below

#### Blink Snipe & Chase — Detailed Plan

**Overview:** Group of stalkers blinks together to burst down a high-value target in one volley,
then retreats. Alternatively, chases down retreating enemies after winning a fight.
Two distinct modes with different lifecycles and squad-handling patterns.

##### Retreat Detection (shared by both modes)

**How to tell if enemy is running away:**
1. **`cy_is_facing` (per-unit, instant):** `NOT cy_is_facing(enemy, squad_center, angle_error=0.7)`
   means enemy faces away from us (~40° cone). Cheap, one call per candidate. Available for all
   visible enemy units since `unit.facing` comes from raw SC2 proto.
2. **Position delta (per-group, across frames):** Track `_enemy_center_prev[squad_id]` each frame.
   If `dist(enemy_center, squad_center)` increases over ~5 consecutive frames, group is retreating.
   More robust than per-unit facing (ignores jitter from individual units turning).
3. **Combined signal (recommended):** Enemy considered retreating when:
   - Position delta shows enemy group pulling away for ≥5 frames, OR
   - ≥50% of nearby enemies fail `cy_is_facing(enemy, squad_center, angle_error=0.7)`
   Either signal alone triggers; both together = high confidence.

Note: `unit.orders` is NOT available for enemy units (hidden by SC2 API). Cannot check
order_target. `unit.facing` + position tracking are the only reliable signals.

---

##### MODE 1: Chase (finish retreating enemies)

**When:** Squad combat sim says TIE_OR_BETTER (`bot._squad_engagement_tracker[squad_id]["can_engage"]`)
AND enemy is retreating (retreat detection above) AND high-value target exists among the runners.

**Gate checklist (ALL must pass):**
- [ ] `squad.can_engage == True` (existing hysteresis from combat sim)
- [ ] Enemy retreating (combined signal above)
- [ ] Target value ≥ `CHASE_MIN_VALUE` (reuse `TACTICAL_BONUS + army_value * TYPE_VALUE_SCALE`)
- [ ] `tactical_grid[target.position] < CHASE_TACTICAL_MAX` (~210 = target not heavily supported)
- [ ] Enough stalkers with blink ready to kill target: `ready_count >= needed_to_kill`

**Stalker selection:** Pick minimum stalkers needed to one-shot the target.
Sort candidates by: highest shield first, then closest to target. Only stalkers with
`EFFECT_BLINK_STALKER in abilities` AND `shield_health_percentage > SNIPE_MIN_HEALTH`.

**Behavior:**
- `StutterGroupForward(group=chase_stalkers, ..., target=target_unit, enemies=nearby_enemies)`
- Use blink ONLY if target gaps the chasers (distance increases beyond stalker range + 2)
- Stay until target dead, then release stalkers back to normal micro
- If target stops retreating and engages, re-evaluate with combat sim — abort if losing

**Squad handling: Pattern A (temporary role swap)**
- On commit: `assign_role(tag, UnitRole.CONTROL_GROUP_ONE)` for each chase stalker
- Separate control loop for chase role (runs alongside control_main_army)
- On release (target dead / timeout / low HP / re-engaged losing): `assign_role(tag, UnitRole.ATTACKING)`
- One-frame lag for squad manager to reflect change — acceptable for multi-second chase lifecycle
- Precedent: BASE_DEFENDER role already uses this pattern

**Exit conditions (any triggers release):**
- Target dies
- Chase timeout (e.g., 8 seconds — one full blink cooldown cycle)
- Any chaser drops below `STALKER_BLINK_HEALTH_THRESHOLD` (0.4) — blink out via existing P2 logic
- Combat sim flips to LOSS — re-evaluate, likely abort

---

##### MODE 2: Snipe (volley-and-retreat)

**When:** High-value target exists that we can kill in one coordinated volley.
Does NOT require winning the fight overall — just requires the trade to be worth it
and an acceptable escape path.

**Two sub-modes based on Printf's blink micro methods:**

**Snipe-A: Walk-in, blink-out (preferred — safest)**
- Stalkers walk into weapon range of target, fire one volley, then GROUP BLINK to retreat
- Blink is SAVED for escape → guaranteed exit
- Use when: `target.ground_range <= stalker.ground_range + 1.5` (we can walk into range)
  AND walk-in path is safe (`ground_grid` samples along approach are below danger threshold)
- Targets: High Templar (6 range), Ghost (6), Sentry (5), Viper (0 auto-atk),
  Disruptor (0 auto-atk), Medivac (0), Warp Prism (0)

**Snipe-B: Blink-in, walk-out (riskier — for out-ranged targets)**
- Stalkers GROUP BLINK to a landing spot in weapon range of target, fire one volley, then WALK out
- Blink is SPENT on approach → retreat is on foot (blink on 7s cooldown)
- Use when: target out-ranges us OR walk-in path is too dangerous
- Targets: Colossus (9 range), Tempest (15), Siege Tank (13), Liberator AG (10),
  Carrier (8+interceptors), Battlecruiser (6 but tanky)
- REQUIRES stricter escape-path gate (tactical grid lane check + ground_grid sampling)

**Gate checklist (ALL must pass for both sub-modes):**
- [x] Target value ≥ `SNIPE_MIN_TARGET_VALUE` (~18.0) via `effective_value()` (TACTICAL_BONUS + army_value * 0.5)
- [x] ~~Target isolation via tactical_grid~~ → Replaced with combat sim gate + corridor safety check (simpler, more reliable)
- [x] Damage math: `needed = ceil((target.shield + target.health) / dmg_per_stalker) + buffer`
      Must have `ready_count >= needed`. Buffer: +1 for mobile targets, +0 for static.
- [x] All selected stalkers have `EFFECT_BLINK_STALKER in abilities` (blink ready)
- [x] All selected stalkers have `shield_health_percentage > SNIPE_MIN_HEALTH` (0.75)
- [x] Combat sim gate: `can_win_fight(snipe_stalkers, nearby_enemies)` must be TIE or better
- [ ] (Snipe-B only) Escape path: sample 3-4 tiles from landing toward squad_center on
      `tactical_grid` — all must be ≤ `TACTICAL_ESCAPE_MAX` (~260). If corridor is enemy-dominated,
      abort — we can't walk out.
- [x] (Snipe-A only) Walk-in corridor safety: sample points along approach on influence grid —
      all must be ≤ `SNIPE_CORRIDOR_MAX_VALUE` (~15). Blocks walking through full armies.

**Stalker selection:** Same as Chase — pick minimum needed, prefer highest-shield + closest.

**Behavior (Snipe-A actual implementation — no explicit states, weapon-cooldown driven):**

```
Each frame checks unit state and decides:
  not enough in range  → PathGroupToTarget (pure move, weapons recharge en route)
  enough in range      → AMoveGroup (volley on in-range stalkers only)
  all weapons cooling  → GroupUseAbility blink retreat (all stalkers incl. stragglers)
  target dead/lost     → GroupUseAbility blink retreat immediately
  stalkers all dead    → cleanup only
```

"Enough" = at least `min_to_kill` (raw needed without overkill buffer) stalkers in range.
No frame counting or explicit state machine — weapon cooldown is the transition signal.

**If target dies during approach:** Blink retreat → release stalkers.
**If target dies during volley:** Blink retreat → release stalkers.
**If stalker dies during any phase:** `_snipe_committed.pop(tag, None)` in `on_unit_destroyed`.

**Squad handling: Pattern B (skip-set with TTL)**
- Keep stalkers in ATTACKING role (too short-lived for role churn)
- `bot._snipe_committed: dict[int, int] = {}` — maps tag → expiry_frame
- Group commands issued BEFORE per-unit loop in `control_main_army`
- Per-unit loop skips committed tags: `if tag in _snipe_committed and game_loop < expiry: continue`
- Fan-out / formation / choke retreat also filter committed tags
- TTL expires naturally; cleaned up in `on_unit_destroyed`
- Global cap: `SNIPE_COMMIT_COOLDOWN` (~180 frames = 8s) per squad — prevents re-committing
  before previous snipe group's blink is off cooldown

---

##### Shared Constants (bot/constants.py)

```
SNIPE_MIN_HEALTH = 0.75               # don't dive if already hurt
SNIPE_MIN_TARGET_VALUE = 18.0         # from TACTICAL_BONUS + army_value * 0.5
SNIPE_ISOLATION_MIN = 12.0            # unit_value minus tactical excess
SNIPE_OVERKILL_BUFFER_MOBILE = 1      # +1 for moving targets (BC, Medivac)
SNIPE_OVERKILL_BUFFER_STATIC = 0      # +0 for static targets (HT, Tank)
SNIPE_EXIT_FRAMES = 30                # hold volley state (~1.3s, one weapon cycle)
SNIPE_COMMIT_COOLDOWN = 180           # frames between snipe commits per squad (~8s)
CHASE_MIN_VALUE = 10.0                # lower bar for chase (already winning)
CHASE_TACTICAL_MAX = 210              # target must not be heavily supported
CHASE_TIMEOUT_SECONDS = 8.0           # max chase duration
TACTICAL_ESCAPE_MAX = 260             # max tactical grid value along escape lane
RETREAT_DETECTION_FRAMES = 5          # consecutive frames of distance increase = retreating
```

##### Integration Points

- **Entry point:** `control_main_army` in `bot/combat/combat.py`, per-squad, BEFORE per-unit micro loop
- **New file:** `bot/combat/group_snipe.py` — plan evaluator + state machine
  (or inline in combat.py if small enough — decide during implementation)
- **Skip-set wiring:** filter `_snipe_committed` tags in per-unit ranged loop, fan-out,
  choke retreat, and any other squad-level command that might override group orders
- **Cleanup:** `on_unit_destroyed` pops tags from `_snipe_committed` and chase role
- **Debug:** render snipe/chase state, committed tags, gate pass/fail in debug overlay

##### Implementation Order

1. [x] **Damage math helper** — `stalkers_needed_to_kill`, `damage_per_volley`, `effective_value`
2. [x] **Snipe-A (walk-in, blink-out)** — `execute_snipe_a`, `try_commit_snipe`, skip-set pattern
3. [ ] **Snipe-B (blink-in, walk-out)** — adds tactical grid escape check
4. [ ] **Chase mode** — adds role-swap pattern + retreat detection
5. [ ] **Debug overlay** — gate visualization, committed stalker markers
6. [x] **Tuning (Snipe-A)** — tested, volley fires reliably, blink retreat confirmed working

##### Cython Extensions & API Toolkit

Functions from `cython_extensions` and python-sc2 mapped to each part of the plan.
Always prefer `cy_*` over python-sc2 equivalents on hot paths.

**Damage Math (`stalkers_needed_to_kill`)**
| Function | Source | Role |
|---|---|---|
| `unit.calculate_damage_vs_target(target)` | python-sc2 `Unit` | Exact damage/volley including upgrades, armor, bonus damage. Core of kill math. |
| `cy_range_vs_target(unit, target)` | `combat_utils` | Weapon range to target (air/ground aware). Use for Snipe-A vs Snipe-B sub-mode selection (compare stalker range vs target range). Replaces manual `ground_range`/`air_range` checks. |
| `cy_attack_ready(ai, unit, target)` | `combat_utils` | Checks weapon cooldown + turn rate. Use in VOLLEY state to confirm stalkers can fire before retreating. |

**Target Selection & Evaluation**
| Function | Source | Role |
|---|---|---|
| `cy_closest_to(position, units)` | `units_utils` | Find nearest high-value target to squad. |
| `cy_sorted_by_distance_to(units, position)` | `units_utils` | Sort stalker candidates by distance to target (pick closest N healthy ones). |
| `cy_closer_than(units, distance, position)` | `units_utils` | Find enemies within radius of landing spot → threat count at destination. Also find stalkers near target for candidate pool. |
| `cy_find_units_center_mass(units, distance)` | `units_utils` | Returns (center, count_within_distance). Use to check enemy density around target — low count = isolated target → snipe; high count = buried → skip. |
| `cy_in_attack_range(unit, enemies, bonus_distance)` | `units_utils` | Check which enemies a stalker can shoot. Use for APPROACH→VOLLEY transition (are we in range of the target?). |

**Geometry & Position Calculation**
| Function | Source | Role |
|---|---|---|
| `cy_towards(start, target, distance)` | `geometry` | Compute blink landing spot: `cy_towards(stalker_center, target.position, STALKER_BLINK_RANGE)`. Compute retreat point: `cy_towards(landing, squad_center, STALKER_BLINK_RANGE)`. Compute escape-path sample points along corridor. Returns tuple — wrap in `Point2()` only when needed. |
| `cy_distance_to(p1, p2)` | `geometry` | All distance checks: target distance, chase gap detection, retreat-distance tracking. 157ns. |
| `cy_distance_to_squared(p1, p2)` | `geometry` | Use for pure comparison (no sqrt needed): "is target closer than X?" → compare against X². ~30% faster. |
| `cy_angle_to(from_pos, to_pos)` | `geometry` | Compute angle from enemy to squad. Combined with `cy_angle_diff(enemy.facing, angle)` for custom retreat detection alternative. |
| `cy_angle_diff(a, b)` | `geometry` | Absolute angle difference. If `cy_angle_diff(enemy.facing, cy_angle_to(enemy.pos, squad_center)) > π/2` → enemy faces away. More precise than `cy_is_facing` for retreat detection if needed. |
| `cy_center(units)` | `units_utils` | Fast centroid of stalker group or enemy group. Use for group-level position tracking (retreat detection frame-to-frame delta). |

**Retreat Detection**
| Function | Source | Role |
|---|---|---|
| `cy_is_facing(unit, other, angle_error)` | `combat_utils` | Primary retreat signal: `not cy_is_facing(enemy, squad_center, 0.7)` = enemy faces away. 323ns per call. |
| `cy_angle_to` + `cy_angle_diff` | `geometry` | Alternative: `cy_angle_diff(enemy.facing, cy_angle_to(enemy.pos, squad_center)) > 1.57` → enemy faces away from us. More precise angular control than `cy_is_facing`. |
| `cy_center(enemies)` | `units_utils` | Track enemy group centroid per frame for position-delta retreat detection. |

**Path & Grid Safety**
| Function | Source | Role |
|---|---|---|
| `cy_all_points_below_max_value(grid, value, points)` | `numpy_helper` | **Escape-path check (Snipe-B):** sample 3-4 points from landing toward squad_center on tactical_grid, check ALL ≤ `TACTICAL_ESCAPE_MAX`. Single call replaces a per-point loop. **Walk-in check (Snipe-A):** sample points on ground_grid. |
| `cy_point_below_value(grid, position, limit)` | `numpy_helper` | Single-point safety: landing spot check on tactical_grid, target isolation check. |
| `cy_last_index_with_value(grid, value, points)` | `numpy_helper` | Walk-in pathability: sample points toward target on pathing_grid, find how far we can walk before hitting unpathable terrain. If last_index < required distance → walk-in blocked → switch to Snipe-B or abort. |
| `cy_in_pathing_grid_ma(grid, position)` | `general_utils` | Verify blink landing position is on walkable terrain. |

**Chase Mode**
| Function | Source | Role |
|---|---|---|
| `cy_further_than(units, distance, position)` | `units_utils` | Detect if target has gapped the chasers: if target in `cy_further_than([target], stalker_range + 2, squad_center)` → use blink to close. |
| `cy_in_attack_range(stalker, [target])` | `units_utils` | Check if chasers are still in range. If not, blink to close gap. |

**Not used (reviewed & excluded)**
| Function | Reason excluded |
|---|---|
| `cy_adjust_moving_formation` | Docs warn "don't use during combat". Only useful for pre-combat march — our snipe is a burst combat action. |
| `cy_find_aoe_position` | AoE targeting, not relevant for single-target snipe. |
| `cy_dijkstra` / `DijkstraPathing` | Full pathfinding — too heavy for a per-frame snipe evaluation. Grid sampling via `cy_all_points_below_max_value` is cheaper and sufficient. Could revisit for complex escape pathing later. |
| `cy_flood_fill_grid` | Map analysis tool, not combat-relevant. |
| `cy_pick_enemy_target` | Lowest-HP targeting. We already use `select_target()` from `target_scoring.py` with weighted scoring. |
| `cy_find_correct_line` | Line-finding geometry for walls, not combat. |
| `cy_translate_point_along_line` | Translates along a slope. `cy_towards` is more intuitive for our point-to-point calculations. |

##### Backlog (deferred)

- **Printf Method 3 (front-decoy bait):** Select frontmost stalker, blink it back when enemy
  fires volley, then all-in attack + group blink retreat. Requires detecting enemy volley commit
  (no engaged_target_tag for enemies). Hard without more API data. Park for later.
- **Cliff blink snipe:** Blink onto high ground for base entry. Requires vision gate
  (`bot.is_visible(landing)`) + observer coordination. Separate feature.

*Guardian Shield (Sentry):*
- [x] Activate before ranged damage lands (gated on `has_ranged_enemies` — only casts when enemy has ranged units)
- [x] Spread coverage across army / squads (avoid overlapping shields via `GUARDIAN_SHIELD_OVERLAP_DISTANCE`)

*Hallucination Scout (Sentry):*
- [x] Trigger from intel system when blind (`_intel_urgency > HUNT_URGENCY_THRESHOLD` and no 3rd observer)
- [x] Use `get_hunt_target()` for pathing (same as observer hunt mode)
- [x] Priority: only hallucinate if no 3rd observer available
- [x] Hallucinate Phoenix (fastest scout unit, flies over terrain)
- [x] Blind ramp trigger: cast Phoenix when army blocked at ramp with no army observer
- [x] High ground Phoenix holds position at ramp top for persistent vision
- [x] Early game base scout: replace worker scout with hallucinated Phoenix following standard probe waypoints
- [x] Debug labels: HALU:HIGH_GND, HALU:BASE_SCOUT, HALU:SCOUT

*Force Field (Sentry):*
- [x] Split enemy army to fight favorable portion
- [x] Block Ramp attack with force field

*Storm (High Templar):*
- [x] Cast on clumped ranged units (UseAOEAbility, min 4 targets, avoids friendly fire + stacking)

*Feedback (High Templar):*
- [x] Target energy-based threats (Ghosts, Vipers, HTs, Infestors, Ravens — FEEDBACK_TARGET_TYPES const)

*Purification Nova (Disruptor):*
- [x] Fire into clumps

*Archon merge:*
- [x] Merge depleted HT pairs when both < 50 energy (merge_high_templars, request_archon_morph)

*Recall (Nexus/Mothership):*
- [x] Mass recall army out of a losing fight

### Later

*Graviton Beam (Phoenix):*
- [ ] Lift key unit out of fight (Siege Tank, Queen, Immortal)

*Warp Prism Immortal micro:*
- [ ] Load Immortal into Prism between attacks, drop it to fire, pick it back up before it takes damage

*Workers*
- [ ] Move the workers in the base of a widow mine drop to another base and send a single probe to run into the widow mine to take the hit so the others escape

*Stalker*
- [ ] Blink onto the high ground to gain entry into a base