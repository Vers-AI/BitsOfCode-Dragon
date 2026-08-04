# Expansion Improvement Plan — Race-Specific Signals

**Created:** 2026-08-03
**Status:** Backlog (not yet implemented)
**Depends on:** `track_enemy_timings` running every frame (currently build-order-only)

## Background

The current `_has_map_control()` in `bot/managers/macro.py` uses only race-neutral
signals (threat system, commenced attack, intel freshness, passive opponent).
This plan covers the race-specific signals that need race-aware interpretation.

## Prerequisite: Move `track_enemy_timings` to Every Frame

**Problem:** `track_enemy_timings(bot)` is called only during the build order phase
(`bot/bot.py:391`). Once `build_order_runner.build_completed` is True, the following
signals stop updating:

- `bot._enemy_bases_count` — stale after build runner completes
- `bot._enemy_nat_started_at` — set once, never updated for 3rd/4th bases
- `bot._enemy_worker_count` / `_enemy_worker_count_history` — frozen
- `bot._enemy_army_supply` — frozen
- `bot._enemy_gas_workers_count` — frozen

**Fix:** Move the `track_enemy_timings(self)` call outside the `if not build_completed`
block in `bot/bot.py` so it runs every frame regardless of build order status.

**Cost:** +0 (moving one line). Perf: `track_enemy_timings` is O(n) over enemy units,
already runs during build order phase — extending to full game is safe.

## Race-Specific Signals

### 1. Race-Aware Expansion Mirroring

**Pro Principle:** "The opponent just invested in an expansion. I could either expand
myself to keep up, or attack now while the opponent's army is briefly stretched."
*(Satirist SC2 AI)*

**Why race-specific:** Zerg routinely operates on more bases than Protoss/Terran.
Matching a Zerg's base count as Protoss would over-expand. The "mirror" signal
needs race-aware thresholds.

**Proposed Logic:**
```python
# In _has_map_control(), after the hard gates:
enemy_bases = _count_visible_enemy_townhalls(bot)  # live count from enemy_structures
our_bases = len(bot.townhalls)

if bot.enemy_race == Race.Zerg:
    # Zerg expected to be +1 base on us. Don't mirror — they're ahead by design.
    # Expand if we're 2+ bases behind (they're droning hard, we need to keep up).
    if enemy_bases - our_bases >= 2:
        return True
elif bot.enemy_race == Race.Terran:
    # Terran typically matches base count. If they've expanded, it's safe to mirror.
    if enemy_bases >= our_bases:
        return True
elif bot.enemy_race == Race.Protoss:
    # PvP: mirror is critical. If they expand and we don't, we fall behind.
    if enemy_bases >= our_bases:
        return True
```

**Cost:** +1 (new helper + race branches). Needs `_count_visible_enemy_townhalls()`.
**Risk:** Low — only adds positive signals, doesn't block.

### 2. Race-Aware All-In Detection

**Pro Principle:** "Match their base count and play defensively, then expand only
after they throw away their army against your well-defended base." *(Reddit)*

**Why race-specific:** A Protoss on 1 base vs our 2 is likely all-in. A Zerg on 2
bases vs our 2 is standard play. The "all-in" signal needs race-aware thresholds.

**Proposed Logic:**
```python
# Negative signal: opponent is on fewer bases → possible all-in
if bot.enemy_race == Race.Protoss and enemy_bases < our_bases:
    # Protoss behind on bases = investing in army, not economy = all-in risk
    # Don't expand further; match their base count and defend
    return False  # Blocks expansion in the cautious gate section
elif bot.enemy_race == Race.Terran and enemy_bases < our_bases - 1:
    # Terran 2+ bases behind is unusual — likely all-in or heavy pressure
    return False
# Zerg on fewer bases is not necessarily all-in (they may be droning)
```

**Cost:** +1 (race branches in the cautious gate section).
**Risk:** Medium — could block expansion vs Zerg 1-base all-ins if not tuned.

### 3. Enemy Worker Economy Comparison

**Pro Principle:** Worker count is the most fundamental economic indicator. If the
opponent has significantly fewer workers, they're investing in army/tech — be cautious.

**Why race-specific:** Zerg drone counts scale differently (can inject-mass drones),
Protoss probe production is linear, Terran can MULE-burst minerals.

**Proposed Logic:**
```python
# After prerequisite (track_enemy_timings runs every frame)
enemy_workers = bot._enemy_worker_count
our_workers = bot.workers.amount

if bot.enemy_race == Race.Zerg:
    # Zerg can drone hard; if they have way more workers, they're droning = safe to expand
    # If they have fewer workers than us, they cut drones for army = all-in risk
    if enemy_workers < our_workers * 0.7:
        return False  # Zerg cut drones = army incoming
elif bot.enemy_race == Race.Protoss:
    # PvP: worker parity is expected. Significant gap means different builds.
    if enemy_workers < our_workers * 0.6:
        return False  # Probe cut = all-in
elif bot.enemy_race == Race.Terran:
    # Terran can MULE, so raw worker count underestimates their income
    if enemy_workers < our_workers * 0.5:
        return False  # Heavy SCV cut = all-in
```

**Cost:** +2 (race-aware thresholds, needs tuning).
**Risk:** Medium — worker counts are unreliable when we haven't scouted.

### 4. Enemy Army Supply Comparison

**Pro Principle:** "Never want to expand when you think your opponent is about to
attack." Army supply comparison is a rough proxy for attack readiness.

**Partially race-specific:** Supply is comparable across races (1 supply = 1 supply),
but the *implication* of a supply gap differs — Zerg can remax instantly, Protoss
can't. Use as a rough gate with race-aware urgency multipliers.

**Proposed Logic:**
```python
enemy_supply = bot._enemy_army_supply
our_supply = bot.supply_army

if enemy_supply > our_supply * 1.5:
    # Opponent has significantly more army — possible timing attack incoming
    if bot.enemy_race == Race.Zerg:
        # Zerg can remax instantly — more dangerous, block at lower threshold
        return False
    else:
        # Protoss/Terran remax slower — slightly more forgiving
        if enemy_supply > our_supply * 1.8:
            return False
```

**Cost:** +1 (race-aware thresholds).
**Risk:** Medium — army supply is unreliable from stale intel.

### 5. Enemy Gas Timing Interpretation

**Pro Principle:** Gas timing reveals strategy — early gas = tech/aggression, late
gas = macro/expand. This is deeply race-specific.

**Why race-specific:** 1-gas Zerg = standard (ling/bane), 2-gas Protoss = tech
(Stargate/DT), Terran gas timing depends on build (bio vs mech).

**Proposed Logic:** Defer to the existing cheese detection system (`detect_cheese`)
which already interprets gas timing in a race-aware way. Don't duplicate.

**Cost:** +0 (reuse existing).
**Risk:** N/A.

## Build-Specific Signals

### 6. Defender's Advantage — Natural Expansion During Attacks

**Pro Principle:** "Defensive fights almost always have an advantage." *(Liquipedia)*
"Expand further out only after they throw away their army against your
well-defended base." *(Reddit)* "Get your third base whenever you can defend it."
*(Scribd SC2 guide)*

**Why build-specific:** The defender's advantage only applies to expansions
**behind your defensive line** (the natural, walled with batteries). Expanding
the 3rd/4th base during an attack is still suicidal — those are beyond the wall
and have no defender's advantage. This signal depends on the bot having built
defensive infrastructure (wall + shield battery), which varies by build and
opponent race.

**Simple version (IMPLEMENTED 2026-08-03):**

`_can_expand_natural_under_attack(bot)` in `macro.py` allows the natural
expansion during an attack if:
- We're on 1 base (expanding to the natural, not beyond)
- We have a completed shield battery within 20 units of the natural

This bypasses the `_under_attack` hard gate in three locations:
1. `_has_map_control()` Gate 1
2. `banking_for_expansion` condition (via `blocked_by_attack`)
3. Non-banking expansion path (via `blocked_by_attack`)

The `ExpansionController`'s ground-grid `is_position_safe` check remains as the
final safety gate — if enemies are physically at the natural, the Nexus won't
be placed.

**Enhanced version (BACKLOG):**

The simple version only checks for a shield battery. The enhanced version
should also check for:
- Wall buildings (gateway/cybercore) near the natural choke, not just a battery
- Threat level from `assess_threat` — only allow if the threat is survivable
  (not an overwhelming force that will break the wall)
- Whether the battery has energy (a depleted battery provides no defense)
- Race-specific: Zerg ling flood vs Protoss stalker push have different
  wall requirements (full wall vs gatekeeper gap for zealots)

## Implementation Priority

1. **Prerequisite: Move `track_enemy_timings` to every frame** — enables all signals
2. **Race-aware expansion mirroring** (#1) — highest ROI, lowest risk
3. **Defender's advantage — enhanced** (#6) — simple version implemented; enhanced version (wall check, battery energy, threat level) backlog
4. **Race-aware all-in detection** (#2) — medium ROI, medium risk
5. **Enemy worker economy comparison** (#3) — medium ROI, needs tuning
6. **Enemy army supply comparison** (#4) — low ROI, high noise
7. **Gas timing** (#5) — already covered by cheese detection

## References

- Pro expansion principles: see `_has_map_control()` docstring in `macro.py`
- Satirist SC2 AI article on expansion mirroring: https://satirist.org/learn-game/systems/starcraft/
- Blizzard expansion guide: https://news.blizzard.com/en-us/starcraft2/4488311/game-guide-base-expansions
- Liquipedia "How to Practice": "When you get an advantage, expand."
- Reddit: "Match their base count and play defensively, then expand after they throw away their army."