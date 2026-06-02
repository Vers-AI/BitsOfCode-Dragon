# MindMe's Weighted Scoring System — Markdown Download

```markdown
# MindMe's Weighted Scoring System — Implementation Reference

> **Context:** This document captures a scoring/weighting system designed by a competitive StarCraft II bot developer (MindMe, Protoss, Go-based bot) for unit targeting and combat decisions. The system was explained during a recorded roundtable session (Postbots #19). This reference is written for an LLM coding assistant with no prior context of the conversation.

> **Goal:** Adapt this system for a Python SC2 bot (Protoss) built on the ARES framework. The bot uses **squads** (not fully decentralized control), so this scoring system should be applied **per-unit within a squad context** — each unit in a squad still needs to decide which enemy to target and whether to engage.

---

## 1. Overview: What the Scoring System Does

Every friendly combat unit evaluates every visible enemy unit and assigns it a **score**. The unit then acts on the highest-scored target. The score is a single number composed of multiple weighted factors layered together.

This replaces simple "attack closest" logic with context-aware targeting that produces emergent behaviors like **focus fire**, **target prioritization**, and **adaptive engagement**.

---

## 2. Score Composition

The score for a given enemy target is built in layers. Each layer adds or subtracts from the total score based on a weighted factor.

### Layer 1 — Distance (Foundation)

The most basic and essential factor. Calculate the distance from your unit to the potential target. Closer targets score higher.

```

score = -distance(my_unit, enemy_unit)

```

This alone gives you "attack closest" behavior. Everything else builds on top of this.

### Layer 2 — Target Health

Enemy units with lower health are worth more — they're closer to dying, so finishing them off removes damage from the field faster.

```

health_score = max_health - current_health

# Or normalized:

health_score = (1 - current_health / max_health) * HEALTH_WEIGHT

```

With just distance + health, you already get **focus fire** behavior: units naturally converge on the lowest-health enemy that's nearby.

### Layer 3 — Unit Type Value

Not all units are equal. A High Templar is worth more than a Zealot. A Siege Tank in siege mode is a higher priority than a Marine. Assign a base value weight to each enemy unit type.

```

type_score = UNIT_TYPE_WEIGHTS[enemy_unit.type_id]

```

This can also incorporate **counter-table logic**: if your unit is strong against this enemy type, increase the score (your unit should prefer targets it's effective against).

### Layer 4+ — Additional Situational Factors

Keep layering on factors as needed. Examples from MindMe's system:

- **Damage your unit can deal to this target** — prefer targets you're effective against
- **Threat the enemy poses to you** — high-DPS enemies that can hit you score higher
- **Positional modifiers** — high ground enemies get a penalty (they're harder to kill due to miss chance); enemies in choke points may get a bonus or penalty depending on your goal
- **Engagement state** — is this enemy already being attacked by your allies? (can be used to encourage or discourage focus fire stacking)

---

## 3. Two-Tier Score Structure

The scoring has two tiers:

### Base Score (Universal)

Applied the same way for all of your units. This represents the global understanding of "how valuable is this enemy target to kill."

Factors:
- Enemy unit type value
- Enemy health percentage (lower = higher score)
- Counter-table matchup bonuses
- Strategic value (is it a spell caster? a detector? a production building?)

### Per-Unit Score (Individual)

Specific to each of your units. This is what makes each unit's decision unique.

Factors:
- Distance from this specific unit to the target
- This unit's damage effectiveness against the target
- This unit's current health (low health units may prefer safer/closer targets)
- Whether this unit is already in range of the target
- Any role-specific modifiers (e.g., a unit assigned to base defense should heavily weight enemies near the base)

### Final Score

```

final_score = base_score(enemy) + per_unit_score(my_unit, enemy)

```

Each of your units calculates this for every visible enemy, then acts on the highest-scored target.

---

## 4. Weight Ranges and Tuning

### Scale

MindMe uses a weight range of roughly **-10,000 to +10,000**. Despite this wide range, he noted that **even 1-2 points can change targeting behavior**. The system is sensitive at the margins.

### Negative Weights Are Common

Many factors use negative weights. For example:
- Distance is inherently negative (farther = worse)
- Penalties for attacking targets on high ground
- Penalties for attacking targets near detection (for cloaked units)
- Penalties for low-priority targets

### Tuning Process

1. **Watch replays** of the bot playing
2. **Identify bad targeting** — two enemies standing next to each other, bot shoots the wrong one
3. **Adjust the weight** of the factor that should have differentiated them
4. **Re-test** and repeat

This is an iterative, manual process. MindMe emphasized the **80/20 rule**: get 80% of the value with simple weights, don't over-engineer. Start simple and refine over time.

### Unit Wrapper

Each unit gets its own **wrapper object** with approximately **100 variables** tracking state. One key variable mentioned: `is_engaged` (boolean — is there an enemy within a certain radius, ~8-10 game units).

---

## 5. Combat Sim Integration

The scoring system works alongside a **combat simulation** that operates at two levels:

### Global Combat Sim ("Should we fight?")

Compares the **total known enemy army** vs the **total own army**. If the bot would win the fight on a flat, neutral battleground with no positioning advantages, the army is given the green light to attack.

This is the macro-level decision: attack, defend, or contain.

When the global sim says "attack" but the army reaches a defended position (bunkers, siege tanks, etc.), the **local** assessment overrides — units won't dive in. The emergent result is **containing behavior**: the army sits outside the enemy base, applying pressure without suiciding.

### Local Combat Assessment ("Should THIS unit engage?")

Per-unit or per-area check. Before a unit commits to attacking its highest-scored target, it evaluates whether it can survive the engagement. If it can't, it waits. As more friendly units arrive, eventually the local assessment flips to "winnable" and units engage.

This naturally prevents **trickling** — units don't run in one by one.

---

## 6. Engagement Threshold

When moving across the map (not yet in combat), units use **grouping behavior** to stay together.

Once approximately **30% of the group is engaged** (enemy units within ~8-10 radius), the grouping behavior switches off and individual unit scoring/micro takes over.

"Engaged" is defined as: an enemy unit is within a certain radius (roughly 8-10 game distance units) of the friendly unit.

---

## 7. Army Movement (Pre-Combat Grouping)

When moving to an engagement and no fighting is happening, units group up using **two concentric radiuses**:

- **Inner radius** — the desired cluster zone. Units inside this radius are considered "grouped."
- **Outer radius** — units in this zone are told to move toward the inner radius, but the inner group does **not** wait for them.

This prevents the whole army from stalling because one unit just warped in at the base. Fast units (Stalkers) will step outside the inner radius, pause briefly, and let slower units catch up — then resume moving.

Once the 30% engagement threshold is hit, grouping stops and the scoring system fully takes over.

---

## 8. Counter Tables

The scoring system feeds into (and is fed by) **counter tables** — a mapping of enemy unit compositions to preferred own-unit compositions. The counter tables adapt to the enemy automatically:

- The bot observes the enemy's current unit composition
- The counter table determines which of your unit types are most effective
- These matchup bonuses feed into the **base score** for target selection
- The same data informs **production decisions** (what to build next)

This means the bot's targeting AND production are always adapting to the enemy.

---

## 9. Implementation Skeleton (Pseudocode)

```

def score_target(my_unit, enemy_unit, game_state):

score = 0.0

# --- Base Score (same for all own units) ---

# Unit type value

score += UNIT_TYPE_VALUE.get(enemy_unit.type_id, 0)

# Health: lower health = higher priority

health_ratio = enemy_[unit.health](http://unit.health) / enemy_[unit.health](http://unit.health)_max

score += (1 - health_ratio) * HEALTH_WEIGHT

# Counter table bonus

score += COUNTER_TABLE.get(

(my_unit.type_id, enemy_unit.type_id), 0

)

# --- Per-Unit Score (specific to this unit) ---

# Distance: closer = better (negative weight)

dist = my_unit.position.distance_to(enemy_unit.position)

score -= dist * DISTANCE_WEIGHT

# Damage effectiveness

dps = calculate_dps(my_unit, enemy_unit)

score += dps * DPS_WEIGHT

# Already in range bonus

if dist <= my_unit.ground_range:

score += IN_RANGE_BONUS

# --- Situational Modifiers ---

# High ground penalty

if enemy_on_high_ground(enemy_unit, my_unit):

score -= HIGH_GROUND_PENALTY

# Threat level: high-DPS enemies near this unit

if enemy_can_attack(enemy_unit, my_unit):

enemy_dps = calculate_dps(enemy_unit, my_unit)

score += enemy_dps * THREAT_WEIGHT

return score

def select_target(my_unit, visible_enemies, game_state):

"""Each unit calls this to pick its target."""

best_target = None

best_score = float('-inf')

for enemy in visible_enemies:

s = score_target(my_unit, enemy, game_state)

if s > best_score:

best_score = s

best_target = enemy

return best_target

def should_engage(squad, known_enemies):

"""Global combat sim — should this squad fight?"""

own_strength = sum(unit_strength(u) for u in squad.units)

enemy_strength = sum(unit_strength(e) for e in known_enemies)

return own_strength > enemy_strength  # simplified

def is_locally_safe(my_unit, nearby_enemies, nearby_allies):

"""Local combat check — can this unit survive engaging?"""

local_own = sum(unit_strength(u) for u in nearby_allies)

local_enemy = sum(unit_strength(e) for e in nearby_enemies)

return local_own > local_enemy  # simplified

```

---

## 10. Key Principles for Implementation

1. **Start with distance only.** Get "attack closest" working, then layer on health, then unit type value, etc.
2. **Weights are sensitive.** Even 1-2 points matter. Test after each new factor.
3. **Watch replays to tune.** When the bot targets the wrong unit, adjust that factor's weight.
4. **80/20 rule.** Simple weights get you most of the way. Don't over-engineer early.
5. **Negative weights are normal.** Many factors subtract from score (distance, penalties).
6. **Scope to squads.** Each unit in a squad scores targets within the squad's engagement zone, not the whole map.
7. **Global before local.** Check "should we fight at all?" before letting individual units pick targets.
8. **Engagement threshold.** Switch from group movement to individual micro once ~30% of units are engaged.

---

## 11. Adaptation Notes for ARES / python-sc2

- Use `unit.position.distance_to()` for distance calculations
- `unit.health`, `unit.health_max`, `unit.shield`, `unit.shield_max` for health scoring
- `unit.ground_range` and `unit.air_range` for range checks
- ARES provides combat sim utilities — check if they can be used for the global/local assessment
- Counter tables can be stored as a dict mapping `(own_type_id, enemy_type_id)` → weight
- Consider frame-time safety: scoring every unit against every enemy is O(n×m). For large armies, consider only scoring enemies within a reasonable radius
- Workers in gas disappear from the API — keep a memory count if tracking enemy worker count

---

*Source: Postbots #19 roundtable transcript. System designed by MindMe (NegativeZero / Protoss / Go-based bot).*
```