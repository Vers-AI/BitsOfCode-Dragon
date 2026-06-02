# PvZ Zergling Rush Detector — Hybrid Rule + ML Spec

## Overview

A hybrid detection system combining expert heuristics with a lightweight ML model.

- **Rules** handle clear, early rushes (auto-TRUE guards)
- **ML** refines classification when rules are uncertain or scouting is incomplete
- **Goal:** Improve accuracy while preserving reliability of hand-crafted rules

---

## Labels

| Label | Definition |
|-------|------------|
| **12_pool** | Very early spawning pool (~12 supply), earliest Zergling rush |
| **speedling** | Early pool + early gas for fast Metabolic Boost timing rush |
| **none** | Not a rush (negative class) |

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────┐
│                  AUTO-TRUE GUARDS                       │
│   Early ling ≤1:45 → 12_pool                           │
│   Slow-ling contact ≤2:00 → 12_pool                    │
│   Speed-ling contact ≤2:40 → speedling                 │
│                  ↓ fires? → return label (skip ML)      │
└─────────────────────────────────────────────────────────┘
                         │ no
                         ▼
┌─────────────────────────────────────────────────────────┐
│               FEATURE EXTRACTION                        │
│  Raw timings: pool_start, nat_time, gas_time,          │
│               queen_time, ling_seen, ling_contact,      │
│               speed_start, ling_has_speed               │
│  Rule scores: score_12p, score_speed                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            LOGISTIC REGRESSION MODEL                    │
│  Input: feature vector                                  │
│  Output: p(12_pool), p(speedling), p(none)             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FINAL CLASSIFICATION                       │
│  if max(probs) > 0.55 → return ML label                │
│  else → rule-based tiebreaker:                         │
│      score_12p ≥ 5 → "12_pool"                         │
│      score_speed ≥ 5 → "speedling"                     │
│      else → "none"                                      │
└─────────────────────────────────────────────────────────┘
```

---

## Timing Constants (Faster game seconds)

| Name | Time | Purpose |
|---|---:|---|
| `T_POOL_12P_START_MIN` | **38.0** (≈0:38) | 12-pool start window min |
| `T_POOL_12P_START_MAX` | **42.0** (≈0:42) | 12-pool start window max |
| `T_POOL_12P_DONE` | **65.0** (≈1:05) | 12-pool completion |
| `T_POOL_SPEED_START_MIN` | **48.0** (≈0:48) | Speedling pool start min |
| `T_POOL_SPEED_START_MAX` | **52.0** (≈0:52) | Speedling pool start max |
| `T_NAT_CHECK` | **80.0** (≈1:20) | Natural hatch expected by this time |
| `T_LING_EARLY` | **105.0** (≈1:45) | Auto-TRUE: early ling sighting |
| `T_LING_12P_SEEN` | **115.0** (≈1:55) | 12-pool lings seen threshold |
| `T_CONTACT_SLOW` | **120.0** (≈2:00) | Auto-TRUE: slow-ling contact |
| `T_LING_SPEED_SLOW` | **135.0** (≈2:15) | Speedling slow-lings appear |
| `T_CONTACT_SPEED` | **160.0** (≈2:40) | Auto-TRUE: speed-ling contact |
| `T_SPEED_START_EARLY` | **85.0** (≈1:25) | Early speed research start |
| `T_QUEEN_CHECK` | **120.0** (≈2:00) | No queen by this time = rush signal |

**Late-scout reconstruction:**
```python
start_time = bot.time - building.build_progress * building._type_data.cost.time / 22.4
```

---

## Auto-TRUE Guards (evaluate first, skip ML if fires)

| Guard | Condition | Result |
|---|---|---|
| A | Any Zergling **seen ≤ 1:45** | **Return `12_pool`** |
| B | Slow-ling **contacts nat ≤ 2:00** | **Return `12_pool`** |
| C | Speed-ling **contacts nat ≤ 2:40** | **Return `speedling`** |

---

## Rule Signals: `12_pool` (score_12p)

| Signal | Condition | Weight |
|---|---|---:|
| Early pool start | Pool start ≈ **0:38–0:42** | +4 |
| Pool done early | Pool **done ≤ 1:05** | +3 |
| No natural | Natural absent at **≥ 1:20** | +3 |
| Very early lings | Lings **seen ≤ 1:55** | +4 |
| No queen | No queen by **2:00** | +2 |

**Notes:** Gas may be present or absent (does NOT change the label)

---

## Rule Signals: `speedling` (score_speed)

| Signal | Condition | Weight |
|---|---|---:|
| Speedling pool timing | Pool start ≈ **0:48–0:52** | +3 |
| Early gas | Extractor with workers mining early | +3 |
| Speed research early | Speed starts **≤ 1:25** | +4 |
| Slow lings appear | Lings seen around **2:05–2:15** | +2 |
| Speed-ling contact | Speedlings contact **≤ 2:40** | +4 |
| Queen timing normal | Queen started **1:50–2:00** | −1 (damper) |
| Natural on time | Natural hatch **≤ 1:20** | −1 (damper) |

**Notes:** Natural may appear on time (disguise element)

---

## Feature Vector (for ML)

All features normalized or encoded for the logistic regression model.

### Raw Timing Features
| Feature | Description | Encoding |
|---------|-------------|----------|
| `pool_start` | Estimated pool start time | seconds (float), -1 if unknown |
| `nat_start` | Estimated natural start time | seconds (float), -1 if absent |
| `gas_time` | Extractor first seen time | seconds (float), -1 if none |
| `queen_time` | Queen first seen time | seconds (float), -1 if none |
| `ling_seen` | First ling sighting time | seconds (float), -1 if none |
| `ling_contact` | First ling contact at nat | seconds (float), -1 if none |
| `speed_start` | Speed research start time | seconds (float), -1 if none |
| `ling_has_speed` | Lings have Metabolic Boost | 0 or 1 |
| `gas_workers` | Workers on gas (if visible) | int, 0 if unknown |

### Rule Score Features
| Feature | Description |
|---------|-------------|
| `score_12p` | Rule-computed 12_pool score |
| `score_speed` | Rule-computed speedling score |

**Total features:** 11

---

## ML Model

- **Type:** Logistic Regression (multi-class, softmax)
- **Training:** Offline, from logged game data
- **File:** `data/rush_detector_model.pkl` (or similar)
- **Loaded at:** Bot initialization (`on_start`)

**Why Logistic Regression:**
- Works with small datasets (100–200 examples)
- Interpretable coefficients
- Smooth probability outputs
- Fast inference
- Easy to debug

---

## Classification Logic (Pseudocode)

```python
def classify_rush(bot) -> str:
    # 1. Auto-TRUE guards (skip ML entirely)
    if first_ling_seen <= 105.0:  # 1:45
        return "12_pool"
    if first_ling_contact <= 120.0 and not ling_has_speed:
        return "12_pool"
    if first_ling_contact <= 160.0 and ling_has_speed:
        return "speedling"
    
    # 2. Extract features
    features = extract_feature_vector(bot)
    
    # 3. Compute rule scores
    score_12p = compute_12pool_score(bot)
    score_speed = compute_speedling_score(bot)
    features["score_12p"] = score_12p
    features["score_speed"] = score_speed
    
    # 4. ML prediction
    probs = model.predict_proba(features)  # [p_12pool, p_speedling, p_none]
    max_prob = max(probs)
    ml_label = ["12_pool", "speedling", "none"][argmax(probs)]
    
    # 5. Final decision
    if max_prob > 0.55:
        return ml_label
    else:
        # Rule-based tiebreaker
        if score_12p >= 5:
            return "12_pool"
        elif score_speed >= 5:
            return "speedling"
        else:
            return "none"
```

---

## Logging Schema

Log every classification for training data collection.

```json
{
    "map_name": "string",
    "rush_distance_seconds": float,
    
    // Raw timing features
    "pool_start": float,
    "nat_start": float,
    "gas_time": float,
    "queen_time": float,
    "ling_seen": float,
    "ling_contact": float,
    "speed_start": float,
    "ling_has_speed": bool,
    "gas_workers": int,
    
    // Rule scores
    "score_12p": int,
    "score_speed": int,
    
    // ML outputs (once model is deployed)
    "p_12pool": float,
    "p_speedling": float,
    "p_none": float,
    
    // Final classification
    "auto_true_fired": bool,
    "rush_label": "12_pool" | "speedling" | "none",
    
    // Outcome (for training)
    "result": "Victory" | "Defeat",
    "game_time_seconds": float
}
```

---

## Implementation Phases

### Phase 1: Data Collection (current)
- Log all features + rule-based label
- No ML model yet
- Build training dataset

### Phase 2: Model Training (offline)
- Train logistic regression on logged data
- Validate accuracy
- Export model file

### Phase 3: Hybrid Deployment
- Load model at bot init
- Use ML predictions with 0.55 confidence threshold
- Rule-based tiebreaker for uncertain cases
- Continue logging for model improvement


### Notes
- Zerg buildings count as a worker when counting workers for models