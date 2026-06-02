# Task: Add 2021 Stalker-Centric PvT Build to PiG_Bot

## Context
The bot currently has one PvT build (`B2GM_PVT_Standard_Build`, robo-centric, 2023 B2GM).
We are adding a second PvT build option (`B2GM_PVT_Stalker_Centric_2021`) based on PiG's 2021 Bronze-to-GM Diamond 1 / Masters 3 PvT.

This is the **core "if everything goes well" path only**. Reactive branches and turtle transitions are handled by other systems and are out of scope.

---

## Section 1: Core Differences Between 2023 and 2021 PvT

| Dimension | 2023 Current (Robo-Centric) | 2021 New (Stalker-Centric) |
|---|---|---|
| Opener | 1-gate expand | 1-gate expand (same) |
| First tech building | Robo at ~supply 26 | Twilight Council at ~supply 30 |
| First upgrade after Warpgate | Thermal Lance | Blink |
| Second upgrade | Charge | Ground Weapons +1 |
| Upgrade chain | Thermal Lance → Charge → Ground upgrades | Blink → Weapons+1 → Armor+1 → Armor+2 → Weapons+2 → Charge (later) |
| Second tech building | Robo Bay | Forge |
| Third tech building | Second Robo | None in core path |
| Core army | Stalker + Immortal + Colossus + Zealot + Archon | Stalker + Zealot only |
| Gas count | 4 gases (3rd/4th at supply 35) | 2 gases for entire core build |
| Gateway count target | Lower, scales slowly | 3 → 8 at 3rd nexus → 12 at 4th nexus |
| Scouting source | Observer (from Robo) at ~supply 34 | Probe scout only |
| Base count | 3 bases, 66 probes | 4 bases, ~70 probes (minerals only beyond 3rd) |

Fork point: supply 26-30. 2023 builds Robo here. 2021 builds Twilight + extra Gateways instead.

---

## Section 2: Full 2021 PvT Build Order (Top to Bottom)

### Opening (identical to 2023, no changes needed)
- 14 Pylon at ramp wall
- 16 Gateway at ramp wall
- 16 Chrono Nexus
- 16 Worker Scout
- 16 Gas
- 20 Expand
- 20 Cybernetics Core at ramp wall
- 21 Pylon
- 21 Gas
- 22 Warpgate Research
- 22 Chrono Cyber

### Early units (identical to 2023)
- 23 Stalker
- 25 Stalker
- 27 Chrono Gateway
- 27 Stalker

### Fork point (diverges from 2023)
- 30 Twilight Council (replaces 26 Robo)
- 32 Pylon
- 33 Stalker x3 (warp-in round, 6 stalkers total)

### Tech and expansion
- 36 Blink research + Chrono Twilight
- 38 Gateway x2 (3 gates total)
- 40 Forge
- 42 Ground Weapons +1 + Chrono Forge
- 46 Expand (3rd Nexus)
- 48 Pylon

### Mid-game ramp (handoff to dynamic layer starts here)
- 55 Gateway x5 (8 gates total)
- 60 Charge
- 62 Ground Armor +1
- 70 Expand (4th Nexus)
- 75 Gateway x4 (12 gates total)
- 85 Ground Armor +2
- 95 Ground Weapons +2

### Steady-state targets
- Army composition: Stalker-heavy with Zealot support
- Production: 12 Gateways
- Bases: 4
- Gases: 2
- Probes: ~70

---

## Section 3: Changes Required

### 3.1 Build Order File (`protoss_builds.yml`)
- Add new entry: `B2GM_PVT_Stalker_Centric_2021`
- Do NOT include any Robo, Robo Bay, Immortal, or Observer steps
- Insert Twilight → Blink → Forge where Robo → Robo Bay would be
- Add explicit Gateway scaling steps: 3 → 8 → 12
- Add explicit Nexus expansion steps: 3rd → 4th

### 3.2 Army Composition
- Define a new PvT-specific composition for this build
- Primary unit: Stalker
- Secondary unit: Zealot
- Do NOT include Immortal, Colossus, Archon, or HighTemplar in the core composition
- The bot must select this composition when `B2GM_PVT_Stalker_Centric_2021` is the active build

### 3.3 Upgrade Priority
- When this build is active, upgrade research order must be:
  1. Blink (first priority after Warpgate)
  2. Ground Weapons +1
  3. Ground Armor +1
  4. Ground Armor +2
  5. Ground Weapons +2
  6. Charge (later, once gateways and 4th base are up)
- Prevent automatic Charge research triggered by Twilight Council presence. In this build, Twilight exists for Blink.

### 3.4 Scouting Source
- Core path uses probe scout only
- No Observer in the core path

### 3.5 Economy Targets
- Gas count target: 2 gases total (not 4)
- Probe count target: up to ~70
- Beyond the 3rd base, probes go to minerals only
- Base count target: 4

### 3.6 Gateway Scaling Thresholds
- 3 Gateways until 3rd Nexus starts
- 8 Gateways once 3rd Nexus is building
- 12 Gateways once 4th Nexus is building

### 3.7 Build Selection
- The bot must be able to identify which PvT build is active
- Army composition, upgrade priority, and Gateway scaling logic must all branch on the active build
- When `B2GM_PVT_Standard_Build` is active: use existing 2023 behavior
- When `B2GM_PVT_Stalker_Centric_2021` is active: use all the 2021 settings above

---

## Out of Scope (Do Not Implement)
- Reactive cannons vs cloaked widow mines / banshees
- Reactive Robo + Observer vs armoury widow mines
- 3-rax pressure response
- Turtle transition (4 gases, triple Robo, Disruptors, DT Shrine)
- Adept harass / shade micro

These are handled by other systems or are intentionally excluded from this build's core path.
