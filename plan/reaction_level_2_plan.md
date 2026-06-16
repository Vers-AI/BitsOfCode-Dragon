# Reaction Level 2 — PiGBot Strategy Responses

Derived from PiG's B2GM 2023 guide. These are **what the bot should do** once a strategy is detected. Detection is handled by the Bayesian belief layer (Level 1). This document defines the reaction policies only.

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

---

## Policy 2: Cannon Rush

**Trigger label:** `cheese/cannon_rush`

**PiG's reaction:**
1. If no enemy Forge visible → ignore the pylons (they're just vision, not cannon rush)
2. Pull workers to kill in priority order: complete/near-complete Cannons → Probes → Pylons
3. Battery on natural

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

---

## Coverage Check

| Area | Policy | B2GM Source |
|------|--------|-------------|
| Cheese / early all-in | Policy 1 | Standard Cheese Reaction section |
| Cannon rush | Policy 2 | "Vs Cannon Rush" section |
| Proxy rax/gate | Policy 3 | "Response to proxy rax" + "Big reaction / Small reaction" |
| All-in (2-base push, ravager/ling, chargelot, BC) | Policy 4 | Multiple game lessons + "Big 2-base push" + "Ravager ling all-in" |
| Air (mutas, skytoss, BCs, turtle) | Policy 5 | "MUTALISKS" section + "How to Vs Skytoss/Skyterran/Turtle Players" |