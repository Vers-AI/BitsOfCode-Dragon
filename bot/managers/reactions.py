# bot/managers/reactions.py
import numpy as np

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.units import Units
from sc2.position import Point2


# Ares imports
from ares.consts import UnitRole, WORKER_TYPES, UnitTreeQueryType
from ares.behaviors.combat.individual import PathUnitToTarget, WorkerKiteBack
from ares.behaviors.combat import CombatManeuver
from ares.managers.manager_mediator import ManagerMediator
from ares.dicts.unit_data import UNIT_DATA

from cython_extensions import (
    cy_distance_to, cy_distance_to_squared, cy_center, cy_find_units_center_mass,
    cy_closest_to, cy_attack_ready, cy_in_attack_range, cy_pick_enemy_target,
    cy_structure_pending_ares
)

from bot.utilities.intel import get_enemy_cannon_rushed
from bot.utilities.cheese_detection import detect_cheese
from bot.constants import (
    UNDER_ATTACK_VALUE_THRESHOLD,
    UNDER_ATTACK_RATIO_THRESHOLD,
    UNDER_ATTACK_CLEAR_VALUE,
    COMMON_UNIT_IGNORE_TYPES,
    MEMORY_EXPIRY_TIME,
)



def defend_cannon_rush(bot):
    """
    Defends against cannon rush by pulling appropriate number of workers.
    Uses continue-based priority chain for clean worker control.
    Workers automatically return to mining when no threats present.
    
    Args:
        bot: The bot instance
    """
    # Get enemy units in base area
    enemy_units: Units = bot.mediator.get_units_in_range(
        start_points=[bot.start_location],
        distances=14,
        query_tree=UnitTreeQueryType.AllEnemy,
    )[0]
    
    enemy_probes = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PROBE)
    enemy_cannons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PHOTONCANNON)
    enemy_pylons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PYLON)
    
    # Check if cannon rush is active
    has_cannon_threats = bool(enemy_probes or enemy_cannons or enemy_pylons)
    
    if not has_cannon_threats:
        # No threats - clean up if we were defending
        if getattr(bot, '_cannon_rush_active', False):
            defending_workers = bot.mediator.get_units_from_role(
                role=UnitRole.DEFENDING,
                unit_type=UnitTypeId.PROBE
            )
            # Return all defenders to gathering
            for worker in defending_workers:
                bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
            
            # Complete cheese reaction build if both threats cleared
            if (bot._used_cheese_response
                and not getattr(bot, '_worker_rush_active', False)  # Other threat also clear
                and bot.build_order_runner.chosen_opening == "Cheese_Reaction_Build"
                and not bot.build_order_runner.build_completed):
                bot.build_order_runner.set_build_completed()
                bot._used_cheese_response = False
                print(f"Cheese reaction build completed - cannon rush cleared at {bot.time:.1f}s")
            
            # Reset flags
            bot._cannon_rush_active = False
            bot._cannon_rush_response = False
            bot._under_attack = False
        return
    
    # Set initial flags if not already set
    if not getattr(bot, '_cannon_rush_active', False):
        bot._cannon_rush_active = True
        bot.build_order_runner.switch_opening("Cheese_Reaction_Build", remove_completed=False)
        bot._used_cheese_response = True
        bot._under_attack = True
        bot._worker_cannon_rush_response = True
    
    # Calculate how many workers to pull (cannon-specific formula)
    workers_needed = min(24, len(enemy_cannons) + (len(enemy_probes) // 2) + 8)
    
    # Get current defending workers
    defending_workers = bot.mediator.get_units_from_role(
        role=UnitRole.DEFENDING,
        unit_type=UnitTypeId.PROBE
    )
    
    # Get workers that should be mining (not already defending)
    available_workers = bot.workers.filter(
        lambda w: w.tag not in defending_workers.tags
    )
    
    # Assign more workers if needed
    while len(defending_workers) < workers_needed and available_workers:
        worker = available_workers.closest_to(bot.start_location)
        if not worker:
            break
        bot.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
        defending_workers.append(worker)
        available_workers.remove(worker)
    
    # Per-worker control with priority chain (from example pattern)
    for worker in defending_workers:
        # 1. Handle resource return (from example)
        if worker.is_carrying_resource and bot.townhalls:
            worker.return_resource()
            continue
        
        # 2. Cannon-specific prioritization (your logic - keep this!)
        # Prioritize cannons that are nearly complete or complete
        urgent_targets = enemy_cannons.filter(
            lambda c: c.build_progress > 0.5 or c.is_ready
        )
        
        if urgent_targets:
            target = cy_closest_to(worker.position, urgent_targets)
            worker.attack(target)
            continue
        
        if enemy_probes:
            target = cy_closest_to(worker.position, enemy_probes)
            # Only attack if in range and ready (smarter targeting)
            if cy_attack_ready(bot, worker, target):
                worker.attack(target)
            else:
                worker.move(target.position)
            continue
        
        if enemy_cannons:  # Cannons < 50% complete
            target = cy_closest_to(worker.position, enemy_cannons)
            worker.attack(target)
            continue
        
        if enemy_pylons:
            target = cy_closest_to(worker.position, enemy_pylons)
            worker.attack(target)
            continue
        
        # 3. Automatic fallback to mining (from example - no timer needed!)
        if bot.mineral_field:
            mf = cy_closest_to(worker.position, bot.mineral_field)
            worker.gather(mf)
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

def defend_worker_rush(bot):
    """
    Defends against worker rush by pulling appropriate number of workers.
    Uses continue-based priority chain for clean worker control.
    Workers automatically return to mining when no threats present.
    
    Args:
        bot: The bot instance
    """
    # Get all enemy units in our base and filter for workers
    defense_point = bot.natural_expansion if bot.structures.closer_than(8, bot.natural_expansion) else bot.start_location
    
    enemy_units = bot.mediator.get_units_in_range(
        start_points=[defense_point],
        distances=25,  # Larger radius to catch workers coming in
        query_tree=UnitTreeQueryType.AllEnemy,
    )[0]
    enemy_workers = enemy_units.filter(lambda u: u.type_id in WORKER_TYPES)

    # Check if worker rush is active
    has_worker_threats = bool(enemy_workers)
    
    if not has_worker_threats:
        # No threats - clean up if we were defending
        if getattr(bot, '_worker_rush_active', False):
            defending_workers = bot.mediator.get_units_from_role(
                role=UnitRole.DEFENDING,
                unit_type=UnitTypeId.PROBE
            )
            # Return all defenders to gathering
            for worker in defending_workers:
                bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
            
            # Complete cheese reaction build if both threats cleared
            if (bot._used_cheese_response
                and not getattr(bot, '_cannon_rush_active', False)  # Other threat also clear
                and bot.build_order_runner.chosen_opening == "Cheese_Reaction_Build"
                and not bot.build_order_runner.build_completed):
                bot.build_order_runner.set_build_completed()
                bot._used_cheese_response = False
                print(f"Cheese reaction build completed - worker rush cleared at {bot.time:.1f}s")
            
            # Reset flags
            bot._worker_rush_active = False
            bot._not_worker_rush = True
            bot._under_attack = False
        return

    # Set initial flags if not already set
    if not getattr(bot, '_worker_rush_active', False):
        bot._worker_rush_active = True
        bot.build_order_runner.switch_opening("Cheese_Reaction_Build", remove_completed=False)
        bot._used_cheese_response = True
        bot._under_attack = True
        bot._not_worker_rush = False

    # Get current defending workers
    defending_workers = bot.mediator.get_units_from_role(
        role=UnitRole.DEFENDING,
        unit_type=UnitTypeId.PROBE
    )
    
    # Calculate how many workers to pull (worker rush specific: 1.5x enemy workers)
    workers_needed = min(16, max(4, int(len(enemy_workers) * 1.5)))
    
    # Get workers that should be mining (not already defending)
    available_workers = bot.workers.filter(
        lambda w: w.tag not in defending_workers.tags
    )
    
    # Assign more workers if needed
    while len(defending_workers) < workers_needed and available_workers:
        worker = available_workers.closest_to(bot.start_location)
        if not worker:
            break
        bot.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
        defending_workers.append(worker)
        available_workers.remove(worker)
    
    # Per-worker control with priority chain (from example pattern)
    for worker in defending_workers:
        # 1. Handle resource return (from example)
        if worker.is_carrying_resource and bot.townhalls:
            worker.return_resource()
            continue
        
        # 2. Worker rush specific: use WorkerKiteBack for micro
        if enemy_workers:
            target = cy_closest_to(worker.position, enemy_workers)
            # Use WorkerKiteBack behavior for better micro (kiting)
            bot.register_behavior(WorkerKiteBack(unit=worker, target=target))
            continue
        
        # 3. Automatic fallback to mining (from example - no timer needed!)
        if bot.mineral_field:
            mf = cy_closest_to(worker.position, bot.mineral_field)
            worker.gather(mf)
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)


def cheese_reaction(bot):
    """
    Builds pylon/gateway/shield battery to defend early cheese.
    """
    #print(f"Current build: {bot.build_order_runner.chosen_opening}")
    
    remove_completed = False
    if bot.structures(UnitTypeId.CYBERNETICSCORE).exists:
        remove_completed = True
    
    bot.build_order_runner.switch_opening("Cheese_Reaction_Build", remove_completed=remove_completed)
    
    # Cancel a fast-expanding Nexus if it's started and we detect cheese
    pending_townhalls = cy_structure_pending_ares(bot, UnitTypeId.NEXUS)
    if pending_townhalls == 1 and bot.time < 2 * 60:
        for pt in bot.townhalls.not_ready:
            bot.mediator.cancel_structure(structure=pt)

    



def early_threat_sensor(bot):
    """
    Detects early threats like zergling rush, proxy zealots, etc.
    Sets flags so the bot can respond (e.g., cheese_reaction).
    """
    if bot.mediator.get_enemy_worker_rushed and bot.game_state == 0:
        bot._not_worker_rush = False
        bot._used_cheese_response = True
    
    # Check for cannon rush
    elif get_enemy_cannon_rushed(bot):
        bot._used_cheese_response = True
        bot._cannon_rush_response = True
    
    elif (
        (detect_cheese(bot))
        or (bot.mediator.get_enemy_marauder_rush and bot.time < 150.0)
        or bot.mediator.get_enemy_marine_rush
        or bot.mediator.get_is_proxy_zealot
        or bot.mediator.get_enemy_ravager_rush
        or bot.mediator.get_enemy_went_marine_rush
        or bot.mediator.get_enemy_four_gate
        or bot.mediator.get_enemy_roach_rushed
    ):
        bot._used_cheese_response = True
    


# ===== THREAT ASSESSMENT FUNCTIONS =====
# Moved from combat.py for better separation of concerns

def assess_threat_severity(bot, enemy_units: Units, threat_location: Point2) -> dict:
    """
    Enhanced threat assessment that determines both severity and appropriate response.
    Returns dict with threat_level, required_units, and response_type.
    """
    if not enemy_units:
        return {"threat_level": 0, "required_units": 0, "response_type": "none"}
    
    # Calculate base threat value
    threat_value = 0
    unit_count = enemy_units.amount
    
    # Categorize threats by type
    harassment_units = enemy_units.filter(lambda u: u.type_id in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG
    })
    
    combat_units = enemy_units.filter(lambda u: u.type_id not in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG,
        UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE
    })
    
    # Weight different unit types
    for unit in enemy_units:
        if unit.type_id in UNIT_DATA:
            base_weight = UNIT_DATA[unit.type_id]['army_value']
            # Scale by health percentage
            health_factor = (unit.health + unit.shield) / (unit.health_max + unit.shield_max)
            threat_value += base_weight * health_factor
    
    # Distance factor - closer threats are more urgent
    closest_base = min(bot.townhalls, key=lambda th: cy_distance_to_squared(th.position, threat_location))
    distance_to_base = cy_distance_to(threat_location, closest_base.position)
    distance_factor = max(0.5, 2.0 - (distance_to_base / 20.0))
    threat_value *= distance_factor
    
    # Determine response type and required units
    if harassment_units.amount >= unit_count * 0.8 and unit_count <= 6:
        # Mostly harassment units
        response_type = "harassment_response"
        required_units = min(4, max(2, unit_count))
        threat_level = min(3, threat_value)
    elif combat_units.amount > 3 or threat_value > 15:
        # Significant combat threat
        response_type = "combat_response" 
        required_units = max(6, int(threat_value * 0.8))
        threat_level = min(10, threat_value)
    else:
        # Mixed or small threat
        response_type = "patrol_response"
        required_units = min(3, max(1, unit_count // 2))
        threat_level = min(5, threat_value)
    
    return {
        "threat_level": int(threat_level),
        "required_units": required_units,
        "response_type": response_type,
        "harassment_ratio": harassment_units.amount / max(1, unit_count),
        "unit_count": unit_count,
        "threat_value": threat_value
    }


def assess_threat(bot, enemy_units: Units, own_forces: Units, return_details: bool = False):
    """
    Enhanced threat assessment that can return simple score or detailed analysis.
    
    Args:
        return_details: If True, returns dict with detailed analysis. If False, returns int score.
    """
    if not enemy_units:
        return {"threat_level": 0, "required_units": 0, "response_type": "none"} if return_details else 0
    
    # Calculate base threat value
    threat_value = 0
    unit_count = enemy_units.amount
    
    # Check for damage-dealing low threats (e.g., lings attacking probes/buildings)
    damage_bonus = _assess_damage_threat(bot, enemy_units)
    
    # Categorize threats by type  
    harassment_units = enemy_units.filter(lambda u: u.type_id in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG
    })
    
    # Weight different unit types
    for unit in enemy_units:
        if unit.type_id in UNIT_DATA:
            base_weight = UNIT_DATA[unit.type_id]['army_value']
            # Scale by health percentage
            health_factor = (unit.health + unit.shield) / max(1, unit.health_max + unit.shield_max)
            threat_value += base_weight * health_factor
    
    # Density check: adjust threat level based on enemy clustering
    center = cy_center(enemy_units)
    cluster_count = 0
    for unit in enemy_units:
        if cy_distance_to(unit.position, center) <= 5.0:
            cluster_count += 1
    
    # If fewer than 3 enemy units are clustered, scale down the threat level
    if cluster_count < 3:
        threat_value *= 0.5
    
    # Apply damage bonus for units actively damaging our assets
    threat_value += damage_bonus
    
    # Simple integer return for backward compatibility
    if not return_details:
        return max(round(threat_value), 0)
    
    # Enhanced details for new system
    # (This replaces assess_threat_severity functionality)
    if len(bot.townhalls) > 0:
        closest_base = min(bot.townhalls, key=lambda th: cy_distance_to_squared(th.position, center))
        distance_to_base = cy_distance_to(center, closest_base.position)
        distance_factor = max(0.5, 2.0 - (distance_to_base / 20.0))
        threat_value *= distance_factor
    
    # Apply damage bonus again after distance factor (for detailed analysis)
    threat_value += damage_bonus
    
    # Determine response type and required units
    # Damage-dealing threats get elevated response even if low unit count
    if damage_bonus > 2.0:  # Critical damage being dealt
        response_type = "damage_response"
        required_units = min(3, max(1, unit_count + 1))  # +1 extra for damage threats
        threat_level = min(6, max(3, threat_value))  # Minimum level 3 for damage threats
    elif harassment_units.amount >= unit_count * 0.8 and unit_count <= 6:
        response_type = "harassment_response"
        required_units = min(4, max(2, unit_count))
        threat_level = min(3, threat_value)
    elif unit_count > 3 and threat_value > 10:
        response_type = "combat_response" 
        required_units = max(6, int(threat_value * 0.8))
        threat_level = min(10, threat_value)
    else:
        response_type = "patrol_response"
        required_units = min(3, max(1, unit_count // 2))
        threat_level = min(5, threat_value)
    
    return {
        "threat_level": int(threat_level),
        "required_units": required_units,
        "response_type": response_type,
        "harassment_ratio": harassment_units.amount / max(1, unit_count),
        "unit_count": unit_count,
        "threat_value": threat_value
    }


def _assess_damage_threat(bot, enemy_units: Units) -> float:
    """
    Assess if low-threat units are actively doing damage to our assets.
    Returns bonus threat value for units near damaged/low-health friendly units.
    """
    damage_bonus = 0.0
    
    # Get our damaged units (buildings + workers)
    damaged_buildings = bot.structures.filter(lambda s: s.health_percentage < 1.0)
    damaged_workers = bot.workers.filter(lambda w: w.health_percentage < 1.0)
    
    # Get very low health units that need immediate attention
    critical_buildings = bot.structures.filter(lambda s: s.health_percentage < 0.5)
    critical_workers = bot.workers.filter(lambda w: w.health_percentage < 0.3)
    
    # Check if enemy units are near damaged assets
    for enemy in enemy_units:
        # Higher bonus for units near critical assets
        if critical_buildings:
            closest_critical = critical_buildings.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_critical.position) <= 3.0:
                damage_bonus += 3.0  # High priority for units attacking critical buildings
        
        if critical_workers:
            closest_critical_worker = critical_workers.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_critical_worker.position) <= 2.0:
                damage_bonus += 2.0  # Medium priority for units attacking low-health workers
        
        # Medium bonus for units near damaged assets
        if damaged_buildings:
            closest_damaged = damaged_buildings.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_damaged.position) <= 3.0:
                damage_bonus += 1.5
        
        if damaged_workers:
            closest_damaged_worker = damaged_workers.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_damaged_worker.position) <= 2.0:
                damage_bonus += 1.0
    
    return damage_bonus


def allocate_defensive_forces(bot, threat_info: dict, threat_location: Point2, enemy_units: Units) -> Units:
    """
    Smart unit allocation based on threat value and air/ground requirements.
    Uses army_value matching and capability filtering like the examples.
    """
    available_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    
    if not available_units:
        return Units([], bot)
    
    # Dynamic threat composition check (following example patterns)
    has_air = any(unit.is_flying for unit in enemy_units)
    has_ground = any(not unit.is_flying for unit in enemy_units)
    
    # Filter for units that can actually attack (exclude disruptors, etc.)
    combat_capable = available_units.filter(lambda u: u.can_attack)
    
    # Filter capable units (exactly like examples)
    if has_air and has_ground:
        # Mixed threat - prioritize units that can attack both, fallback to air-capable
        capable_units = combat_capable.filter(lambda u: u.can_attack_both)
        if not capable_units:
            capable_units = combat_capable.filter(lambda u: u.can_attack_air)
    elif has_air:
        capable_units = combat_capable.filter(lambda u: u.can_attack_air)
    else:
        capable_units = combat_capable.filter(lambda u: u.can_attack_ground)
    
    if not capable_units:
        # Fallback to any combat capable units if no specific match found
        capable_units = combat_capable
    
    response_type = threat_info["response_type"]
    threat_value = threat_info.get("threat_value", 0)
    
    # For damage-dealing threats, send closest capable units immediately
    if response_type == "damage_response":
        closest_units = capable_units.sorted(lambda u: cy_distance_to_squared(u.position, threat_location))
        required_count = min(threat_info["required_units"], len(closest_units))
        return closest_units[:required_count]
    
    # Value-based allocation for proportional response
    selected_units = []
    current_value = 0.0
    target_value = max(threat_value, 1.0)  # Match threat value exactly
    
    # Sort by distance (closest units respond, regardless of value)
    sorted_units = capable_units.sorted(lambda u: cy_distance_to_squared(u.position, threat_location))
    
    # Unit allocation logic
    
    # Select units until threat is adequately countered
    for unit in sorted_units:
        unit_value = UNIT_DATA.get(unit.type_id, {}).get('army_value', 1.0)
        
        # Add this unit
        selected_units.append(unit)
        current_value += unit_value
        
        # Track current allocation value
        
        # Stop when we have enough value OR enough units for safety
        if current_value >= target_value or len(selected_units) >= 3:
            break
    
    # Ensure minimum response for non-trivial threats
    if not selected_units and capable_units:
        selected_units = [capable_units.closest_to(threat_location)]
    
    return Units(selected_units, bot)


def threat_detection(bot, main_army: Units) -> None:
    """
    Enhanced threat detection with smart force allocation.
    Detects threats near bases and responds with appropriate force levels
    instead of always redirecting the entire main army.
    """
    # Import here to avoid circular dependency
    from bot.combat import control_main_army
    
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases

    # Combine ground + air threats
    all_threats = {}
    for key, value in ground_near.items():
        all_threats[key] = value.copy()
    for key, value in flying_near.items():
        all_threats.setdefault(key, set()).update(value)
    
    # Aggregate total threat value from all bases for global _under_attack flag
    total_threat_value = 0
    if all_threats:
        all_enemy_tags = set()
        for enemy_tags in all_threats.values():
            all_enemy_tags.update(enemy_tags)
        all_threatening_units = bot.enemy_units.tags_in(all_enemy_tags)
        
        if all_threatening_units:
            threat_info = assess_threat(bot, all_threatening_units, main_army, return_details=True)
            assert isinstance(threat_info, dict), "assess_threat with return_details=True should return dict"
            total_threat_value = threat_info.get("threat_value", 0)
    
    # Calculate threat ratio: (threat near bases) / (total known enemy army)
    # Filters: workers, scouts (overlords/observers), and non-combat units
    # Filter expired ghosts (age >= 30s) from threat ratio denominator;
    # UnitCacheManager retains units indefinitely but they expire from memory at 30s
    known_enemy = bot.mediator.get_cached_enemy_army
    known_army_value = sum(
        UNIT_DATA.get(u.type_id, {}).get('army_value', 1.0) 
        for u in known_enemy 
        if u.type_id not in WORKER_TYPES and u.type_id not in COMMON_UNIT_IGNORE_TYPES
        and u.age < MEMORY_EXPIRY_TIME
    ) if known_enemy else 0
    threat_ratio = total_threat_value / max(known_army_value, 1.0)
    
    # Update _under_attack flag with hysteresis (higher threshold to set, lower to clear)
    # Runs every frame even when no threats detected to allow flag to clear properly
    if bot._under_attack:
        if total_threat_value < UNDER_ATTACK_CLEAR_VALUE:
            bot._under_attack = False
    else:
        if total_threat_value >= UNDER_ATTACK_VALUE_THRESHOLD:
            bot._under_attack = True
        elif threat_ratio >= UNDER_ATTACK_RATIO_THRESHOLD:
            bot._under_attack = True

    # Detect cloaked/burrowed enemies near bases that need detection
    # These may not appear in ground_near/flying_near if partially visible
    # but still pose a threat our defenders can't fight without an observer
    cloaked_threat_positions: list[Point2] = []
    if bot.townhalls:
        th_positions = [th.position for th in bot.townhalls]
        enemies_near_bases = bot.mediator.get_units_in_range(
            start_points=th_positions,
            distances=18,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )
        for result in enemies_near_bases:
            for enemy in result:
                if not (enemy.is_cloaked or enemy.is_burrowed):
                    continue
                if enemy.is_revealed:
                    continue
                cloaked_threat_positions.append(enemy.position)
    bot._cloaked_threat_positions = cloaked_threat_positions

    # Per-base defender allocation
    if all_threats:
        main_army_should_respond = False
        
        for base_location, enemy_tags in all_threats.items():
            enemy_units = bot.enemy_units.tags_in(enemy_tags)
            if not enemy_units:
                continue
                
            threat_position, _ = cy_find_units_center_mass(enemy_units, 10.0)
            threat_position = Point2(threat_position)
            
            # Assess threat level for this specific base location
            threat_info = assess_threat(bot, enemy_units, main_army, return_details=True)
            assert isinstance(threat_info, dict), "assess_threat with return_details=True should return dict"
            base_threat_value = threat_info.get("threat_value", 0)
            
            # Allocate defenders proportionally to threat level with capped response
            if base_threat_value > 0:
                existing_defenders = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
                
                # Cap defenders at 150% of threat value (minimum 10) to avoid over-committing
                existing_defender_value = sum(UNIT_DATA.get(u.type_id, {}).get('army_value', 1.0) for u in existing_defenders)
                max_defender_value = max(total_threat_value * 1.5, 10.0)
                
                defender_cap_reached = existing_defender_value >= max_defender_value
                
                # Assign new defenders only if below cap (unit control handled in combat.py)
                if not defender_cap_reached:
                    new_defenders = allocate_defensive_forces(bot, threat_info, threat_position, enemy_units)
                    if new_defenders:
                        for unit in new_defenders:
                            bot.mediator.assign_role(tag=unit.tag, role=UnitRole.BASE_DEFENDER)
                
                # Store threat position for defensive targeting in combat systems
                bot._defender_threat_position = threat_position
        
        # Flag main army involvement for coordination with attack logic
        bot._main_army_defending = main_army_should_respond
