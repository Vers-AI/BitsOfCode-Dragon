from typing import Optional
from itertools import cycle

from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole, UnitTreeQueryType
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import AMove, ShootTargetInRange, KeepUnitSafe, PathUnitToTarget, StutterUnitBack
from ares.behaviors.combat.group import AMoveGroup, PathGroupToTarget, KeepGroupSafe, StutterGroupBack
from ares.behaviors.macro import SpawnController, ProductionController, AutoSupply
from ares.managers.manager_mediator import ManagerMediator

from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_closest_to, cy_pick_enemy_target, cy_find_units_center_mass, cy_attack_ready

from itertools import chain


from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId 
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2

from bot.speedmining import get_speedmining_positions
from bot.speedmining import split_workers
from bot.speedmining import mine

import numpy as np


COMMON_UNIT_IGNORE_TYPES: set[UnitTypeId] = {
            UnitTypeId.EGG,
            UnitTypeId.LARVA,
            UnitTypeId.CREEPTUMORBURROWED,
            UnitTypeId.CREEPTUMORQUEEN,
            UnitTypeId.CREEPTUMOR,
            UnitTypeId.MULE,
            UnitTypeId.PROBE,
            UnitTypeId.SCV,
            UnitTypeId.DRONE,
            UnitTypeId.OVERLORD,
            UnitTypeId.OVERSEER,
            UnitTypeId.LOCUSTMP,
            UnitTypeId.LOCUSTMPFLYING,
            UnitTypeId.ADEPTPHASESHIFT,
            UnitTypeId.CHANGELING,
            UnitTypeId.CHANGELINGMARINE,
            UnitTypeId.CHANGELINGZEALOT,
            UnitTypeId.CHANGELINGZERGLING,
}

class DragonBot(AresBot):
    current_base_target: Point2
    expansions_generator: cycle
    _begin_attack_at_supply: float

    def __init__(self, game_step_override: Optional[int] = None):
        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)
        
        self.townhall_saturations = {}               # lists the mineral saturation of townhalls in queues of 40 frames, we consider the townhall saturated if max_number + 1 >= ideal_number
        self.assimilator_age = {}                    # this is here to tackle an issue with assimilator having 0 workers on them when finished, although the building worker is assigned to it
        self.unit_roles = {}                         # dictionary to keep track of the roles of the units
        self.scout_targets = {}                      # dictionary to keep track of scout targets
        self.bases = {}                              # dictionary to keep track of bases
    
        # Flags
        self._commenced_attack: bool = False
        self._used_cheese_defense: bool = False
        self._used_rush_defense: bool = False
        self._under_attack: bool = False
        self.assigned_ranged: bool = False # for ranged units
    
    @property
    def attack_target(self) -> Point2:
        # if we already have a target and it's still alive, stick with it
        if hasattr(self, '_attack_target') and self._attack_target in self.enemy_structures:
            print("Target still alive", self._attack_target)
            return self._attack_target.position

        elif self.enemy_structures:
            self._attack_target = cy_pick_enemy_target(self.enemy_structures)
            print("Picking target", self._attack_target)
            return self._attack_target.position
            
        # not seen anything in early game, just head to enemy spawn
        elif self.time < 240.0:
            return self.enemy_start_locations[0]
        # else search the map
        else:
            # cycle through expansion locations
            if self.is_visible(self.current_base_target):
                self.current_base_target = next(self.expansions_generator)
    
            return self.current_base_target
    
    # Army Compositions
    @property
    def Standard_Army(self) -> dict:
        return {
            UnitTypeId.IMMORTAL: {"proportion": 0.4, "priority": 0},
            UnitTypeId.COLOSSUS: {"proportion": 0.3, "priority": 2},
            UnitTypeId.HIGHTEMPLAR: {"proportion": 0.2, "priority": 1},
            UnitTypeId.ZEALOT: {"proportion": 0.1, "priority": 3},
        }
    
    @property
    def cheese_defense_army(self) -> dict:
        return {
            UnitTypeId.ZEALOT: {"proportion": 0.5, "priority": 0},
            UnitTypeId.STALKER: {"proportion": 0.4, "priority": 1},
            UnitTypeId.ADEPT: {"proportion": 0.1, "priority": 2},
        }

    async def on_start(self) -> None:
        await super(DragonBot, self).on_start()
        
        print("Game started")
        self.speedmining_positions = get_speedmining_positions(self)
        split_workers(self)   

        self.nexus_creation_times = {nexus.tag: self.time for nexus in self.townhalls.ready}  # tracks the creation time of Nexus

        self.current_base_target = self.enemy_start_locations[0]  # set the target to the enemy start location
        self.bases = {nexus.tag: nexus for nexus in self.townhalls.ready}  # store the bases in a dictionary
        
        # Sort the expansion locations by distance to the enemy start location
        self.expansion_locations_list.sort(key=lambda loc: loc.distance_to(self.enemy_start_locations[0]))

        # Use the sorted expansion locations as your scout targets
        self.scout_targets = self.expansion_locations_list
        
        self.natural_expansion: Point2 = await self.get_next_expansion()
        self._begin_attack_at_supply = 30.0
        
        self.expansions_generator = cycle(
            [pos for pos in self.expansion_locations_list]
        )

        print("Build Chosen:",self.build_order_runner.chosen_opening)

        from sc2.ids.unit_typeid import UnitTypeId
        
        
            

    async def on_step(self, iteration: int) -> None:
        await super(DragonBot, self).on_step(iteration)

        self.resource_by_tag = {unit.tag: unit for unit in chain(self.mineral_field, self.gas_buildings)}

        mine(self, iteration)

        # retrieve all attacking units & scouts
        Main_Army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        Scout = self.mediator.get_units_from_role(role=UnitRole.SCOUTING)
        Warp_Prism = self.mediator.get_units_from_role(role=UnitRole.DROP_SHIP)
        worker_scouts: Units = self.mediator.get_units_from_role(role=UnitRole.BUILD_RUNNER_SCOUT, unit_type=self.worker_type)
        
  

        # Detect threats
        # If there are enemy units near our bases, respond to the threat
        if self.townhalls.exists and self.all_enemy_units.closer_than(30, self.townhalls.center):
            self.threat_response(Main_Army)

        # Checks for cheese defense
        if self.time > 2*60 and self.time < 3*60 + 30:
            if worker_scouts.exists:
                enemy_buildings = self.enemy_structures
                if enemy_buildings.amount == 1 and self.enemy_structures.of_type([UnitTypeId.NEXUS, UnitTypeId.COMMANDCENTER, UnitTypeId.HATCHERY]).exists:
                    self.build_order_runner.set_build_completed()
                    self.register_behavior(SpawnController(self.cheese_defense_army))
                    self.register_behavior(ProductionController(self.cheese_defense_army, base_location=self.start_location))
                    self._used_cheese_defense = True
        # Backstop check for if something went wrong
        if self.minerals > 1200 and self.build_order_runner.build_completed == False:
            self.build_order_runner.set_build_completed()
            self.register_behavior(SpawnController(self.Standard_Army))
            self.register_behavior(ProductionController(self.Standard_Army, base_location=self.start_location))
            
        

        if self._used_cheese_defense or self._used_rush_defense:
            if self.get_total_supply(Main_Army) <= self._begin_attack_at_supply:
                self._commenced_attack = False
            elif self._commenced_attack and not self._under_attack:
                self.Control_Main_Army(Main_Army, self.attack_target)

            elif self.get_total_supply(Main_Army) >= self._begin_attack_at_supply:
                self._commenced_attack = True


        ## Macro and Army control
        if self.build_order_runner.build_completed and not self._used_cheese_defense and not self._used_rush_defense:
            self.register_behavior(AutoSupply(base_location=self.start_location))
            # self.register_behavior(ProductionController(self.Standard_Army, base_location=self.start_location))  # disabled for now

            if Warp_Prism:
                    prism_location = Warp_Prism[0].position
                    self.register_behavior(SpawnController(self.Standard_Army,spawn_target=prism_location))
            else:
                self.register_behavior(SpawnController(self.Standard_Army))
            
            if self.get_total_supply(Main_Army) <= self._begin_attack_at_supply:
                self._commenced_attack = False
            elif self._commenced_attack and not self._under_attack:             
                self.Control_Main_Army(Main_Army, self.attack_target)
                
            elif self.get_total_supply(Main_Army) >= self._begin_attack_at_supply:
                self._commenced_attack = True

        # Additional Probes
        if self.townhalls.amount == 3 and self.workers.amount < 66:
            if self.can_afford(UnitTypeId.PROBE):
                self.train(UnitTypeId.PROBE)

        ### FAIL SAFES
        #Activate the scout if it exists if not build one
        if Scout:
            self.Control_Scout(Scout, Main_Army)
        else:
            if self.time > 4*60:
                if self.structures(UnitTypeId.ROBOTICSFACILITY).ready:    
                    if self.units(UnitTypeId.OBSERVER).amount < 1 and self.already_pending(UnitTypeId.OBSERVER) == 0:
                        if self.can_afford(UnitTypeId.OBSERVER):
                            self.train(UnitTypeId.OBSERVER)
                            
        # Check for unpowered builds
        # TODO - use On_Unit_destroyed to check if the destroyed is a pylon, if so check if buildings are unpowered, if so build pylon

        
        # if a Warp Prism exists, send it to follow the main army
        if Warp_Prism:
            self.Warp_Prism_Follower(Warp_Prism, Main_Army)
            
        # Checking if there are 2 high templar to warp in Archons
        if self.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
            for templar in self.units(UnitTypeId.HIGHTEMPLAR).ready:
                templar(AbilityId.MORPH_ARCHON)

            
    async def on_unit_created(self, unit: Unit) -> None:
        await super(DragonBot, self).on_unit_created(unit)
        # Asign all units to the attacking role using ares unit role system
        typeid: UnitTypeId = unit.type_id
        # don't assign workers or buildings to the attacking role
        if typeid in ALL_STRUCTURES or typeid in WORKER_TYPES:
            return

        # add scouting role to Observer and Drop_Ship role warp prism else add attacking role
        if typeid == UnitTypeId.OBSERVER:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.SCOUTING)
        elif typeid == UnitTypeId.WARPPRISM:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.DROP_SHIP)
            unit.move(self.natural_expansion.towards(self.game_info.map_center, 1))
        else:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
            unit.attack(self.natural_expansion.towards(self.game_info.map_center, 1))
        

    async def on_building_construction_complete(self, building):
        await super(DragonBot, self).on_building_construction_complete(building)
        if building.type_id == UnitTypeId.NEXUS:
            self.nexus_creation_times[building.tag] = self.time  # update the creation time when a Nexus is created
            self.bases[building.tag] = building  # add the Nexus to the bases dictionary

        
        
        
    
    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        # TODO - use a threat detection if it on a building (or something like that)
        await super(DragonBot, self).on_unit_took_damage(unit, amount_damage_taken)
        if unit.type_id not in ALL_STRUCTURES:
            return
        
        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            self.mediator.cancel_structure(structure=unit)



    # Function to defend against worker rushes and cannon rushes
    # TODO  - WorkerKiteBack to worker defense
    async def defend_worker_cannon_rush(self, enemy_probes, enemy_cannons):
        # Select a worker
        if worker := self.mediator.select_worker(target_position=self.start_location):
            self.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)

        # Retrieve workers with a DEFENDING role
        defending_workers: Units = self.mediator.get_units_from_role(role=UnitRole.DEFENDING, unit_type=UnitTypeId.PROBE)

        # Assign workers to attack enemy probes and cannons
        for probe in enemy_probes:
            if defending_worker := defending_workers.closest_to(probe):
                await defending_worker.attack(probe)

        for cannon in enemy_cannons:
            if defending_worker := defending_workers.closest_to(cannon):
                await defending_worker.attack(cannon)
    
    def Control_Main_Army(self, Main_Army: Units, target: Point2) -> None:
        squads: list[UnitSquad] = self.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15.5)
        pos_of_main_squad: Point2 = self.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
        grid: np.ndarray = self.mediator.get_ground_grid

        for squad in squads:
            Main_Army_Actions = CombatManeuver()
            
            squad_position: Point2 = squad.squad_position
            units: list[Unit] = squad.squad_units
            squad_tags: set[int] = squad.tags

            all_close: Units = self.mediator.get_units_in_range(
                    start_points=[squad_position],
                    distances=25,
                    query_tree=UnitTreeQueryType.AllEnemy,
                    return_as_dict=False,
                )[0].filter(lambda u: not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES)            
            
            if all_close:
                melee: list[Unit] = [u for u in units if u.ground_range <= 3]
                ranged: list[Unit] = [u for u in units if u.ground_range > 3]
                target = cy_pick_enemy_target(all_close)                
                if ranged:
                    # keep units safe when they are low in shields else stutter back
                    for unit in ranged:
                        ranged_maneuver: CombatManeuver = CombatManeuver()
                        if unit.shield_health_percentage < 0.2:
                            ranged_maneuver.add(KeepUnitSafe(unit, grid))
                        else:
                            ranged_maneuver.add(StutterUnitBack(unit, target=target, grid=grid))
                        self.register_behavior(ranged_maneuver)
                else:
                     # Melee Actions
                    melee_maneuver: CombatManeuver = CombatManeuver()
                    melee_maneuver.add(AMoveGroup(group=melee, group_tags={u.tag for u in melee}, target=target.position))
                    self.register_behavior(melee_maneuver)
                                    
                
            else:
                # TODO - Fix grouping up without messing with Target, by iterating over all units checking distance to squad position if too far come close
                # # Check if the squad is already close to the target
                if pos_of_main_squad.distance_to(squad_position) > 0.2 and pos_of_main_squad.distance_to(target) > 0.1:
                    # Move towards the position of the main squad to regroup
                    Main_Army_Actions.add(PathGroupToTarget(start=squad_position, group=units, group_tags=squad_tags, target=pos_of_main_squad, grid=grid, sense_danger=False, success_at_distance=0.1))      
                else:
                   # Main_Army_Actions.add(PathGroupToTarget(start=squad_position, group=units, group_tags=squad_tags, target=target.position, grid=grid, sense_danger=True, success_at_distance=25))
                    Main_Army_Actions.add(AMoveGroup(group=units, group_tags=squad_tags, target=target.position))

                    
                self.register_behavior(Main_Army_Actions)



        
    
        
        

    # Function to Control Warp Prism
    def Warp_Prism_Follower(self, Warp_Prism: Units, Main_Army: Units)-> None:
        #declare a new group maneuver
        Warp_Prism_Actions: CombatManeuver = CombatManeuver()

        air_grid: np.ndarray = self.mediator.get_air_grid

        # Warp Prism to morph into Phase Mode if close by, transport mode to follow if no unit is being warped in 
        for prism in Warp_Prism:
            if Main_Army:
                distance_to_center = prism.distance_to(Main_Army.center)
                if distance_to_center < 15:
                    if prism.is_idle:
                        prism(AbilityId.MORPH_WARPPRISMPHASINGMODE)
                else:
                    not_ready_units = [unit for unit in self.units if not unit.is_ready and unit.distance_to(prism) < 6.5]
                    if prism.type_id == UnitTypeId.WARPPRISMPHASING and not not_ready_units:
                        prism(AbilityId.MORPH_WARPPRISMTRANSPORTMODE)

                    # Calculate a new target position that is 5 distance units away from Main_Army.center
                    direction_vector = (prism.position - Main_Army.center).normalized
                    new_target = Main_Army.center + direction_vector * 5
                    if prism.type_id == UnitTypeId.WARPPRISM:
                        Warp_Prism_Actions.add(
                            PathUnitToTarget(
                                unit=prism,
                                target=new_target,
                                grid=air_grid,
                                danger_distance=10
                            )
                        )
            else:
                Warp_Prism_Actions.add(PathUnitToTarget(unit=prism, target=self.natural_expansion, grid=air_grid, danger_distance=10))

        self.register_behavior(Warp_Prism_Actions)
    
    
    def Control_Scout(self, Scout: Units, Main_Army: Units)-> None:
        #declare a new group maneuver
        Scout_Actions: CombatManeuver = CombatManeuver()
        # get an air grid for the scout to path on
        air_grid: np.ndarray = self.mediator.get_air_grid

        # Create a list of targets for the scout
        targets = self.expansion_locations_list[:5] + [self.enemy_start_locations[0]]
        
        if self._commenced_attack:
        #follow the main army if it has commenced attack
            target = Main_Army.center.towards(self.attack_target, 20)
            for unit in Scout:
                Scout_Actions.add(
                    PathUnitToTarget(
                        unit=unit,
                        target=target,
                        grid=air_grid,                    )
                )
            self.register_behavior(Scout_Actions)

        else:
            # If there's no current target or the current target is None, set the first target
            if not hasattr(self, 'current_scout_target') or self.current_scout_target is None:
                if targets:
                    self.current_scout_target = targets[0]

            #Move scout to the main base to scout unless its in danger
            for unit in Scout:
                
                if unit.shield_percentage < 1:
                    Scout_Actions.add(
                    KeepUnitSafe(
                        unit=unit,
                        grid=air_grid
                    )
                    )
                else:
                    # If the unit is not in danger, move it to the current target
                    if unit.distance_to(self.current_scout_target) < 1:
                        # If the unit has reached its current target, move it to the next target
                        if self.current_scout_target is not None:
                            current_index = targets.index(self.current_scout_target)
                            if current_index + 1 < len(targets):
                                self.current_scout_target = targets[current_index + 1]
                            else:
                                # If the unit has visited all targets, set its current target to None
                                self.current_scout_target = None

                    if self.current_scout_target is not None:
                        Scout_Actions.add(
                            PathUnitToTarget(
                                unit=unit,
                                target=self.current_scout_target,
                                grid=air_grid,
                                danger_distance=10
                            )
                        )

            self.register_behavior(Scout_Actions)
    

    
    def threat_response(self, Main_Army: Units) -> None:
        ground_enemy_near_bases: dict[int, set[int]] = self.mediator.get_ground_enemy_near_bases
        flying_enemy_near_bases: dict[int, set[int]] = self.mediator.get_flying_enemy_near_bases
        
        if ground_enemy_near_bases or flying_enemy_near_bases:
            # Merge ground and air threats
            all_enemy = {}
            for key, value in ground_enemy_near_bases.items():
                all_enemy[key] = value.copy()
            for key, value in flying_enemy_near_bases.items():
                if key in all_enemy:
                    all_enemy[key].update(value)
                else:
                    all_enemy[key] = value.copy()
            # Retrieve actual enemy units and assess threat
            for _, enemy_tags in all_enemy.items():
                enemy_units: Units = self.enemy_units.tags_in(enemy_tags)
                own_forces: Units = Main_Army 
                self.assess_threat(enemy_units, own_forces)
                # If threat_level is needed, add logic here to process it
        
            #Checks for Early Game Threats
            if self.time < 5*60 and self.townhalls.first:
                # Initialize categories
                unit_categories = {'pylons': [], 'enemyWorkerUnits': [], 'cannons': []}
                
                # Retrieve and categorize units from tags
                for _, enemy_tags in ground_enemy_near_bases.items():
                    enemy_units = self.enemy_units.tags_in(enemy_tags)
                    for unit in enemy_units:
                        if unit.type_id == UnitTypeId.PYLON:
                            unit_categories['pylons'].append(unit)
                            print("Pylon Detected")
                        elif unit.type_id in [UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE]:
                            unit_categories['enemyWorkerUnits'].append(unit)
                        elif unit.type_id == UnitTypeId.PHOTONCANNON:
                            unit_categories['cannons'].append(unit)
                            print("Cannon Detected")
                
                # Check for specific units and act accordingly
                if unit_categories['pylons'] or len(unit_categories['enemyWorkerUnits']) >= 4 or unit_categories['cannons']:
                    self.build_order_runner.set_build_completed()
                    self.defend_worker_cannon_rush(unit_categories['enemyWorkerUnits'], unit_categories['cannons'])
                    self._used_rush_defense = True
                    print("Defending against worker/cannon rush")
                if self._used_rush_defense:
                    if not ground_enemy_near_bases:
                        self.register_behavior(SpawnController(self.cheese_defense_army))
                        self.register_behavior(ProductionController(self.cheese_defense_army, base_location=self.start_location))
                        print("Building cheese defense army")
            else:
                # If there's a threat and we have a main army, send the army to defend
                if self.assess_threat(enemy_units, own_forces) > 5 and Main_Army:
                    self._under_attack = True
                    # TODO - pass out num_units to the function to tell how many units in the threat
                    threat_position, num_units = cy_find_units_center_mass(enemy_units, 10.0)
                    threat_position = Point2(threat_position)
                    self.Control_Main_Army(Main_Army, threat_position)
                    print("Under Attack")
                    if not self.build_order_runner.build_completed:
                        self.build_order_runner.set_build_completed()
                        self.register_behavior(SpawnController(self.Standard_Army))
                        self.register_behavior(ProductionController(self.Standard_Army, base_location=self.start_location))
                elif not ground_enemy_near_bases and self._under_attack:
                    self.Control_Main_Army(Main_Army, self.natural_expansion.towards(self.game_info.map_center, 1))
                    print("No longer under attack")
                    self._under_attack = False
                else:
                    self._under_attack = False
    
    def assess_threat(self,enemy_units_near_bases, own_forces):
        threat_level = 0
        # Increase threat level based on number and type of enemy units
        """ 
        This section could be changed to instead of being an arbitrary 
        number to measure level of threat, 
        it could be their value in minerals and with 25% weight to gas"""
        for unit in enemy_units_near_bases:
            if unit.type_id in [UnitTypeId.MARINE, 
                             UnitTypeId.ZEALOT, 
                             UnitTypeId.ZERGLING, 
                             UnitTypeId.ADEPT,
                             UnitTypeId.STALKER,
                             UnitTypeId.ROACH,
                             UnitTypeId.REAPER,
                             UnitTypeId.MARAUDER,
                             UnitTypeId.SENTRY,
                             UnitTypeId.HYDRALISK,
                             UnitTypeId.BANELING,
                             UnitTypeId.HELLION,
                             UnitTypeId.HELLIONTANK,
                             UnitTypeId.HIGHTEMPLAR,
                             UnitTypeId.MUTALISK,
                             UnitTypeId.BANSHEE,
                             UnitTypeId.VIKING,
                             UnitTypeId.VIKINGFIGHTER,
                             UnitTypeId.PHOENIX,
                             UnitTypeId.ORACLE,
                             UnitTypeId.RAVEN,
                             UnitTypeId.GHOST
                            ]:
                threat_level += 2  # Combat units are a higher threat
            elif unit.type_id in [UnitTypeId.SIEGETANK, 
                                UnitTypeId.IMMORTAL, 
                                UnitTypeId.CYCLONE,
                                UnitTypeId.DISRUPTOR, 
                                UnitTypeId.COLOSSUS,
                                UnitTypeId.RAVAGER,
                                UnitTypeId.LURKER,
                                UnitTypeId.VOIDRAY,
                                UnitTypeId.CARRIER,
                                UnitTypeId.BATTLECRUISER,
                                UnitTypeId.TEMPEST,
                                UnitTypeId.BROODLORD,
                                UnitTypeId.ULTRALISK,
                                UnitTypeId.THOR,
                                UnitTypeId.SIEGETANKSIEGED,
                                UnitTypeId.LIBERATOR,
                                UnitTypeId.LIBERATORAG,
                                UnitTypeId.LURKERBURROWED,
                                UnitTypeId.DARKTEMPLAR,
                                UnitTypeId.ARCHON,
                                UnitTypeId.CORRUPTOR,
                                UnitTypeId.WIDOWMINE,
                                UnitTypeId.INFESTOR,
                                UnitTypeId.INFESTORBURROWED,
                                UnitTypeId.SWARMHOSTBURROWEDMP,
                                UnitTypeId.VIPER,
                                UnitTypeId.WIDOWMINEBURROWED
                                 ]:
                threat_level += 3  # Heavy units are an even higher threat
            else:
                threat_level += 1  # Other units contribute less to the threat
    
        # Adjust threat level based on proximity to key structures
        # Example: if enemy units are very close to a key structure, increase threat level
        # This part is left as an exercise for implementation
    
        # Adjust threat level based on your own defensive capabilities
        if own_forces.amount > enemy_units_near_bases.amount * 2:
            threat_level -= 2  # Having a significantly larger force reduces the threat level
    
        # Return the final threat level
        return threat_level
    
    
    
    async def on_end(self, game_result: Result) -> None:
        await super(DragonBot, self).on_end(game_result)
    

    