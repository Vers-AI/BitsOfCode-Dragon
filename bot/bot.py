from typing import Optional

from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import AMoveGroup


from itertools import chain


from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2




from bot.speedmining import get_speedmining_positions
from bot.speedmining import split_workers
from bot.speedmining import mine


class DragonBot(AresBot):
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

    
    
    async def on_start(self) -> None:
        await super(DragonBot, self).on_start()
        
        print("Game started")
        self.client.game_step = 2    
        self.speedmining_positions = get_speedmining_positions(self)
        split_workers(self)   

        self.nexus_creation_times = {nexus.tag: self.time for nexus in self.townhalls.ready}  # tracks the creation time of Nexus

        self.target = self.enemy_start_locations[0]  # set the target to the enemy start location
        
        print("Build Chosen:",self.build_order_runner.chosen_opening)
    
    async def on_step(self, iteration: int) -> None:
        await super(DragonBot, self).on_step(iteration)

        self.resource_by_tag = {unit.tag: unit for unit in chain(self.mineral_field, self.gas_buildings)}

        mine(self, iteration)

        # retrieve all attacking units
        Main_Army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)

        
        #check if the B2GM_Starting_Build is completed, if so send all the units to the enemy base

        if self.build_order_runner.chosen_opening == "B2GM_Starting_Build" and self.build_order_runner.build_completed:             
            self.Control_Main_Army(Main_Army, self.target)
            
    async def on_unit_created(self, unit: Unit) -> None:
        await super(DragonBot, self).on_unit_created(unit)
        # Asign all units to the attacking role using ares unit role system
        typeid: UnitTypeId = unit.type_id
        # don't assign workers or buildings to the attacking role
        if typeid in ALL_STRUCTURES or typeid in WORKER_TYPES:
            return

        # assign all other units to the attacking role by defualt
        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

         

    async def on_building_construction_complete(self, building):
        await super(DragonBot, self).on_building_construction_complete(building)
        if building.type_id == UnitTypeId.NEXUS:
            self.nexus_creation_times[building.tag] = self.time  # update the creation time when a Nexus is created

    def Control_Main_Army(self, Main_Army: Units, target: Point2)-> None:
        #declare a new group manvuever
        Main_Army_Actions: CombatManeuver = CombatManeuver()

        #Add amove to the main army
        Main_Army_Actions.add(
            AMoveGroup(
                group=Main_Army,
                group_tags={unit.tag for unit in Main_Army},
                target=self.target,
            )
        )   
        self.register_behavior(Main_Army_Actions)

    async def on_end(self, game_result: Result) -> None:
        await super(DragonBot, self).on_end(game_result)
    

    #
    # async def on_building_construction_complete(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_building_construction_complete(unit)
    #
    #     # custom on_building_construction_complete logic here ...
    #
    
    #
    # async def on_unit_destroyed(self, unit_tag: int) -> None:
    #     await super(MyBot, self).on_unit_destroyed(unit_tag)
    #
    #     # custom on_unit_destroyed logic here ...
    #
    # async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
    #     await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)
    #
    #     # custom on_unit_took_damage logic here ...

    