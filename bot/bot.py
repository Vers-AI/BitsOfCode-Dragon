from sc2.bot_ai import BotAI, Race
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2
from sc2.unit import Unit


class CompetitiveBot(BotAI):
    NAME: str = "DragonBot"
    """This bot's name"""
    #keep track of the last expansion index
    def __init__(self):
        super().__init__()
        self.last_expansion_index = -1

    RACE: Race = Race.Protoss
    """This bot's Starcraft 2 race.
    Options are:
        Race.Terran
        Race.Zerg
        Race.Protoss
        Race.Random
    """

    async def on_start(self):
        """
        This code runs once at the start of the game
        Do things here before the game starts
        """
        print("Game started")
        
    #Create a list of  all gold starting locations
    def _find_gold_expansions(self) -> list[Point2]:
            gold_mfs: list[Unit] = [
                mf for mf in self.mineral_field
                if mf.type_id in {UnitTypeId.RICHMINERALFIELD, UnitTypeId.RICHMINERALFIELD750}
            ]

            gold_expansions: list[Point2] = []

            # check if gold_mfs are on the map
            if len(gold_mfs) > 0:     
                for expansion in self.expansion_locations_list:
                    for gold_mf in gold_mfs:
                        if gold_mf.position.distance_to(expansion) < 12.5:
                            gold_expansions.append(expansion)
                            break
            # sort gold_expansions by proximity to the bot's starting location
            gold_expansions = sorted(
                gold_expansions,
                key=lambda expansion: self.start_location.distance_to(expansion)
            )


            return gold_expansions        
        
    async def on_step(self, iteration: int):
        """
        This code runs continually throughout the game
        """
        target_base_count = 4    # Number of total bases to expand to before stoping
        expansion_loctions_list = self._find_gold_expansions()   # Define expansion locations

        
        
        
    
        await self.distribute_workers() #puts idle workers to work

        nexus = self.townhalls.ready.random

        
                
        if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
            if self.can_afford(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus):
                nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                print("Chrono Boosted")

        # check if there are any more locations to expand to
        

        # if we have less than 4 bases and we have enough minerals and we are not building a nexus, build a nexus at gold expansions
        if self.townhalls.ready.amount < target_base_count and self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS):
            # if we have not expanded to all the locations
            if self.last_expansion_index + 1 < len(expansion_loctions_list):
            # increment the last expansion index
                self.last_expansion_index += 1
                print (self.last_expansion_index)
            await self.expand_now(location=expansion_loctions_list[self.last_expansion_index])
            print("Expanding")

        # Build a pylon if we are low on supply and less than supply cap of 200
        if self.supply_left < 2 and self.already_pending(UnitTypeId.PYLON) == 0 or self.supply_used > 15 and self.supply_left < 4 and self.already_pending(UnitTypeId.PYLON) < 2: 
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus.position.towards_with_random_angle(self.game_info.map_center, distance=5))
            return
       
        # train probes on nexuses that are undersaturated
        if nexus.assigned_harvesters < nexus.ideal_harvesters and nexus.is_idle:
            if self.supply_workers + self.already_pending(UnitTypeId.PROBE) <  self.townhalls.amount * 22 and nexus.is_idle:
                if self.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)
                    

            
        
        # build gateways
            if self.can_afford(UnitTypeId.GATEWAY) and self.structures(UnitTypeId.GATEWAY).amount < 8:
                pylon = self.structures(UnitTypeId.PYLON).ready
                if pylon.exists:
                    if self.can_afford(UnitTypeId.GATEWAY):
                        await self.build(UnitTypeId.GATEWAY, near=pylon.closest_to(nexus))
            #if we have no cybernetics core, build one
            if self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1 and self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.structures(UnitTypeId.GATEWAY).ready:
                pylon = self.structures(UnitTypeId.PYLON).ready
                if pylon.exists:
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon.closest_to(nexus))       

        # build gas
        for nexus in self.townhalls.ready:
            vgs = self.vespene_geyser.closer_than(15, nexus)
            for vg in vgs:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vg.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1, vg).exists:
                    worker.build(UnitTypeId.ASSIMILATOR, vg)
                    

        # Check for ready gateways and build zealots
        zealots = self.units(UnitTypeId.ZEALOT)
        gateways = self.structures(UnitTypeId.GATEWAY).ready.idle
        if gateways.exists and self.can_afford(UnitTypeId.ZEALOT):
            for gateway in gateways:
                if len(zealots) < 200:
                    gateway.train(UnitTypeId.ZEALOT)
        
        # if we hit supply cap attack        
        if self.supply_cap == 200:
            for zealot in zealots:
                zealot.attack(self.enemy_start_locations[0])

        
                            

        

        
            
    
    async def on_end(self, result: Result):
        """
        This code runs once at the end of the game
        Do things here after the game ends
        """
        print("Game ended.")
