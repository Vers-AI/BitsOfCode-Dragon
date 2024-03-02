""" 
This bot is a bot from episode 2 of Bits of Code. 
It is a simple bot that expands to gold bases and builds zealots trying to reach Max supply by 5:42 in game time. 
use the map Prion Terrace. 
"""

from loguru import logger

from sc2.bot_ai import BotAI, Race
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2
from sc2.unit import Unit
from sc2 import position
from sc2.constants import UnitTypeId


class CompetitiveBot(BotAI):
    NAME: str = "DragonBot"
    """This bot's name"""
    #keep track of the last expansion index
    def __init__(self):
        super().__init__()
        self.last_expansion_index = -1
        self.first_nexus = None

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
            gold_expansions.sort(key=lambda x: x.distance_to(self.start_location))

            return gold_expansions        
    
    async def warp_new_units(self, pylon):
        #warp in zealots from warpgates near a pylon if below supply cap
        for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle:
            abililities = await self.get_available_abilities(warpgate)
            if self.can_afford(UnitTypeId.ZEALOT) and AbilityId.WARPGATETRAIN_ZEALOT in abililities and self.supply_used < 200:
                position = pylon.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_ZEALOT, position, placement_step=1)
                if placement is None:
                    # return ActionResult.CantFindPlacementLocation
                    logger.info("can't place")
                    return
                warpgate.warp_in(UnitTypeId.ZEALOT, placement)
    
    
    async def on_step(self, iteration: int):
        """
        This code runs continually throughout the game
        """
        target_base_count = 6    # Number of total bases to expand to before stoping
        expansion_loctions_list = self._find_gold_expansions()   # Define expansion locations

        await self.distribute_workers()  # Distribute workers to mine minerals and gas
        nexus = self.townhalls.ready.random
        closest = self.start_location
        
    
      
        

        # Build a pylon if we are low on supply up until 4 bases after 4 bases build pylons until supply cap is 200
        if self.supply_left <= 2 and self.already_pending(UnitTypeId.PYLON) == 0 and self.structures(UnitTypeId.PYLON).amount < 1: 
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus.position.towards(self.game_info.map_center, 10))
                print(self.time_formatted, "building pylon")
        # When we hit 4 gateways, build an extra Pylon if we have less than 2
        elif self.structures(UnitTypeId.PYLON).amount  < 2 and self.townhalls.amount == 10:
            if self.can_afford(UnitTypeId.PYLON) and self.already_pending(UnitTypeId.PYLON) == 0:
                await self.build(UnitTypeId.PYLON, near=closest.towards_with_random_angle(self.game_info.map_center, 15))
                print(self.time_formatted, "building pylon - 2")
        # After 13 warpgates, build pylons until supply cap is 200 and we are at 6 bases - pylon explosion
        elif self.structures(UnitTypeId.GATEWAY).amount + self.structures(UnitTypeId.WARPGATE).amount >= 11 and self.supply_cap < 200:
            direction = Point2((0, 3))  
            if self.can_afford(UnitTypeId.PYLON) and self.structures(UnitTypeId.PYLON).amount + self.already_pending(UnitTypeId.PYLON) < 14:
                await self.build(UnitTypeId.PYLON, near=closest.position + direction.towards(self.game_info.map_center, 10))

       
        # train probes on nexuses that are undersaturated
        # if nexus.assigned_harvesters < nexus.ideal_harvesters and nexus.is_idle:
        if self.supply_workers + self.already_pending(UnitTypeId.PROBE) <  self.townhalls.amount * 22 and nexus.is_idle:
            if self.can_afford(UnitTypeId.PROBE):
                nexus.train(UnitTypeId.PROBE)
        
                    
        # Building Probes
        if self.supply_used < 200 and self.structures(UnitTypeId.PYLON).amount == 14: # quick build to 200 supply with probes
            for nexus in self.townhalls.ready:
                if self.can_afford(UnitTypeId.PROBE) and nexus.is_idle:
                    nexus.train(UnitTypeId.PROBE)
        
        # move a probe 275 minerals in advance to each in advance to the next base location to cut down on travel time
        """if self.last_expansion_index < 3:
            if self.minerals > 275: 
                location = expansion_loctions_list[self.last_expansion_index] 
                worker = self.workers.closest_to(location)
                worker.move(location, queue=False)"""
            
                    
        # if we have less than target base count and build 5 nexuses at gold bases and then build at other locations
        if self.last_expansion_index < 3 and self.townhalls.amount < target_base_count:
            if self.can_afford(UnitTypeId.NEXUS): 
                    self.last_expansion_index += 1
                    await self.expand_now(location=expansion_loctions_list[self.last_expansion_index])
                    print(self.time_formatted, "expanding to gold bases", self.last_expansion_index, "of", len(expansion_loctions_list), "total current bases=", self.townhalls.amount)
        if self.last_expansion_index == 3 and self.townhalls.amount < target_base_count:
            if self.can_afford(UnitTypeId.NEXUS):
                await self.expand_now()
                print(self.time_formatted, "expanding to",self.last_expansion_index,"th location, total current bases=", self.townhalls.amount)
                
            
            
        
        # Key buildings: after 4 nexuses are built, build gateways and cybernetics core once pylon is complete and keep building up to 13 warpgates after warpgate researched
        if self.structures(UnitTypeId.PYLON).ready:
            # Select a pylon
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            # Get the positions around the pylon
            # Get the positions around the pylon
            positions = [Point2((pylon.position.x + x, pylon.position.y + y)) for x in range(-6, 7) for y in range(-6, 7)]
            # Sort the positions by distance to the pylon
            positions.sort(key=lambda pos: pylon.position.distance_to(pos))
            if self.townhalls.amount >= 4 and self.structures(UnitTypeId.GATEWAY).amount + self.structures(UnitTypeId.WARPGATE).amount < 1 and self.already_pending(UnitTypeId.GATEWAY) == 0 and not self.structures(UnitTypeId.CYBERNETICSCORE):
                for pos in positions:
                    # Check if the position is valid for building
                    if await self.can_place_single(UnitTypeId.GATEWAY, pos):
                    # If the position is valid, build the gateway
                        if self.can_afford(UnitTypeId.GATEWAY):
                            await self.build(UnitTypeId.GATEWAY, near=pos)
            elif self.structures(UnitTypeId.WARPGATE).amount + self.structures(UnitTypeId.GATEWAY).amount < 11 and self.townhalls.amount == 6:
                for pos in positions:
                # Check if the position is valid for building
                    if await self.can_place_single(UnitTypeId.GATEWAY, pos):
                    # If the position is valid, build the gateway
                        if self.can_afford(UnitTypeId.GATEWAY):
                            await self.build(UnitTypeId.GATEWAY, near=pos)
                            break
            if self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1 and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0 and self.structures(UnitTypeId.ASSIMILATOR):
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.structures(UnitTypeId.GATEWAY).ready:
                    for pos in positions:
                        # Check if the position is valid for building
                        if await self.can_place_single(UnitTypeId.CYBERNETICSCORE, pos):
                            # If the position is valid, build the Cybernetics Core
                            await self.build(UnitTypeId.CYBERNETICSCORE, near=pos)
                            print(self.time_formatted, "building cybernetics core")
                            break

       
        # build 1 gas near the starting nexus
        if self.townhalls.amount >= 4:
            if self.structures(UnitTypeId.ASSIMILATOR).amount + self.already_pending(UnitTypeId.ASSIMILATOR) < 1:
                if self.can_afford(UnitTypeId.ASSIMILATOR):
                    vgs = self.vespene_geyser.closer_than(15, closest)
                    if vgs:
                        worker = self.select_build_worker(vgs.first.position)
                        if worker is None:
                            return
                        worker.build(UnitTypeId.ASSIMILATOR, vgs.first)
        
        # saturate the gas 
        for assimilator in self.gas_buildings:
            if assimilator.assigned_harvesters < assimilator.ideal_harvesters:
                workers = self.workers.closer_than(10, assimilator)
                if workers:
                    workers.random.gather(assimilator)   
        # put idle probes to work
        for probe in self.workers.idle:
            probe.gather(self.mineral_field.closest_to(nexus))

        # Research Warp Gate if Cybernetics Core is complete
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.can_afford(UpgradeId.WARPGATERESEARCH) and self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0:
            ccore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
            if ccore.is_idle:
                ccore.research(UpgradeId.WARPGATERESEARCH)
                        
        # Morph to warp gates when warp gate research is complete
        if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0 and self.structures(UnitTypeId.GATEWAY).ready:
            for gateway in self.structures(UnitTypeId.GATEWAY).ready.idle:
                gateway(AbilityId.MORPH_WARPGATE)

        # warp in zealots from warpgates near a pylon if there are 6 warpgates else build zealots
        if self.structures(UnitTypeId.WARPGATE).ready:
            if self.structures(UnitTypeId.WARPGATE).amount + self.structures(UnitTypeId.GATEWAY).amount >= 11:
                await self.warp_new_units(pylon)
        elif not self.structures(UnitTypeId.WARPGATE).ready: 
            if self.structures(UnitTypeId.GATEWAY).amount + self.already_pending(UnitTypeId.GATEWAY)>= 10:
                for gateway in self.structures(UnitTypeId.GATEWAY).ready.idle:
                    if self.can_afford(UnitTypeId.ZEALOT):
                        gateway.train(UnitTypeId.ZEALOT)
        
        
        
        
        # if we hit supply cap surrender if not move zealtots to the center of the map
        zealots = self.units(UnitTypeId.ZEALOT)
        if self.supply_used == 200:
            print(self.time_formatted, "supply cap reached with:", self.structures(UnitTypeId.WARPGATE).ready.amount, "warpgates","+", self.structures(UnitTypeId.PYLON).ready.amount, "pylons", "and", self.townhalls.amount, "nexuses", "+", self.units(UnitTypeId.ZEALOT).amount, "zealots","and", self.workers.amount, "probes")
            await self.chat_send("Suppy Cap Reached at:" + self.time_formatted)
            self.client.leave
        else:
            for zealot in zealots:
                zealot(AbilityId.ATTACK, nexus.position.towards(self.game_info.map_center, 5))
        

        # Chrono boost nexus if cybernetics core is not idle and warpgates WARPGATETRAIN_ZEALOT is not available         
        if self.structures(UnitTypeId.WARPGATE).amount + self.structures(UnitTypeId.GATEWAY).amount >= 11:
            warpgates = self.structures(UnitTypeId.WARPGATE).ready
            for warpgate in warpgates:
                abilities = await self.get_available_abilities(warpgate)
                if not AbilityId.WARPGATETRAIN_ZEALOT in abilities:
                    if not warpgate.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        if nexus.energy >= 50:
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, warpgate)
        elif self.structures(UnitTypeId.CYBERNETICSCORE).ready:
            ccore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
            if not ccore.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not ccore.is_idle:
                if nexus.energy >= 50:
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, ccore)
        else:
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not nexus.is_idle:
                if nexus.energy >= 50 and self.time > 17:
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
        
    
        


        

        
            
    
    async def on_end(self, result: Result):
        """
        This code runs once at the end of the game
        Do things here after the game ends
        """
        print("Game ended.")
