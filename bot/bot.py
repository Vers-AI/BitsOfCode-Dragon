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
            gold_expansions.sort(key=lambda x: x.distance_to(self.start_location))

            return gold_expansions        
        
    async def on_step(self, iteration: int):
        """
        This code runs continually throughout the game
        """
        target_base_count = 4    # Number of total bases to expand to before stoping
        expansion_loctions_list = self._find_gold_expansions()   # Define expansion locations

        await self.distribute_workers()  # Distribute workers to mine minerals and gas
        nexus = self.townhalls.ready.random
                
        

        # Build a pylon if we are low on supply and less than supply cap of 200
        if self.supply_left < 2 and self.already_pending(UnitTypeId.PYLON) == 0 or self.supply_used > 15 and self.supply_left < 6 and self.already_pending(UnitTypeId.PYLON) < 2: 
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus.position.towards(self.game_info.map_center, 5))
            return
       
        # train probes on nexuses that are undersaturated
        if nexus.assigned_harvesters < nexus.ideal_harvesters and nexus.is_idle:
            if self.supply_workers + self.already_pending(UnitTypeId.PROBE) <  self.townhalls.amount * 22 and nexus.is_idle:
                if self.can_afford(UnitTypeId.PROBE):
                    nexus.train(UnitTypeId.PROBE)
                    
        # if we have less than 4 bases and we have enough minerals and we are not building a nexus, build a nexus at gold expansions
        if self.townhalls.ready.amount < target_base_count and self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS):
            # if we have not expanded to all the locations
            if self.last_expansion_index + 1 < len(expansion_loctions_list):
            # increment the last expansion index
                self.last_expansion_index += 1
                print (self.last_expansion_index)
            await self.expand_now(location=expansion_loctions_list[self.last_expansion_index])
            print("Expanding")
            
        
        # build 4 gateways and cybernetics core once pylon is complete and keep building up to 8 warpgates
        if self.structures(UnitTypeId.PYLON).ready:
            pylon = self.structures(UnitTypeId.PYLON).ready.random
            if self.structures(UnitTypeId.GATEWAY).amount < 4 and self.can_afford(UnitTypeId.GATEWAY) and self.structures(UnitTypeId.WARPGATE).ready.amount < 8:
                await self.build(UnitTypeId.GATEWAY, near=pylon.position.towards(self.game_info.map_center, 5))
        if self.structures(UnitTypeId.CYBERNETICSCORE).amount < 1 and self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0 and self.structures(UnitTypeId.GATEWAY).ready.amount >= 1:
            await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon.position.towards(nexus.position, 5))

       
        # build 1 gas 
        if self.structures(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.ASSIMILATOR) == 0 and self.structures(UnitTypeId.ASSIMILATOR).amount < 1:
            for nexus in self.townhalls.ready:
                vgs = self.vespene_geyser.closer_than(15, nexus)
                for vg in vgs:
                    if not self.can_afford(UnitTypeId.ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vg.position)
                    if worker is None:
                        break
                    if not self.gas_buildings or not self.gas_buildings.closer_than(1, vg):
                        worker.build_gas(vg)
                        worker.stop(queue=True)    
        
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


        # If there are warpates and warpin is available, warp in zealots from warpgates near a pylon else train zealots from gateways
        if self.structures(UnitTypeId.WARPGATE).ready:
            for warpgate in self.structures(UnitTypeId.WARPGATE).ready.idle:
                abililities = await self.get_available_abilities(warpgate)
                if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left > 0 and AbilityId.WARPGATETRAIN_ZEALOT in abililities and self.townhalls.amount >= 3:
                    position = pylon.position.to2.random_on_distance(4)
                    warpgate.warp_in(UnitTypeId.ZEALOT, position)
        else:
            for gateway in self.structures(UnitTypeId.GATEWAY).ready.idle:
                if self.can_afford(UnitTypeId.ZEALOT) and self.townhalls.amount >= 2:
                    gateway.train(UnitTypeId.ZEALOT)
        
        
        # if we hit supply cap attack
        zealots = self.units(UnitTypeId.ZEALOT)
        if self.supply_cap == 200:
            for zealot in zealots:
                zealot.attack(self.enemy_start_locations[0])

        #Chrono boost nexus if cybernetics core is not idle       
        if not self.structures(UnitTypeId.CYBERNETICSCORE).ready:
            if not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not nexus.is_idle:
                if nexus.energy >= 50:
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
        else:
            ccore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
            if not ccore.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and not ccore.is_idle:
                if nexus.energy >= 50:
                    nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, ccore)
        


        

        
            
    
    async def on_end(self, result: Result):
        """
        This code runs once at the end of the game
        Do things here after the game ends
        """
        print("Game ended.")
