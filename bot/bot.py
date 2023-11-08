from sc2.bot_ai import BotAI, Race
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId


class CompetitiveBot(BotAI):
    NAME: str = "DragonBot"
    """This bot's name"""

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
        
        
    async def on_step(self, iteration: int):
        """
        This code runs continually throughout the game
        Populate this function with whatever your bot should do!
        """
        """
        print(f"this is my bot in iteration {iteration}") #print iteration
        """
        for loop_nexus in self.workers:
            if self.can_afford(UnitTypeId.PROBE):
                self.townhalls.ready.random.train(UnitTypeId.PROBE)
        
                # Add break statement here if you only want to train one
            else:
                # Can't afford probes anymore
                break
            
    async def on_end(self, result: Result):
        """
        This code runs once at the end of the game
        Do things here after the game ends
        """
        print("Game ended.")
