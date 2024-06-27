import random


from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from sc2.constants import * 
from sc2.ids.unit_typeid import *
from sc2.ids.ability_id import *


class ManageThreats(object):
    def __init__(self, client, game_data):
        # usage:
        # self.defenseGroup = ManageThreats(self._client, self._game_data)

        # class data:
        self._client = client
        self._game_data = game_data
        self.threats = {}
        self.assignedUnitsTags = set()
        self.unassignedUnitsTags = set()

        # customizable parameters upon instance creation
        self.retreatLocations = None # retreat to the nearest location if hp percentage reached below "self.retreatWhenHp"
        self.retreatWhenHp = 0 # make a unit micro and retreat when this HP percentage is reached

        self.attackLocations = None # attack any of these locations if there are no threats

        self.treatThreatsAsAllies = False # if True, will mark threats as allies and tries to protect them instead 
        # self.defendRange = 5 # if a unit is in range within 5 of any of the threats, attack them

        self.clumpUpEnabled = False
        self.clumpDistance = 7 # not yet tested - sums up the distance to the center of the unit-ball, if too far away and not engaged with enemy: will make them clump up before engaging again

        self.maxAssignedPerUnit = 10 # the maximum number of units that can be assigned per enemy unit / threat

        self.leader = None # will be automatically assigned if "self.attackLocations" is not None
        
        availableModes = ["closest", "distributeEqually"] # todo: focus fire
        self.mode = "closest"

    def addThreat(self, enemies):
        if isinstance(enemies, Units):
            for unit in enemies:
                self.addThreat(unit)
        elif isinstance(enemies, Unit):            
            self.addThreat(enemies.tag)
        elif isinstance(enemies, int):
            if enemies not in self.threats:
                self.threats[enemies] = set()

    def clearThreats(self, threats=None):
        # accepts None, integer or iterable (with tags) as argument
        if threats is None:
            threats = self.threats
        elif isinstance(threats, int):
            threats = set([threats]) 

        # check for dead threats:
        for threat in threats:
            if threat in self.threats:
                unitsThatNowHaveNoTarget = self.threats.pop(threat) # remove and return the set
                self.assignedUnitsTags -= unitsThatNowHaveNoTarget
                self.unassignedUnitsTags |= unitsThatNowHaveNoTarget # append the tags to unassignedUnits

    def addDefense(self, myUnits):
        if isinstance(myUnits, Units):
            for unit in myUnits:
                self.addDefense(unit)
        elif isinstance(myUnits, Unit):            
            self.addDefense(myUnits.tag)
        elif isinstance(myUnits, int):
            if myUnits not in self.assignedUnitsTags:
                self.unassignedUnitsTags.add(myUnits)

    def removeDefense(self, myUnits):
        if isinstance(myUnits, Units):
            for unit in myUnits:
                self.removeDefense(unit)
        elif isinstance(myUnits, Unit):            
            self.removeDefense(myUnits.tag)
        elif isinstance(myUnits, int):
            self.assignedUnitsTags.discard(myUnits)
            self.unassignedUnitsTags.discard(myUnits)
            for key in self.threats.keys():
                self.threats[key].discard(myUnits)

    def setRetreatLocations(self, locations, removePreviousLocations=False):
        if self.retreatLocations is None or removePreviousLocations:
            self.retreatLocations = []
        if isinstance(locations, list):
            # we assume this is a list of points or units
            for location in locations:
                self.retreatLocations.append(location.position.to2)
        else:
            self.retreatLocations.append(location.position.to2)

    def unassignUnit(self, myUnit):
        for key, value in self.threats.items():
            # if myUnit.tag in value:
            value.discard(myUnit.tag)
                # break
        self.unassignedUnitsTags.add(myUnit.tag)
        self.assignedUnitsTags.discard(myUnit.tag)

    def getThreatTags(self):
        """Returns a set of unit tags that are considered as threats 
        
        Returns:
            set -- set of enemy unit tags
        """
        return set(self.threats.keys())

    def getMyUnitTags(self):
        """Returns a set of tags that are in this group
        
        Returns:
            set -- set of my unit tags
        """
        return self.assignedUnitsTags | self.unassignedUnitsTags

    def centerOfUnits(self, units):
        if isinstance(units, list):
            units = Units(units, self._game_data)
        assert isinstance(units, Units)
        assert units.exists
        if len(units) == 1:
            return units[0].position.to2
        coordX = sum([unit.position.x for unit in units]) / len(units)
        coordY = sum([unit.position.y for unit in units]) / len(units)
        return Point2((coordX, coordY))

    async def update(self, myUnitsFromState, enemyUnitsFromState, enemyStartLocations, iteration):
        # example usage: attackgroup1.update(self.units, self.known_enemy_units, self.enemy_start_locations, iteration)
        assignedUnits = myUnitsFromState.filter(lambda x:x.tag in self.assignedUnitsTags)
        unassignedUnits = myUnitsFromState.filter(lambda x:x.tag in self.unassignedUnitsTags)
        if not self.treatThreatsAsAllies:
            threats = enemyUnitsFromState.filter(lambda x:x.tag in self.threats)
        else:
            threats = myUnitsFromState.filter(lambda x:x.tag in self.threats)
        aliveThreatTags = {x.tag for x in threats}
        deadThreatTags = {k for k in self.threats.keys() if k not in aliveThreatTags}

        # check for dead threats:
        self.clearThreats(threats=deadThreatTags)
        
        # check for dead units:
        self.assignedUnitsTags = {x.tag for x in assignedUnits}
        self.unassignedUnitsTags = {x.tag for x in unassignedUnits}
        # update dead assigned units inside the dicts
        for key in self.threats.keys():
            values = self.threats[key]
            self.threats[key] = {x for x in values if x in self.assignedUnitsTags}

        # if self.treatThreatsAsAllies:
        #     print("supportgroup threat tags:", self.getThreatTags())
        #     print("supportgroup existing threats:", threats)
        #     for k,v in self.threats.items():
        #         print(k,v)
        #     print("supportgroup units unassigned:", unassignedUnits)
        #     print("supportgroup units assigned:", assignedUnits)

        canAttackAir = [QUEEN, CORRUPTOR]
        canAttackGround = [ROACH, BROODLORD, QUEEN, ZERGLING]

        recentlyAssigned = set()
        # assign unassigned units a threat # TODO: attackmove on the position or attack the unit?
        for unassignedUnit in unassignedUnits.filter(lambda x:x.health / x.health_max > self.retreatWhenHp):
            # if self.retreatLocations is not None and unassignedUnit.health / unassignedUnit.health_max < self.retreatWhenHp:
            #     continue
            # if len(unassignedUnit.orders) == 1 and unassignedUnit.orders[0].ability.id in [AbilityId.ATTACK]:
            #     continue
            if not threats.exists:
                if self.attackLocations is not None and unassignedUnit.is_idle:
                    await self.do(unassignedUnit.move(random.choice(self.attackLocations)))
            else:
                # filters threats if current looped unit can attack air (and enemy is flying) or can attack ground (and enemy is ground unit)
                # also checks if current unit is in threats at all and if the maxAssigned is not overstepped
                filteredThreats = threats.filter(lambda x: x.tag in self.threats and len(self.threats[x.tag]) < self.maxAssignedPerUnit and ((x.is_flying and unassignedUnit.type_id in canAttackAir) or (not x.is_flying and unassignedUnit.type_id in canAttackGround)))

                chosenTarget = None
                if not filteredThreats.exists and threats.exists:
                    chosenTarget = threats.random # for units like viper which cant attack, they will just amove there
                elif self.mode == "closest":
                    # TODO: only attack units that this unit can actually attack, like dont assign air if it cant shoot up
                    if filteredThreats.exists:
                        # only assign targets if there are any threats left
                        chosenTarget = filteredThreats.closest_to(unassignedUnit)
                elif self.mode == "distributeEqually":
                    threatTagWithLeastAssigned = min([[x, len(y)] for x, y in self.threats.items()], key=lambda q: q[1])
                    # if self.treatThreatsAsAllies:
                    #     print("supportgroup least assigned", threatTagWithLeastAssigned)
                    # if self.treatThreatsAsAllies:
                    #     print("supportgroup filtered threats", filteredThreats)
                    if filteredThreats.exists:
                        # only assign target if there are any threats remaining that have no assigned allied units
                        chosenTarget = filteredThreats.find_by_tag(threatTagWithLeastAssigned[0])
                        # if self.treatThreatsAsAllies:
                        #     print("supportgroup chosen target", chosenTarget)
                else:
                    chosenTarget = random.choice(threats)

                if chosenTarget is not None:
                    # add unit to assigned target
                    self.unassignedUnitsTags.discard(unassignedUnit.tag)
                    self.assignedUnitsTags.add(unassignedUnit.tag)
                    self.threats[chosenTarget.tag].add(unassignedUnit.tag)
                    recentlyAssigned.add(unassignedUnit.tag)
                    # threats.remove(chosenTarget)
                    unassignedUnits.remove(unassignedUnit)
                    assignedUnits.append(unassignedUnit)
                    if unassignedUnit.distance_to(chosenTarget) > 3:
                        # amove towards target when we want to help allied units
                        await self.do(unassignedUnit.attack(chosenTarget.position))
                    break # iterating over changing list
        
        # if self.treatThreatsAsAllies and len(recentlyAssigned) > 0:
        #     print("supportgroup recently assigned", recentlyAssigned)

        clumpedUnits = False        
        if assignedUnits.exists and self.clumpUpEnabled:
            amountUnitsInDanger = [threats.closer_than(10, x).exists for x in assignedUnits].count(True)
            # print("wanting to clump up")
            if amountUnitsInDanger < assignedUnits.amount / 5: # if only 10% are in danger, then its worth the risk to clump up again
                # make all units clump up more until trying to push / attack again
                center = self.centerOfUnits(assignedUnits)
                distanceSum = 0
                for u in assignedUnits:
                    distanceSum += u.distance_to(center)
                distanceSum /= assignedUnits.amount

                if distanceSum > self.clumpDistance:
                    clumpedUnits = True
                    for unit in assignedUnits:
                        await self.do(unit.attack(center))

        if not clumpedUnits:
            for unit in assignedUnits:
                if unit.tag in recentlyAssigned:
                    continue  
                # # move close to leader if he exists and if unit is far from leader
                # if self.attackLocations is not None \
                #     and leader is not None \
                #     and unit.tag != leader.tag \
                #     and (unit.is_idle or len(unit.orders) == 1 and unit.orders[0].ability.id in [AbilityId.MOVE]) \
                #     and unit.distance_to(leader) > self.clumpDistance:
                #     await self.do(unit.attack(leader.position))

                # if unit is idle or move commanding, move directly to target, if close to target, amove
                if unit.is_idle or len(unit.orders) == 1 and unit.orders[0].ability.id in [AbilityId.MOVE]:
                    assignedTargetTag = next((k for k,v in self.threats.items() if unit.tag in v), None)
                    if assignedTargetTag is not None:
                        assignedTarget = threats.find_by_tag(assignedTargetTag)                    
                        if assignedTarget is None:
                            self.unassignUnit(unit)
                        elif assignedTarget.distance_to(unit) <= 13 or threats.filter(lambda x: x.distance_to(unit) < 13).exists:
                            await self.do(unit.attack(assignedTarget.position))
                        elif assignedTarget.distance_to(unit) > 13 and unit.is_idle and unit.tag != assignedTarget.tag:
                            await self.do(unit.attack(unit.position.to2.towards(assignedTarget.position.to2, 20))) # move follow command
                    else:
                        self.unassignUnit(unit)
            # # if unit.is_idle:
            # #     self.unassignUnit(unit)
            # elif len(unit.orders) == 1 and unit.orders[0].ability.id in [AbilityId.MOVE]:
            #     # make it amove again
            #     for key, value in self.threats.items():
            #         if unit.tag in value:
            #             assignedTargetTag = key
            #             assignedTarget = threats.find_by_tag(assignedTargetTag)
            #             if assignedTarget is None:
            #                 continue
            #                 # self.unassignUnit(unit)
            #             elif assignedTarget.distance_to(unit) <= 13:
            #                 await self.do(unit.attack(assignedTarget.position))
            #                 break
            #             # elif assignedTarget.distance_to(unit) > 13:
            #             #     await self.do(unit.move(assignedTarget))

        # move to retreatLocation when there are no threats or when a unit is low hp
        if self.retreatLocations is not None and not threats.exists and iteration % 20 == 0:
            for unit in unassignedUnits.idle:
                closestRetreatLocation = unit.position.to2.closest(self.retreatLocations)
                if unit.distance_to(closestRetreatLocation) > 10:
                    await self.do(unit.move(closestRetreatLocation))

        # move when low hp
        elif self.retreatLocations is not None and self.retreatWhenHp != 0:
            for unit in (assignedUnits | unassignedUnits).filter(lambda x:x.health / x.health_max < self.retreatWhenHp):
                closestRetreatLocation = unit.position.to2.closest(self.retreatLocations)
                if unit.distance_to(closestRetreatLocation) > 6:
                    await self.do(unit.move(closestRetreatLocation))

    async def do(self, action):
        r = await self._client.actions(action, game_data=self._game_data)
        return r