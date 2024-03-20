import numpy as np
import math
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from sc2.bot_ai import BotAI
from typing import Dict, Iterable, List, Optional, Set

from sc2.ids.upgrade_id import UpgradeId


SPEEDMINING_DISTANCE = 1.8


# yield the intersection points of two circles at points p0, p1 with radii r0, r1
def get_intersections(p0: Point2, r0: float, p1: Point2, r1: float) -> Iterable[Point2]:
    p01 = p1 - p0
    d = np.linalg.norm(p01)
    if d == 0:
        return  # intersection is empty or infinite
    if d < abs(r0 - r1):
        return  # circles inside of each other
    if r0 + r1 < d:
        return  # circles too far apart
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r0 ** 2 - a ** 2)
    pm = p0 + (a / d) * p01
    po = (h / d) * np.array([p01.y, -p01.x])
    yield pm + po
    yield pm - po


# fix workers bumping into adjacent minerals by slightly shifting the move commands
def get_speedmining_positions(self : BotAI) -> Dict[Point2, Point2]:
    targets = dict()
    worker_radius = self.workers[0].radius
    expansions: Dict[Point2, Units] = self.expansion_locations_dict
    for base, resources in expansions.items():
        for resource in resources:
            mining_radius = resource.radius + worker_radius
            target = resource.position.towards(base, mining_radius)
            for resource2 in resources.closer_than(mining_radius, target):
                points = get_intersections(resource.position, mining_radius, resource2.position, resource2.radius + worker_radius)
                target = min(points, key=lambda p: p.distance_to(self.start_location), default=target)
            targets[resource.position] = target
    return targets


def micro_worker(self : BotAI) -> None:

    if self.townhalls.ready.amount <= 0:
        return

    for unit in self.workers:
        if unit.is_idle and self.unit_roles.get(unit.tag) != "expand":
            townhall = self.townhalls.ready.closest_to(unit)
            patch = self.mineral_field.closest_to(townhall)
            unit.gather(patch)
        if len(unit.orders) == 1: # speedmine
            target = None
            if unit.is_returning and not unit.is_carrying_vespene:
                target = self.townhalls.ready.closest_to(unit)
                move_target = target.position.towards(unit.position, target.radius + unit.radius)
            elif unit.is_gathering:
                target : Unit = self.resource_by_tag.get(unit.order_target)
                if target and not target.is_vespene_geyser and target.position in self.speedmining_positions.keys():
                    move_target = self.speedmining_positions[target.position]
            if target and not target.is_vespene_geyser and 2 * unit.radius < unit.distance_to(move_target) < SPEEDMINING_DISTANCE:
                unit.move(move_target)
                unit(AbilityId.SMART, target, True)


# Saturate assimilator
def handle_assimilator(self : BotAI, step: int):

    # update assimilator ages and dictionary
    for r in self.gas_buildings.ready:
        if not r.tag in self.assimilator_age.keys():
            self.assimilator_age[r.tag] = step
    to_remove = []
    for k in self.assimilator_age.keys():
        if self.gas_buildings.ready.find_by_tag(k) is None:
            to_remove.append(k)
    for i in to_remove:
        self.assimilator_age.pop(i, None)

    # handle workers
    for r in self.gas_buildings.ready:
        if self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0 and self.vespene < 48:
            if r.assigned_harvesters < r.ideal_harvesters and step - self.assimilator_age[r.tag] > 6: # last check because when it is finished there are 0 workers altough the one building goes to it instantly
                workers: Units = self.workers.closer_than(10, r)
                if workers:
                    for w in workers:
                        if not w.is_carrying_minerals and not w.is_carrying_vespene:
                            w.gather(r)
                            return
        if r.assigned_harvesters > r.ideal_harvesters or self.workers.amount <= 6 or self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) > 0 or self.vespene >= 48:
            workers: Units = self.workers.closer_than(6, r)
            if workers:
                for w in workers:
                    if w.is_carrying_vespene:
                        w.return_resource()
                        w.gather(self.resources.mineral_field.closest_to(w), queue=True)
                        return


def dispatch_workers(self : BotAI):
    # remove destroyed nexus from keys
    keys_to_delete = []
    for key in self.townhall_saturations.keys():
        if self.townhalls.ready.find_by_tag(key) is None:
            keys_to_delete.append(key)
    for i in keys_to_delete:
        del self.townhall_saturations[i]

    # add new nexus to keys and update its saturations
    maxes : Dict = {}
    for nexus in self.townhalls.ready:
        if not nexus.tag in self.townhall_saturations.keys():
            self.townhall_saturations[nexus.tag] = []
        if len(self.townhall_saturations[nexus.tag]) >= 40:
            self.townhall_saturations[nexus.tag].pop(0)
        self.townhall_saturations[nexus.tag].append(nexus.assigned_harvesters)
        maxes[nexus.tag] = max(self.townhall_saturations[nexus.tag])
    
    # dispatch workers somewhere else if Nexus has too much of them
    buffer = 3  # number of extra workers to allow before moving workers
    transfer_buffer = 5  # minimum difference in worker counts for a worker to be moved
    nexus_priority = sorted([key for key in maxes.keys() if key in self.nexus_creation_times], key=lambda x: self.nexus_creation_times[x])
    for key in nexus_priority:
        nexus1 = self.townhalls.ready.find_by_tag(key)
        if maxes[key] > nexus1.ideal_harvesters - buffer and (self.time <= 2 * 60 + 20 or self.time > self.worker_transfer_delay):
            for key2 in nexus_priority:
                if key2 == key:
                    continue
                nexus2 = self.townhalls.ready.find_by_tag(key2)
                if maxes[key2] + transfer_buffer < maxes[key]: # only move workers if the difference in worker counts is greater than transfer_buffer
                    for w in self.workers.closer_than(10, nexus1).gathering:
                        if self.mineral_field.closer_than(10, nexus1).find_by_tag(w.order_target) is not None:
                            w.gather(w.position.closest(self.mineral_field.closer_than(10, nexus2)))
                            maxes[key] -= 1
                            for i in range(len(self.townhall_saturations[key])):
                                self.townhall_saturations[key][i] -= 1
                            maxes[key2] += 1
                            for i in range(len(self.townhall_saturations[key2])):
                                self.townhall_saturations[key2][i] += 1
                            break


# distribute initial workers on mineral patches
def split_workers(self : BotAI) -> None:
    minerals = self.expansion_locations_dict[self.start_location].mineral_field.sorted_by_distance_to(self.start_location)
    self.close_minerals = {m.tag for m in minerals[0:4]}
    assigned: Set[int] = set()
    for i in range(self.workers.amount):
        patch = minerals[i % len(minerals)]
        if i < len(minerals):
            worker = self.workers.tags_not_in(assigned).closest_to(patch) # first, each patch gets one worker closest to it
        else:
            worker = self.workers.tags_not_in(assigned).furthest_to(patch) # the remaining workers get longer paths, this usually results in double stacking without having to spam orders
        worker.gather(patch)
        assigned.add(worker.tag)


def mine(self : BotAI, iteration):
    dispatch_workers(self)
    micro_worker(self)
    handle_assimilator(self, iteration)