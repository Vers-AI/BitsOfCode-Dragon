"""Composition Belief — Probability-weighted enemy army tracking.

Purpose: Replace binary age cutoff (age < 30s → keep, age >= 30s → drop) with
         smooth exponential decay. Each enemy unit has P(exists | age, type) that
         degrades over time, so stale intel degrades gradually instead of a cliff.

Key Decisions: Conjugate exponential decay only (O(1) per unit per frame).
               Structure-based priors add "expected but unseen" units from tech trees.
               on_unit_destroyed immediately sets P=0.0 (ground truth, no decay).

Limitations: No per-unit tracking through fog-of-war position changes.
              Structure priors are static lookup tables, not learned from data.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit

from bot.constants import UNIT_HALF_LIFE, DEFAULT_HALF_LIFE, STRUCTURE_SEEN_UNIT_PRIOR

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


@dataclass
class WeightedUnit:
    """A unit with an associated confidence that it still exists."""

    unit: Unit
    confidence: float  # P(unit exists | age, type), 0.0-1.0

    @property
    def weight(self) -> float:
        """Alias for confidence — used in threat_value and army_value calculations."""
        return self.confidence


@dataclass
class CompositionBelief:
    """Tracks P(unit exists | age, unit_type) for every known enemy unit.

    Updated each frame with visible units (snap to P=1.0) and decayed units
    (exponential decay from last-seen time). Destroyed units are removed.
    Also tracks structure-based priors for expected-but-unseen units.

    This object is mutable during update() and frozen into BeliefState after.
    """

    # tag → last-seen game time (seconds)
    _last_seen: dict[int, float] = field(default_factory=dict)
    # tag → P(exists), recalculated each frame from decay
    _confidence: dict[int, float] = field(default_factory=dict)
    # tag → UnitTypeId, for half-life lookup
    _unit_types: dict[int, UnitTypeId] = field(default_factory=dict)
    # tag → unit object from this frame (latest visible or cached)
    _units: dict[int, Unit] = field(default_factory=dict)
    # Structure types we've seen → time first seen
    _seen_structures: dict[UnitTypeId, float] = field(default_factory=dict)
    # Expected-but-unseen units from structure priors: type → (P, source_structure)
    _expected_units: dict[UnitTypeId, tuple[float, UnitTypeId]] = field(default_factory=dict)
    # Destroyed tags — removed from belief, never re-added
    _destroyed: set[int] = field(default_factory=set)
    # Game time of last update
    _last_update_time: float = 0.0

    def update(
        self,
        cached_army,
        visible_structures,
        game_time: float,
        destroyed_tags: set[int],
        unit_last_seen: dict[int, float],
    ) -> None:
        """Update belief state with current observations.

        Args:
            cached_army: Units from ARES UnitCacheManager (may include ghosts)
            visible_structures: Currently visible enemy structures
            game_time: Current game time in seconds
            destroyed_tags: Tags confirmed destroyed this game (from on_unit_destroyed)
            unit_last_seen: External tracking of when each unit tag was last visible
        """
        self._last_update_time = game_time

        # 1. Remove destroyed units (ground truth — they don't exist)
        for tag in destroyed_tags:
            self._destroyed.add(tag)
            self._confidence.pop(tag, None)
            self._last_seen.pop(tag, None)
            self._unit_types.pop(tag, None)
            self._units.pop(tag, None)

        # 2. Update from cached army (may include stale ghost units)
        for unit in cached_army:
            if unit.tag in self._destroyed:
                continue
            self._unit_types[unit.tag] = unit.type_id
            self._units[unit.tag] = unit

            # If unit is currently visible (age ≈ 0), snap to P=1.0
            if unit.age < 1.0:
                self._last_seen[unit.tag] = game_time
                self._confidence[unit.tag] = 1.0
            else:
                # Use external last-seen time if available, otherwise game_time - age
                if unit.tag in unit_last_seen:
                    self._last_seen[unit.tag] = unit_last_seen[unit.tag]
                elif unit.tag not in self._last_seen:
                    # First time seeing this ghost unit — estimate last-seen from age
                    self._last_seen[unit.tag] = max(0.0, game_time - unit.age)

                # Calculate P(exists) from exponential decay
                age_since_seen = game_time - self._last_seen[unit.tag]
                half_life = UNIT_HALF_LIFE.get(unit.type_id, DEFAULT_HALF_LIFE)
                self._confidence[unit.tag] = 0.5 ** (age_since_seen / half_life)

        # 3. Remove stale entries — units not in cached_army and not destroyed
        #    This handles units that fell out of the ARES cache entirely
        current_tags = {u.tag for u in cached_army}
        stale_tags = set(self._confidence.keys()) - current_tags - self._destroyed
        for tag in stale_tags:
            # Decay them one more time before removing
            if tag in self._last_seen and tag in self._unit_types:
                age_since_seen = game_time - self._last_seen[tag]
                half_life = UNIT_HALF_LIFE.get(self._unit_types[tag], DEFAULT_HALF_LIFE)
                prob = 0.5 ** (age_since_seen / half_life)
                # If very unlikely to exist, remove
                if prob < 0.01:
                    self._confidence.pop(tag, None)
                    self._last_seen.pop(tag, None)
                    self._unit_types.pop(tag, None)
                    self._units.pop(tag, None)

        # 4. Update structure-based priors from visible structures
        self._update_structure_priors(visible_structures, game_time)

    def _update_structure_priors(self, visible_structures, game_time: float) -> None:
        """Update expected-but-unseen units based on structures we've seen."""
        # Track which structure types we've seen
        for structure in visible_structures:
            if structure.type_id in STRUCTURE_SEEN_UNIT_PRIOR:
                if structure.type_id not in self._seen_structures:
                    self._seen_structures[structure.type_id] = game_time

        # Compute expected units from all seen structures
        self._expected_units.clear()
        for struct_type, seen_time in self._seen_structures.items():
            if struct_type not in STRUCTURE_SEEN_UNIT_PRIOR:
                continue
            time_since_seen = game_time - seen_time
            # Prior strength decays over time — we should have seen the unit by now
            # if it was going to be produced. After 120s, priors are negligible.
            prior_decay = max(0.0, 1.0 - time_since_seen / 120.0)
            if prior_decay <= 0.0:
                continue
            for unit_type, base_prob in STRUCTURE_SEEN_UNIT_PRIOR[struct_type].items():
                # Only add if we haven't seen any of this unit type yet
                if not any(
                    t == unit_type for t in self._unit_types.values()
                ):
                    existing_prob = self._expected_units.get(unit_type, (0.0, struct_type))[0]
                    # Combine: take max probability across all seen structures
                    combined_prob = max(existing_prob, base_prob * prior_decay)
                    self._expected_units[unit_type] = (combined_prob, struct_type)

    def snapshot(self) -> "CompositionBelief":
        """Create an independent copy for BeliefState consumption.

        Copies all scalar dicts (tags→floats, tags→type_ids, sets) so the
        snapshot is immutable from the consumer's perspective. Shares Unit
        object references — they're read-only game engine snapshots that are
        refreshed each frame, so sharing is safe and avoids unpicklable
        deep-copy failures on Unit objects.
        """
        cp = CompositionBelief.__new__(CompositionBelief)
        cp._last_seen = self._last_seen.copy()
        cp._confidence = self._confidence.copy()
        cp._unit_types = self._unit_types.copy()
        cp._units = self._units.copy()  # shallow — shares Unit refs (read-only)
        cp._seen_structures = self._seen_structures.copy()
        cp._expected_units = self._expected_units.copy()
        cp._destroyed = self._destroyed.copy()
        cp._last_update_time = self._last_update_time
        return cp

    def is_known_tag(self, tag: int) -> bool:
        """Check if a unit tag is tracked (known enemy) or already confirmed destroyed.

        Used by on_unit_destroyed to filter: only enemy tags should be recorded,
        friendly tags are irrelevant and would bloat _destroyed with no-ops.
        """
        return tag in self._unit_types or tag in self._destroyed

    def P_unit_exists(self, tag: int) -> float:
        """P(unit still exists | age, type) for a specific unit tag."""
        if tag in self._destroyed:
            return 0.0
        return self._confidence.get(tag, 0.0)

    def get_weighted_army(
        self,
        exclude_workers: bool = True,
        exclude_structures: bool = True,
        exclude_ignored: bool = True,
        min_confidence: float = 0.01,
    ) -> list[WeightedUnit]:
        """Return enemy army units weighted by P(exists) confidence.

        Args:
            exclude_workers: Filter out SCV/Drone/Probe/MULE
            exclude_structures: Filter out buildings
            exclude_ignored: Filter out COMMON_UNIT_IGNORE_TYPES
            min_confidence: Minimum P(exists) to include (default 0.01, filters noise)

        Returns:
            List of WeightedUnit(unit, confidence) sorted by confidence descending.
        """
        from ares.consts import WORKER_TYPES

        from bot.constants import COMMON_UNIT_IGNORE_TYPES

        result = []
        for tag, confidence in self._confidence.items():
            if confidence < min_confidence:
                continue
            if tag not in self._units:
                continue
            unit = self._units[tag]
            if exclude_workers and unit.type_id in WORKER_TYPES:
                continue
            if exclude_structures and unit.is_structure:
                continue
            if exclude_ignored and unit.type_id in COMMON_UNIT_IGNORE_TYPES:
                continue
            result.append(WeightedUnit(unit=unit, confidence=confidence))
        result.sort(key=lambda wu: wu.confidence, reverse=True)
        return result

    @property
    def freshness(self) -> float:
        """Aggregated intel freshness score, similar to but smoother than
        get_enemy_intel_quality()['freshness']. Weighted average of P(exists)."""
        if not self._confidence:
            return 0.0
        active = [c for c in self._confidence.values() if c >= 0.01]
        if not active:
            return 0.0
        return sum(active) / len(active)

    def P_expected_unit(self, unit_type: UnitTypeId) -> float:
        """P(unit_type is being produced | structures seen).

        Returns 0.0 if no prior exists for this unit type.
        """
        if unit_type not in self._expected_units:
            return 0.0
        return self._expected_units[unit_type][0]

    def get_expected_unit_types(self) -> dict[UnitTypeId, float]:
        """Return all expected-but-unseen unit types with their prior probabilities."""
        return {ut: prob for ut, (prob, _) in self._expected_units.items()}

    @property
    def total_weighted_count(self) -> float:
        """Sum of P(exists) across all tracked units — effective army size estimate."""
        return sum(c for c in self._confidence.values() if c >= 0.01)