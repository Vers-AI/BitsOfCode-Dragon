"""
Performance Monitor
Purpose: Tracks bot economy and spending efficiency for both decision-making and reporting.
Key Decisions: Uses Spending Quotient (SQ) formula from Liquipedia for efficiency metrics.
              Economy state is a simple capacity check; SQ measures spending efficiency.
Limitations: SQ unreliable when income < 600/min (early game).
"""

import math

from bot.constants import RESOURCE_IMBALANCE_RATIO


def get_resource_pressure(bot, sustained: bool = False) -> str:
    """Classify current resource balance — single source of truth for pressure state.

    Used by composition nudging (sustained=False, instantaneous bank) and by
    structural decisions like assimilator queuing (sustained=True, income rates).

    sustained=True uses collection_rate_* — immune to spending bursts, so a
    warp-in cycle that dips the mineral bank won't flip the signal. Right for
    long-lived decisions where a false positive means halting gas infrastructure.
    sustained=False uses bot.minerals / bot.vespene — responsive to the current
    bank, which is the point for composition nudging: produce what we can afford
    *right now*.

    Returns "MINERAL_STARVED", "GAS_STARVED", or "BALANCED".
    Threshold governed by RESOURCE_IMBALANCE_RATIO (default 2.0).

    Perf note: O(1) — two attribute reads. Negligible.
    """
    if sustained:
        minerals = bot.state.score.collection_rate_minerals
        vespene = bot.state.score.collection_rate_vespene
    else:
        minerals = bot.minerals
        vespene = bot.vespene

    if minerals > RESOURCE_IMBALANCE_RATIO * max(vespene, 1):
        return "GAS_STARVED"
    if vespene > RESOURCE_IMBALANCE_RATIO * max(minerals, 1):
        return "MINERAL_STARVED"
    return "BALANCED"


def get_economy_state(bot) -> str:
    """
    Single source of truth for economy health.
    Used to gate production intensity and upgrade timing.
    
    Returns:
        str: "recovery", "reduced", "moderate", or "full"
    
    Thresholds based on SC2 economy research:
    - 1 saturated base (~16 mineral workers) = ~900-1000 minerals/min
    - 1 gate continuous production = ~220-250 minerals/min
    - So ~700+ minerals/min can support 3 gates comfortably
    """
    workers = bot.workers.amount
    mineral_rate = bot.state.score.collection_rate_minerals
    
    # Hard safety gate: critically underdeveloped
    if workers < 20:
        return "recovery"
    
    # Income-based tiers
    if mineral_rate < 1000 or workers < 30:
        return "reduced"
    elif mineral_rate < 1600 or workers < 44:
        return "moderate"
    else:
        return "full"


class PerformanceMonitor:
    """
    Monitors bot performance throughout the game.
    Tracks Spending Quotient and other efficiency metrics for real-time decision-making.
    """
    
    def __init__(self, sample_interval: int = 22, alpha: float = 0.1):
        """
        Initialize the performance monitor.
        
        Args:
            sample_interval: How often to sample (in game steps, default ~1 second)
            alpha: EMA smoothing factor (0.0-1.0, higher = more weight on recent values)
        """
        self.sample_interval = sample_interval
        self.alpha = alpha
        
        # Spending Quotient tracking
        self.avg_unspent_minerals = 0.0
        self.avg_unspent_vespene = 0.0
        self.avg_income_minerals = 0.0
        self.avg_income_vespene = 0.0
        
        # Additional performance metrics (can be extended)
        self.samples_taken = 0
        self.tracking_enabled = False  # Start tracking after early game (6 min)
    
    def update(self, iteration: int, bot) -> None:
        """
        Update performance metrics if it's time to sample.
        Only starts tracking after early game ends (game_state >= 1, after 6 minutes).
        
        Args:
            iteration: Current game iteration/step
            bot: Bot instance (to access minerals, income, etc.)
        """
        # Enable tracking once early game ends (game_state >= 1 means mid/late game)
        if not self.tracking_enabled and bot.game_state >= 1:
            self.tracking_enabled = True
        
        # Only track if enabled and it's time to sample
        if not self.tracking_enabled or iteration % self.sample_interval != 0:
            return
        
        self.samples_taken += 1
        
        # Update average unspent resources (EMA)
        self.avg_unspent_minerals = (
            self.alpha * bot.minerals + 
            (1 - self.alpha) * self.avg_unspent_minerals
        )
        self.avg_unspent_vespene = (
            self.alpha * bot.vespene + 
            (1 - self.alpha) * self.avg_unspent_vespene
        )
        
        # Update average income (EMA)
        current_income = bot.state.score.collection_rate_minerals
        current_gas_income = bot.state.score.collection_rate_vespene
        
        self.avg_income_minerals = (
            self.alpha * current_income + 
            (1 - self.alpha) * self.avg_income_minerals
        )
        self.avg_income_vespene = (
            self.alpha * current_gas_income + 
            (1 - self.alpha) * self.avg_income_vespene
        )
    
    def get_current_sq(self) -> float:
        """
        Get current Spending Quotient for real-time decision-making.
        Uses COMBINED income and unspent resources (minerals + gas together).
        
        Note: Combined SQ with EMA values gives higher numbers than the
        game-averaged SQ on Liquipedia (typically 100-240 vs 50-100) because
        our EMA starts mid-game when income is already high.
        
        Returns:
            Current combined SQ score
        """
        combined_income = self.avg_income_minerals + self.avg_income_vespene
        combined_unspent = self.avg_unspent_minerals + self.avg_unspent_vespene
        return self._calculate_sq(combined_income, combined_unspent)
    
    def get_mineral_sq(self) -> float:
        """Per-resource SQ for minerals only. More useful for detecting
        mineral-specific banking than combined SQ."""
        return self._calculate_sq(self.avg_income_minerals, self.avg_unspent_minerals)
    
    def get_gas_sq(self) -> float:
        """Per-resource SQ for gas only. More useful for detecting
        gas-specific banking than combined SQ."""
        return self._calculate_sq(self.avg_income_vespene, self.avg_unspent_vespene)
    
    def is_spending_efficiently(self, threshold: float = 120.0) -> bool:
        """
        Check if bot is spending efficiently (for in-game decisions).
        Uses per-resource SQ — both resources must be above threshold.
        
        Args:
            threshold: Minimum per-resource SQ to be considered "efficient"
            
        Returns:
            True if both mineral and gas SQ are above threshold
        """
        return self.get_mineral_sq() >= threshold and self.get_gas_sq() >= threshold
    
    def is_banking_too_much(self, threshold: float = 100.0) -> bool:
        """
        Check if banking too much of either resource.
        Uses per-resource SQ — if EITHER resource SQ is below threshold,
        we have a spending problem with that resource.
        
        Args:
            threshold: Per-resource SQ below which we're banking
            
        Returns:
            True if either resource SQ is below threshold
        """
        return self.get_mineral_sq() < threshold or self.get_gas_sq() < threshold
    
    def get_efficiency_rating(self) -> str:
        """
        Get current efficiency rating (for debugging/logging).
        Uses combined SQ with an adjusted scale for EMA-based values.
        
        Returns:
            Rating string (GRANDMASTER, MASTER, etc.)
        """
        sq = self.get_current_sq()
        return self._get_sq_rating(sq)
    
    def _calculate_sq(self, avg_income: float, avg_unspent: float) -> float:
        """
        Calculate Spending Quotient using formula: SQ(i,u) = 35(0.00137i - ln(u)) + 240
        
        Args:
            avg_income: Average collection rate
            avg_unspent: Average unspent resources
            
        Returns:
            SQ score
        """
        if avg_unspent <= 0:
            avg_unspent = 1  # Avoid ln(0)
        if avg_income <= 0:
            return 0
        
        return 35 * (0.00137 * avg_income - math.log(avg_unspent)) + 240
    
    def _get_sq_rating(self, sq: float) -> str:
        """Convert combined SQ score to skill rating.
        
        Scale is shifted up from Liquipedia's game-average scale (GM=90+)
        because our EMA starts tracking mid-game when income is already high,
        giving combined SQ values typically 100-240.
        """
        if sq >= 200:
            return "GRANDMASTER"
        elif sq >= 170:
            return "MASTER"
        elif sq >= 140:
            return "DIAMOND"
        elif sq >= 120:
            return "PLATINUM"
        elif sq >= 100:
            return "GOLD"
        else:
            return "SILVER"
    
    def should_trigger_freeflow(self, bot) -> bool:
        """
        Use per-resource SQ to detect if we should trigger freeflow mode.
        Triggers when EITHER mineral or gas SQ drops below threshold,
        indicating we're banking that resource and should spend freely.
        
        Per-resource SQ gives more useful thresholds than combined SQ:
        - At 2-base (1800 income): banking 600+ → SQ ~100
        - At 5-base (3600 income): banking 3000+ → SQ ~100
        
        Returns:
            True if per-resource SQ indicates spending problem
        """
        # SQ unreliable below 600 mineral income (per Liquipedia)
        if self.avg_income_minerals < 600 or self.samples_taken < 5:
            return False
        
        # Only trigger if we're in a good economy state but spending poorly
        economy = get_economy_state(bot)
        if economy not in ("moderate", "full"):
            return False
        
        # Per-resource SQ: if either resource is being banked, spend freely
        return self.get_mineral_sq() < 100 or self.get_gas_sq() < 100
