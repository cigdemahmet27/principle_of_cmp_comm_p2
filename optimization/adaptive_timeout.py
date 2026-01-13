"""
Adaptive Timeout Optimization

This module implements adaptive RTO (Retransmission Timeout) calculation
using Jacobson/Karels algorithm with enhancements for burst error channels.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RTTSample:
    """RTT sample with metadata."""
    rtt: float
    timestamp: float
    seq_num: int
    was_retransmit: bool = False


class AdaptiveTimeoutOptimizer:
    """
    Adaptive Timeout Optimization with Jacobson/Karels Algorithm.
    
    Enhanced for burst error channels with:
    - Aggressive backoff on burst detection
    - RTT variance tracking
    - Congestion-aware timeout adjustment
    
    Attributes:
        srtt: Smoothed Round-Trip Time
        rttvar: RTT Variance
        rto: Retransmission Timeout
    """
    
    def __init__(
        self,
        initial_rto: float = 1.0,
        alpha: float = 0.125,      # SRTT smoothing factor (1/8)
        beta: float = 0.25,        # RTTVAR smoothing factor (1/4)
        k: float = 4.0,            # RTO multiplier for variance
        min_rto: float = 0.1,      # Minimum RTO
        max_rto: float = 60.0,     # Maximum RTO
        clock_granularity: float = 0.001,  # Timer granularity
        burst_detection_threshold: int = 3  # Consecutive losses to detect burst
    ):
        """
        Initialize adaptive timeout optimizer.
        
        Args:
            initial_rto: Initial RTO value
            alpha: SRTT smoothing factor
            beta: RTTVAR smoothing factor
            k: Multiplier for RTTVAR in RTO calculation
            min_rto: Minimum RTO value
            max_rto: Maximum RTO value
            clock_granularity: Timer granularity
            burst_detection_threshold: Threshold for burst detection
        """
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.min_rto = min_rto
        self.max_rto = max_rto
        self.clock_granularity = clock_granularity
        self.burst_detection_threshold = burst_detection_threshold
        
        # RTT estimation
        self.srtt: Optional[float] = None
        self.rttvar: Optional[float] = None
        self.rto = initial_rto
        self.initial_rto = initial_rto
        
        # History
        self.rtt_history: List[RTTSample] = []
        self.max_history_size = 100
        
        # Burst detection
        self.consecutive_losses = 0
        self.burst_mode = False
        self.burst_count = 0
        
        # Statistics
        self.updates = 0
        self.backoffs = 0
        self.burst_events = 0
    
    def update(self, rtt_sample: float, was_retransmit: bool = False):
        """
        Update RTO based on new RTT sample.
        
        Uses Jacobson/Karels algorithm:
        - RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - RTT|
        - SRTT = (1 - alpha) * SRTT + alpha * RTT
        - RTO = SRTT + max(G, K * RTTVAR)
        
        Args:
            rtt_sample: Measured RTT in seconds
            was_retransmit: Whether this was a retransmitted frame
        """
        # Don't update from retransmits (Karn's algorithm)
        if was_retransmit:
            return
        
        # Store sample
        sample = RTTSample(
            rtt=rtt_sample,
            timestamp=0,  # Would be set from simulation
            seq_num=0,
            was_retransmit=was_retransmit
        )
        self.rtt_history.append(sample)
        if len(self.rtt_history) > self.max_history_size:
            self.rtt_history.pop(0)
        
        # First sample initialization
        if self.srtt is None:
            self.srtt = rtt_sample
            self.rttvar = rtt_sample / 2
        else:
            # Jacobson/Karels update
            diff = abs(self.srtt - rtt_sample)
            self.rttvar = (1 - self.beta) * self.rttvar + self.beta * diff
            self.srtt = (1 - self.alpha) * self.srtt + self.alpha * rtt_sample
        
        # Calculate RTO
        self._calculate_rto()
        
        # Reset consecutive losses
        self.consecutive_losses = 0
        if self.burst_mode:
            self.burst_mode = False
        
        self.updates += 1
    
    def _calculate_rto(self):
        """Calculate RTO from current estimates."""
        if self.srtt is None:
            return
        
        # RTO = SRTT + max(G, K * RTTVAR)
        variance_term = max(self.clock_granularity, self.k * self.rttvar)
        self.rto = self.srtt + variance_term
        
        # Apply bounds
        self.rto = max(self.min_rto, min(self.max_rto, self.rto))
    
    def on_timeout(self):
        """
        Handle timeout event - apply exponential backoff.
        
        Called when a retransmission timeout occurs.
        """
        self.consecutive_losses += 1
        self.backoffs += 1
        
        # Exponential backoff
        self.rto = min(self.max_rto, self.rto * 2)
        
        # Detect burst
        if self.consecutive_losses >= self.burst_detection_threshold:
            if not self.burst_mode:
                self.burst_mode = True
                self.burst_count += 1
                self.burst_events += 1
                # More aggressive backoff for burst
                self.rto = min(self.max_rto, self.rto * 1.5)
    
    def on_ack_received(self):
        """Handle successful ACK - reset loss counter."""
        self.consecutive_losses = 0
    
    def get_rto(self) -> float:
        """Get current RTO value."""
        return self.rto
    
    def get_srtt(self) -> Optional[float]:
        """Get smoothed RTT."""
        return self.srtt
    
    def get_rttvar(self) -> Optional[float]:
        """Get RTT variance."""
        return self.rttvar
    
    def is_burst_detected(self) -> bool:
        """Check if currently in burst loss mode."""
        return self.burst_mode
    
    def get_statistics(self) -> dict:
        """Get optimizer statistics."""
        rtt_values = [s.rtt for s in self.rtt_history]
        
        return {
            'srtt': self.srtt,
            'rttvar': self.rttvar,
            'rto': self.rto,
            'updates': self.updates,
            'backoffs': self.backoffs,
            'burst_events': self.burst_events,
            'current_burst_mode': self.burst_mode,
            'consecutive_losses': self.consecutive_losses,
            'rtt_samples': len(rtt_values),
            'rtt_min': min(rtt_values) if rtt_values else 0,
            'rtt_max': max(rtt_values) if rtt_values else 0,
            'rtt_mean': sum(rtt_values) / len(rtt_values) if rtt_values else 0
        }
    
    def reset(self):
        """Reset optimizer to initial state."""
        self.srtt = None
        self.rttvar = None
        self.rto = self.initial_rto
        self.rtt_history.clear()
        self.consecutive_losses = 0
        self.burst_mode = False
        self.burst_count = 0
        self.updates = 0
        self.backoffs = 0
        self.burst_events = 0


class EnhancedAdaptiveTimeout(AdaptiveTimeoutOptimizer):
    """
    Enhanced adaptive timeout with burst-specific optimizations.
    
    Additional features:
    - RTT trend detection
    - Proactive timeout adjustment
    - Burst recovery speedup
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Trend detection
        self.rtt_trend_window = 5
        self.trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    
    def _detect_trend(self) -> Optional[str]:
        """Detect RTT trend from recent samples."""
        if len(self.rtt_history) < self.rtt_trend_window:
            return None
        
        recent = [s.rtt for s in self.rtt_history[-self.rtt_trend_window:]]
        
        # Simple linear trend
        increasing = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        decreasing = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))
        
        if increasing:
            return "increasing"
        elif decreasing:
            return "decreasing"
        return "stable"
    
    def update(self, rtt_sample: float, was_retransmit: bool = False):
        """Update with trend detection."""
        super().update(rtt_sample, was_retransmit)
        
        # Detect trend
        self.trend = self._detect_trend()
        
        # Proactive adjustment
        if self.trend == "increasing":
            # RTT is increasing, preemptively increase RTO slightly
            self.rto *= 1.1
            self.rto = min(self.max_rto, self.rto)
    
    def on_burst_recovery(self):
        """
        Called when recovering from burst.
        
        Gradually reduce RTO to normal levels.
        """
        if self.srtt is not None:
            # Faster recovery after burst ends
            target_rto = self.srtt + self.k * self.rttvar
            self.rto = (self.rto + target_rto) / 2  # Smooth transition
            self.rto = max(self.min_rto, min(self.max_rto, self.rto))


if __name__ == "__main__":
    # Test adaptive timeout
    print("=" * 60)
    print("ADAPTIVE TIMEOUT OPTIMIZER TEST")
    print("=" * 60)
    
    optimizer = AdaptiveTimeoutOptimizer(initial_rto=0.5)
    
    # Simulate RTT samples
    import random
    
    print("\nNormal operation (stable RTT around 100ms):")
    for i in range(10):
        rtt = 0.1 + random.gauss(0, 0.01)
        optimizer.update(rtt)
        print(f"  RTT={rtt*1000:.1f}ms -> SRTT={optimizer.srtt*1000:.1f}ms, "
              f"RTO={optimizer.rto*1000:.1f}ms")
    
    print("\nSimulating timeouts (burst losses):")
    for i in range(5):
        optimizer.on_timeout()
        print(f"  Timeout #{i+1}: RTO={optimizer.rto*1000:.1f}ms, "
              f"burst_mode={optimizer.burst_mode}")
    
    print("\nRecovery (successful ACKs):")
    for i in range(5):
        optimizer.on_ack_received()
        rtt = 0.1 + random.gauss(0, 0.01)
        optimizer.update(rtt)
        print(f"  RTT={rtt*1000:.1f}ms -> RTO={optimizer.rto*1000:.1f}ms")
    
    print(f"\nFinal statistics: {optimizer.get_statistics()}")
