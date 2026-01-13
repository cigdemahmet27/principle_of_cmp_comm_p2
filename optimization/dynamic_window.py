"""
Dynamic Window Adjustment

This module implements dynamic window size adjustment strategies
to optimize performance under varying channel conditions.
"""

from typing import Optional, List
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CongestionState(Enum):
    """Congestion control state."""
    SLOW_START = 0
    CONGESTION_AVOIDANCE = 1
    FAST_RECOVERY = 2
    BURST_RECOVERY = 3


@dataclass
class WindowState:
    """Current window state."""
    cwnd: float         # Congestion window (in frames)
    ssthresh: float     # Slow start threshold
    state: CongestionState
    max_window: int     # Maximum allowed window


class DynamicWindowOptimizer:
    """
    Dynamic Window Size Adjustment.
    
    Adapts the send window based on:
    - Channel conditions (burst detection)
    - Congestion signals (losses)
    - RTT variation
    
    Uses a modified TCP-like congestion control algorithm
    adapted for burst error channels.
    """
    
    def __init__(
        self,
        initial_window: int = 1,
        max_window: int = 64,
        initial_ssthresh: int = 32,
        burst_loss_threshold: int = 3,  # Losses indicating burst
        recovery_rate: float = 0.1,     # Window growth in recovery
        enable_aimd: bool = True,       # Additive Increase Multiplicative Decrease
        enable_burst_detection: bool = True
    ):
        """
        Initialize dynamic window optimizer.
        
        Args:
            initial_window: Starting window size
            max_window: Maximum window size
            initial_ssthresh: Initial slow start threshold
            burst_loss_threshold: Losses to trigger burst mode
            recovery_rate: Window growth rate in recovery
            enable_aimd: Enable AIMD congestion control
            enable_burst_detection: Enable burst detection
        """
        self.initial_window = initial_window
        self.max_window = max_window
        self.initial_ssthresh = initial_ssthresh
        self.burst_loss_threshold = burst_loss_threshold
        self.recovery_rate = recovery_rate
        self.enable_aimd = enable_aimd
        self.enable_burst_detection = enable_burst_detection
        
        # Current state
        self.cwnd = float(initial_window)
        self.ssthresh = float(initial_ssthresh)
        self.state = CongestionState.SLOW_START
        
        # Loss tracking
        self.consecutive_losses = 0
        self.total_losses = 0
        self.burst_mode = False
        self.burst_count = 0
        
        # ACK tracking
        self.new_acks_count = 0
        self.duplicate_acks = 0
        
        # History
        self.window_history: List[float] = []
        self.max_history = 1000
        
        # Statistics
        self.window_decreases = 0
        self.window_increases = 0
        self.slow_start_exits = 0
    
    def get_window_size(self) -> int:
        """
        Get current effective window size.
        
        Returns:
            Integer window size
        """
        return max(1, min(int(self.cwnd), self.max_window))
    
    def on_ack(self, ack_count: int = 1):
        """
        Handle successful ACK(s).
        
        Args:
            ack_count: Number of new ACKs received
        """
        self.consecutive_losses = 0
        self.new_acks_count += ack_count
        self.duplicate_acks = 0
        
        if not self.enable_aimd:
            return
        
        if self.state == CongestionState.SLOW_START:
            # Exponential growth
            self.cwnd += ack_count
            
            if self.cwnd >= self.ssthresh:
                self.state = CongestionState.CONGESTION_AVOIDANCE
                self.slow_start_exits += 1
                
        elif self.state == CongestionState.CONGESTION_AVOIDANCE:
            # Linear growth (AIMD - Additive Increase)
            self.cwnd += ack_count / self.cwnd
            
        elif self.state == CongestionState.FAST_RECOVERY:
            # Inflate window temporarily
            self.cwnd += ack_count
            
        elif self.state == CongestionState.BURST_RECOVERY:
            # Slow recovery from burst
            self.cwnd += self.recovery_rate * ack_count
        
        # Apply limits
        self.cwnd = min(self.cwnd, float(self.max_window))
        
        # Record history
        self._record_window()
        self.window_increases += 1
    
    def on_duplicate_ack(self):
        """Handle duplicate ACK."""
        self.duplicate_acks += 1
        
        if not self.enable_aimd:
            return
        
        if self.duplicate_acks >= 3:
            # Fast retransmit trigger
            if self.state != CongestionState.FAST_RECOVERY:
                self._enter_fast_recovery()
    
    def on_loss(self):
        """
        Handle loss event (timeout or NAK).
        """
        self.consecutive_losses += 1
        self.total_losses += 1
        
        if not self.enable_aimd:
            return
        
        # Detect burst
        if (self.enable_burst_detection and 
            self.consecutive_losses >= self.burst_loss_threshold):
            self._enter_burst_recovery()
        else:
            self._enter_congestion_avoidance()
        
        self._record_window()
        self.window_decreases += 1
    
    def on_timeout(self):
        """Handle retransmission timeout."""
        self.on_loss()
        
        if self.enable_aimd:
            # More aggressive reduction on timeout
            self.ssthresh = max(2, self.cwnd / 2)
            self.cwnd = float(self.initial_window)
            self.state = CongestionState.SLOW_START
    
    def _enter_fast_recovery(self):
        """Enter fast recovery state."""
        self.ssthresh = max(2, self.cwnd / 2)
        self.cwnd = self.ssthresh + 3  # +3 for the 3 duplicate ACKs
        self.state = CongestionState.FAST_RECOVERY
    
    def _enter_congestion_avoidance(self):
        """Enter congestion avoidance (multiplicative decrease)."""
        self.ssthresh = max(2, self.cwnd / 2)
        self.cwnd = self.ssthresh
        self.state = CongestionState.CONGESTION_AVOIDANCE
    
    def _enter_burst_recovery(self):
        """Enter burst recovery mode."""
        if not self.burst_mode:
            self.burst_mode = True
            self.burst_count += 1
        
        # More aggressive reduction for burst
        self.ssthresh = max(2, self.cwnd / 4)
        self.cwnd = max(1, self.cwnd / 4)
        self.state = CongestionState.BURST_RECOVERY
    
    def exit_burst_recovery(self):
        """Exit burst recovery mode."""
        self.burst_mode = False
        self.consecutive_losses = 0
        self.state = CongestionState.CONGESTION_AVOIDANCE
    
    def _record_window(self):
        """Record window size in history."""
        self.window_history.append(self.cwnd)
        if len(self.window_history) > self.max_history:
            self.window_history.pop(0)
    
    def get_state(self) -> WindowState:
        """Get current window state."""
        return WindowState(
            cwnd=self.cwnd,
            ssthresh=self.ssthresh,
            state=self.state,
            max_window=self.max_window
        )
    
    def get_statistics(self) -> dict:
        """Get optimizer statistics."""
        avg_window = (sum(self.window_history) / len(self.window_history)
                     if self.window_history else self.cwnd)
        
        return {
            'current_cwnd': self.cwnd,
            'effective_window': self.get_window_size(),
            'ssthresh': self.ssthresh,
            'state': self.state.name,
            'burst_mode': self.burst_mode,
            'burst_count': self.burst_count,
            'total_losses': self.total_losses,
            'consecutive_losses': self.consecutive_losses,
            'window_increases': self.window_increases,
            'window_decreases': self.window_decreases,
            'slow_start_exits': self.slow_start_exits,
            'avg_window': avg_window,
            'max_window': max(self.window_history) if self.window_history else self.cwnd,
            'min_window': min(self.window_history) if self.window_history else self.cwnd
        }
    
    def reset(self):
        """Reset optimizer to initial state."""
        self.cwnd = float(self.initial_window)
        self.ssthresh = float(self.initial_ssthresh)
        self.state = CongestionState.SLOW_START
        self.consecutive_losses = 0
        self.total_losses = 0
        self.burst_mode = False
        self.burst_count = 0
        self.new_acks_count = 0
        self.duplicate_acks = 0
        self.window_history.clear()
        self.window_decreases = 0
        self.window_increases = 0
        self.slow_start_exits = 0


if __name__ == "__main__":
    # Test dynamic window optimizer
    print("=" * 60)
    print("DYNAMIC WINDOW OPTIMIZER TEST")
    print("=" * 60)
    
    optimizer = DynamicWindowOptimizer(
        initial_window=1,
        max_window=64,
        initial_ssthresh=16
    )
    
    print(f"\nInitial state: {optimizer.get_state()}")
    
    # Simulate slow start with ACKs
    print("\n--- Slow Start Phase ---")
    for i in range(20):
        optimizer.on_ack()
        state = optimizer.get_state()
        print(f"ACK {i+1}: cwnd={state.cwnd:.1f}, state={state.state.name}")
        
        if state.state != CongestionState.SLOW_START:
            break
    
    # Simulate congestion avoidance
    print("\n--- Congestion Avoidance Phase ---")
    for i in range(10):
        optimizer.on_ack()
        state = optimizer.get_state()
        print(f"ACK: cwnd={state.cwnd:.1f}")
    
    # Simulate loss
    print("\n--- Loss Event ---")
    optimizer.on_loss()
    state = optimizer.get_state()
    print(f"After loss: cwnd={state.cwnd:.1f}, state={state.state.name}")
    
    # Simulate burst losses
    print("\n--- Burst Loss Event ---")
    for i in range(4):
        optimizer.on_loss()
        state = optimizer.get_state()
        print(f"Loss {i+1}: cwnd={state.cwnd:.1f}, "
              f"state={state.state.name}, burst={optimizer.burst_mode}")
    
    print(f"\nFinal statistics: {optimizer.get_statistics()}")
