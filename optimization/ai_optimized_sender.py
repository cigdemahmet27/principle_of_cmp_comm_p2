"""
AI-Optimized Sender for Selective Repeat ARQ

This module implements an enhanced sender that uses AI-based optimizations:

OPTIMIZATIONS IMPLEMENTED:
========================

1. ADAPTIVE TIMEOUT (Jacobson/Karels Algorithm)
   - Problem: Fixed timeout is either too long (wasted time) or too short (spurious retx)
   - Solution: Dynamically track RTT and adjust RTO:
     * SRTT = (1 - α) × SRTT + α × RTT
     * RTTVAR = (1 - β) × RTTVAR + β × |SRTT - RTT|
     * RTO = SRTT + max(G, K × RTTVAR)
   - Benefit: Faster retransmission when needed, fewer spurious retransmissions

2. DYNAMIC WINDOW SIZING (AIMD - Additive Increase, Multiplicative Decrease)
   - Problem: Fixed window may underutilize channel or cause congestion during bursts
   - Solution: Adapt window based on channel conditions:
     * Slow Start: Double window each RTT until loss
     * Congestion Avoidance: Add 1/cwnd per ACK
     * On Loss: Halve window (multiplicative decrease)
     * Burst Recovery: Reduce to minimum during detected burst
   - Benefit: Better channel utilization, faster recovery from bursts

3. BURST DETECTION AND RESPONSE
   - Problem: During Gilbert-Elliot BAD state, many consecutive frames are lost
   - Solution: Detect burst patterns and pause sending briefly
     * Track consecutive losses
     * If > threshold, enter burst recovery mode
     * Wait before resuming to let channel recover
   - Benefit: Avoid wasting bandwidth during known-bad channel periods

4. FAST RETRANSMIT (on NAK)
   - Problem: Waiting for timeout is slow
   - Solution: Immediately retransmit when NAK received
   - Benefit: Faster recovery from detected losses
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque

from src.arq.frame import Frame, FrameType
from src.arq.sender import SRSender
from optimization.adaptive_timeout import AdaptiveTimeoutOptimizer
from optimization.dynamic_window import DynamicWindowOptimizer, CongestionState


@dataclass
class OptimizationStats:
    """Statistics for optimization performance."""
    rto_updates: int = 0
    window_adjustments: int = 0
    burst_detections: int = 0
    fast_retransmits: int = 0
    timeouts_saved: int = 0  # Retransmits done faster than default timeout


class AIOptimizedSender(SRSender):
    """
    AI-Optimized Selective Repeat ARQ Sender.
    
    Inherits from SRSender and adds:
    - Adaptive timeout management
    - Dynamic window sizing
    - Burst error detection and response
    - Fast retransmit on NAK
    """
    
    def __init__(
        self,
        window_size: int = 8,
        timeout: float = 0.5,
        max_retransmissions: int = 10,
        # Optimization parameters
        enable_adaptive_timeout: bool = True,
        enable_dynamic_window: bool = True,
        enable_burst_detection: bool = True,
        # Adaptive timeout params
        rto_alpha: float = 0.125,  # SRTT smoothing
        rto_beta: float = 0.25,   # RTTVAR smoothing
        rto_k: float = 4.0,       # RTO = SRTT + K * RTTVAR
        min_rto: float = 0.05,    # Minimum RTO
        max_rto: float = 5.0,     # Maximum RTO
        # Dynamic window params
        initial_cwnd: int = 4,
        max_cwnd: int = 64,
        # Burst detection params
        burst_threshold: int = 3  # Consecutive losses to trigger burst mode
    ):
        """
        Initialize AI-optimized sender.
        
        Args:
            window_size: Maximum send window size
            timeout: Initial retransmission timeout
            max_retransmissions: Max retries per frame
            enable_adaptive_timeout: Use adaptive RTO
            enable_dynamic_window: Use dynamic window sizing
            enable_burst_detection: Detect and respond to burst errors
        """
        # Initialize base sender
        super().__init__(
            window_size=window_size,
            timeout=timeout,
            max_retransmissions=max_retransmissions,
            adaptive_timeout=False  # We handle this ourselves
        )
        
        # Optimization flags
        self.enable_adaptive_timeout = enable_adaptive_timeout
        self.enable_dynamic_window = enable_dynamic_window
        self.enable_burst_detection = enable_burst_detection
        
        # Adaptive timeout optimizer
        self.timeout_optimizer = AdaptiveTimeoutOptimizer(
            initial_rto=timeout,
            alpha=rto_alpha,
            beta=rto_beta,
            k=rto_k,
            min_rto=min_rto,
            max_rto=max_rto,
            burst_detection_threshold=burst_threshold
        )
        
        # Dynamic window optimizer
        self.window_optimizer = DynamicWindowOptimizer(
            initial_window=initial_cwnd,
            max_window=max_cwnd,
            enable_burst_detection=enable_burst_detection
        )
        
        # Statistics
        self.opt_stats = OptimizationStats()
        self.default_timeout = timeout
        
        # Burst state
        self.consecutive_losses = 0
        self.burst_threshold = burst_threshold
        self.in_burst_mode = False
        
    def get_effective_window_size(self) -> int:
        """Get current effective window size (may be reduced by congestion control)."""
        if self.enable_dynamic_window:
            return min(self.window.size, self.window_optimizer.get_window_size())
        return self.window.size
    
    def get_current_timeout(self) -> float:
        """Get current RTO value."""
        if self.enable_adaptive_timeout:
            return self.timeout_optimizer.get_rto()
        return self.default_timeout
    
    def can_send(self) -> bool:
        """Check if we can send more frames (respecting dynamic window)."""
        if self.in_burst_mode:
            return False  # Pause during detected burst
        
        effective_window = self.get_effective_window_size()
        current_in_flight = len(self.window.unacked)
        
        return (current_in_flight < effective_window and 
                len(self.send_queue) > 0)
    
    def process_ack(self, ack_num: int, current_time: float) -> bool:
        """
        Process ACK with optimizations.
        
        Args:
            ack_num: Acknowledged sequence number
            current_time: Current simulation time
            
        Returns:
            True if ACK was valid
        """
        # Calculate RTT for this ACK
        if ack_num in self.send_times:
            rtt = current_time - self.send_times[ack_num]
            
            # Update adaptive timeout
            if self.enable_adaptive_timeout:
                was_retransmit = self.timer_manager.get_retransmit_count(ack_num) > 0
                self.timeout_optimizer.update(rtt, was_retransmit)
                self.timeout_optimizer.on_ack_received()
                self.opt_stats.rto_updates += 1
        
        # Process ACK in base class
        result = super().process_ack(ack_num, current_time)
        
        if result:
            # Reset consecutive loss counter
            self.consecutive_losses = 0
            
            # Exit burst mode if we were in it
            if self.in_burst_mode:
                self.in_burst_mode = False
                self.window_optimizer.exit_burst_recovery()
            
            # Update dynamic window (success)
            if self.enable_dynamic_window:
                self.window_optimizer.on_ack(1)
                self.opt_stats.window_adjustments += 1
        
        return result
    
    def process_nak(self, nak_num: int, current_time: float) -> Optional[Frame]:
        """
        Process NAK with fast retransmit.
        
        Args:
            nak_num: NAK'd sequence number
            current_time: Current simulation time
            
        Returns:
            Frame to retransmit, or None
        """
        # Fast retransmit - don't wait for timeout
        self.opt_stats.fast_retransmits += 1
        
        # Record as loss for window adjustment
        self._handle_loss_event()
        
        return super().process_nak(nak_num, current_time)
    
    def _handle_loss_event(self):
        """Handle a loss event (timeout or NAK)."""
        self.consecutive_losses += 1
        
        # Update window optimizer
        if self.enable_dynamic_window:
            self.window_optimizer.on_loss()
            self.opt_stats.window_adjustments += 1
        
        # Check for burst
        if self.enable_burst_detection:
            if self.consecutive_losses >= self.burst_threshold:
                if not self.in_burst_mode:
                    self.in_burst_mode = True
                    self.window_optimizer._enter_burst_recovery()
                    self.opt_stats.burst_detections += 1
    
    def get_frames_to_retransmit(self, current_time: float) -> List[Frame]:
        """
        Get frames that need retransmission (with adaptive timeout).
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of frames to retransmit
        """
        # Use adaptive timeout instead of fixed
        current_rto = self.get_current_timeout()
        
        frames_to_retx = []
        expired_seqs = self.timer_manager.check_expired(current_time)
        
        for seq_num in expired_seqs:
            if seq_num in self.frame_buffer.frames:
                frame = self.frame_buffer.get(seq_num)
                if frame:
                    # Check if we should retransmit
                    retx_count = self.timer_manager.get_retransmit_count(seq_num)
                    
                    if retx_count < self.max_retransmissions:
                        # Mark as retransmission
                        frame.set_retransmit()
                        frames_to_retx.append(frame)
                        
                        # Restart timer with possibly increased timeout
                        if self.enable_adaptive_timeout:
                            self.timeout_optimizer.on_timeout()
                        
                        new_timeout = self.get_current_timeout()
                        self.timer_manager.start_timer(
                            seq_num, current_time, new_timeout
                        )
                        
                        # Check if we saved time compared to default
                        if current_rto < self.default_timeout:
                            self.opt_stats.timeouts_saved += 1
                        
                        # Record loss event
                        self._handle_loss_event()
        
        return frames_to_retx
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics."""
        return {
            'rto_updates': self.opt_stats.rto_updates,
            'window_adjustments': self.opt_stats.window_adjustments,
            'burst_detections': self.opt_stats.burst_detections,
            'fast_retransmits': self.opt_stats.fast_retransmits,
            'timeouts_saved': self.opt_stats.timeouts_saved,
            'current_rto': self.get_current_timeout(),
            'current_window': self.get_effective_window_size(),
            'in_burst_mode': self.in_burst_mode,
            'timeout_stats': self.timeout_optimizer.get_statistics(),
            'window_stats': self.window_optimizer.get_statistics()
        }
    
    def reset(self):
        """Reset sender state."""
        super().reset()
        self.timeout_optimizer.reset()
        self.window_optimizer.reset()
        self.opt_stats = OptimizationStats()
        self.consecutive_losses = 0
        self.in_burst_mode = False


if __name__ == "__main__":
    print("=" * 60)
    print("AI-OPTIMIZED SENDER TEST")
    print("=" * 60)
    
    sender = AIOptimizedSender(
        window_size=16,
        timeout=0.5,
        enable_adaptive_timeout=True,
        enable_dynamic_window=True,
        enable_burst_detection=True
    )
    
    print(f"\nInitial State:")
    print(f"  Window Size: {sender.get_effective_window_size()}")
    print(f"  Timeout: {sender.get_current_timeout()*1000:.1f} ms")
    print(f"  Can Send: {sender.can_send()}")
    
    # Simulate some data
    for i in range(10):
        sender.queue_data(f"Data {i}".encode(), is_last=(i == 9))
    
    print(f"\nAfter queueing 10 segments:")
    print(f"  Queue size: {len(sender.send_queue)}")
    print(f"  Can Send: {sender.can_send()}")
    
    print(f"\nOptimization Stats: {sender.get_optimization_stats()}")
