"""
Timer Management for Selective Repeat ARQ

This module provides per-frame timer management for the ARQ protocol,
handling timeout detection and retransmission triggering.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from enum import Enum
import heapq


class TimerState(Enum):
    """Timer state enumeration."""
    STOPPED = 0
    RUNNING = 1
    EXPIRED = 2


@dataclass(order=True)
class TimerEvent:
    """Timer event for priority queue management."""
    expiry_time: float
    seq_num: int = field(compare=False)
    generation: int = field(compare=False)  # To invalidate cancelled timers


@dataclass
class FrameTimer:
    """
    Per-frame timer implementation.
    
    Attributes:
        seq_num: Sequence number of the frame
        timeout: Timeout duration in seconds
        start_time: Time when timer was started
        state: Current timer state
        retransmit_count: Number of retransmissions
    """
    seq_num: int
    timeout: float
    start_time: float = 0.0
    state: TimerState = TimerState.STOPPED
    retransmit_count: int = 0
    generation: int = 0  # Incremented on each restart
    
    def start(self, current_time: float):
        """
        Start the timer.
        
        Args:
            current_time: Current simulation time
        """
        self.start_time = current_time
        self.state = TimerState.RUNNING
        self.generation += 1
    
    def stop(self):
        """Stop the timer."""
        self.state = TimerState.STOPPED
    
    def restart(self, current_time: float):
        """
        Restart the timer (for retransmission).
        
        Args:
            current_time: Current simulation time
        """
        self.start(current_time)
        self.retransmit_count += 1
    
    def check_expired(self, current_time: float) -> bool:
        """
        Check if timer has expired.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if timer has expired
        """
        if self.state != TimerState.RUNNING:
            return False
        
        if current_time >= self.start_time + self.timeout:
            self.state = TimerState.EXPIRED
            return True
        
        return False
    
    def get_remaining_time(self, current_time: float) -> float:
        """
        Get remaining time until expiration.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Remaining time in seconds (0 if expired or stopped)
        """
        if self.state != TimerState.RUNNING:
            return 0.0
        
        remaining = (self.start_time + self.timeout) - current_time
        return max(0.0, remaining)
    
    def get_expiry_time(self) -> float:
        """Get the absolute expiry time."""
        return self.start_time + self.timeout


class TimerManager:
    """
    Manages multiple per-frame timers efficiently.
    
    Uses a priority queue (min-heap) for efficient timeout detection.
    
    Attributes:
        default_timeout: Default timeout duration
        timers: Dictionary of active timers by sequence number
        timer_queue: Priority queue of timer events
    """
    
    def __init__(
        self,
        default_timeout: float,
        max_retransmissions: int = 10,
        on_timeout: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize timer manager.
        
        Args:
            default_timeout: Default timeout duration in seconds
            max_retransmissions: Maximum retransmissions before giving up
            on_timeout: Callback function when timer expires (receives seq_num)
        """
        self.default_timeout = default_timeout
        self.max_retransmissions = max_retransmissions
        self.on_timeout = on_timeout
        
        self.timers: Dict[int, FrameTimer] = {}
        self.timer_queue: List[TimerEvent] = []
        
        # Statistics
        self.total_timeouts = 0
        self.total_timers_started = 0
    
    def start_timer(
        self,
        seq_num: int,
        current_time: float,
        timeout: Optional[float] = None
    ):
        """
        Start a timer for a frame.
        
        Args:
            seq_num: Sequence number
            current_time: Current simulation time
            timeout: Custom timeout (uses default if None)
        """
        timeout = timeout or self.default_timeout
        
        if seq_num in self.timers:
            # Restart existing timer
            timer = self.timers[seq_num]
            timer.timeout = timeout
            timer.restart(current_time)
        else:
            # Create new timer
            timer = FrameTimer(seq_num=seq_num, timeout=timeout)
            timer.start(current_time)
            self.timers[seq_num] = timer
            self.total_timers_started += 1
        
        # Add to priority queue
        event = TimerEvent(
            expiry_time=timer.get_expiry_time(),
            seq_num=seq_num,
            generation=timer.generation
        )
        heapq.heappush(self.timer_queue, event)
    
    def stop_timer(self, seq_num: int):
        """
        Stop a timer for a frame.
        
        Args:
            seq_num: Sequence number
        """
        if seq_num in self.timers:
            self.timers[seq_num].stop()
            # Don't remove from queue - will be filtered on pop
    
    def cancel_timer(self, seq_num: int):
        """
        Cancel and remove a timer.
        
        Args:
            seq_num: Sequence number
        """
        if seq_num in self.timers:
            del self.timers[seq_num]
    
    def get_timer(self, seq_num: int) -> Optional[FrameTimer]:
        """
        Get timer for a frame.
        
        Args:
            seq_num: Sequence number
            
        Returns:
            FrameTimer or None
        """
        return self.timers.get(seq_num)
    
    def check_timeouts(self, current_time: float) -> List[int]:
        """
        Check for expired timers.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of sequence numbers with expired timers
        """
        expired = []
        
        while self.timer_queue:
            # Peek at next event
            event = self.timer_queue[0]
            
            if event.expiry_time > current_time:
                break
            
            # Pop the event
            heapq.heappop(self.timer_queue)
            
            # Check if timer still exists and generation matches
            timer = self.timers.get(event.seq_num)
            if timer is None:
                continue
            
            if timer.generation != event.generation:
                # Timer was restarted, ignore this old event
                continue
            
            if timer.state != TimerState.RUNNING:
                continue
            
            # Timer has expired
            if timer.check_expired(current_time):
                self.total_timeouts += 1
                expired.append(event.seq_num)
                
                if self.on_timeout:
                    self.on_timeout(event.seq_num)
        
        return expired
    
    def get_next_expiry(self) -> Optional[float]:
        """
        Get the time of the next timer expiry.
        
        Returns:
            Next expiry time or None if no active timers
        """
        # Clean up stale events
        while self.timer_queue:
            event = self.timer_queue[0]
            timer = self.timers.get(event.seq_num)
            
            if timer is None or timer.generation != event.generation:
                heapq.heappop(self.timer_queue)
                continue
            
            if timer.state == TimerState.RUNNING:
                return event.expiry_time
            
            heapq.heappop(self.timer_queue)
        
        return None
    
    def get_retransmit_count(self, seq_num: int) -> int:
        """
        Get retransmission count for a frame.
        
        Args:
            seq_num: Sequence number
            
        Returns:
            Retransmission count
        """
        timer = self.timers.get(seq_num)
        return timer.retransmit_count if timer else 0
    
    def is_max_retransmissions(self, seq_num: int) -> bool:
        """
        Check if frame has reached max retransmissions.
        
        Args:
            seq_num: Sequence number
            
        Returns:
            True if max retransmissions reached
        """
        return self.get_retransmit_count(seq_num) >= self.max_retransmissions
    
    def clear_all(self):
        """Clear all timers."""
        self.timers.clear()
        self.timer_queue.clear()
    
    def clear_range(self, start_seq: int, end_seq: int):
        """
        Clear timers in a sequence number range.
        
        Args:
            start_seq: Start sequence number (inclusive)
            end_seq: End sequence number (exclusive)
        """
        for seq_num in range(start_seq, end_seq):
            self.cancel_timer(seq_num)
    
    def get_active_count(self) -> int:
        """Get number of active timers."""
        return sum(1 for t in self.timers.values() 
                   if t.state == TimerState.RUNNING)
    
    def get_statistics(self) -> dict:
        """Get timer statistics."""
        return {
            'total_timers_started': self.total_timers_started,
            'total_timeouts': self.total_timeouts,
            'active_timers': self.get_active_count(),
            'total_retransmissions': sum(
                t.retransmit_count for t in self.timers.values()
            )
        }
    
    def update_timeout(self, new_timeout: float):
        """
        Update the default timeout value.
        
        Args:
            new_timeout: New timeout value in seconds
        """
        self.default_timeout = new_timeout


class AdaptiveTimeoutCalculator:
    """
    Implements adaptive timeout calculation using Jacobson/Karels algorithm.
    
    This is used for dynamic RTO (Retransmission Timeout) calculation
    based on observed RTT samples.
    """
    
    def __init__(
        self,
        initial_rto: float = 1.0,
        alpha: float = 0.125,
        beta: float = 0.25,
        k: float = 4.0,
        min_rto: float = 0.1,
        max_rto: float = 60.0
    ):
        """
        Initialize adaptive timeout calculator.
        
        Args:
            initial_rto: Initial RTO value
            alpha: SRTT smoothing factor (1/8)
            beta: RTTVAR smoothing factor (1/4)
            k: Multiplier for RTTVAR
            min_rto: Minimum RTO
            max_rto: Maximum RTO
        """
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.min_rto = min_rto
        self.max_rto = max_rto
        
        self.srtt: Optional[float] = None  # Smoothed RTT
        self.rttvar: Optional[float] = None  # RTT variance
        self.rto = initial_rto
        
        self.samples_count = 0
    
    def update(self, rtt_sample: float):
        """
        Update RTO based on new RTT sample.
        
        Uses Jacobson/Karels algorithm:
        - RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - RTT|
        - SRTT = (1 - alpha) * SRTT + alpha * RTT
        - RTO = SRTT + K * RTTVAR
        
        Args:
            rtt_sample: Measured RTT sample
        """
        if self.srtt is None:
            # First sample
            self.srtt = rtt_sample
            self.rttvar = rtt_sample / 2
        else:
            # Subsequent samples
            diff = abs(self.srtt - rtt_sample)
            self.rttvar = (1 - self.beta) * self.rttvar + self.beta * diff
            self.srtt = (1 - self.alpha) * self.srtt + self.alpha * rtt_sample
        
        # Calculate RTO
        self.rto = self.srtt + self.k * self.rttvar
        
        # Clamp RTO
        self.rto = max(self.min_rto, min(self.max_rto, self.rto))
        
        self.samples_count += 1
    
    def get_rto(self) -> float:
        """Get current RTO value."""
        return self.rto
    
    def backoff(self, factor: float = 2.0):
        """
        Apply exponential backoff to RTO.
        
        Args:
            factor: Backoff multiplier
        """
        self.rto = min(self.max_rto, self.rto * factor)
    
    def reset(self, initial_rto: float = 1.0):
        """Reset calculator to initial state."""
        self.srtt = None
        self.rttvar = None
        self.rto = initial_rto
        self.samples_count = 0
    
    def get_statistics(self) -> dict:
        """Get timeout statistics."""
        return {
            'srtt': self.srtt,
            'rttvar': self.rttvar,
            'rto': self.rto,
            'samples': self.samples_count
        }


if __name__ == "__main__":
    # Test timer manager
    print("=" * 60)
    print("TIMER MANAGER TEST")
    print("=" * 60)
    
    def on_timeout(seq_num):
        print(f"  [TIMEOUT] Frame {seq_num} timed out!")
    
    manager = TimerManager(default_timeout=1.0, on_timeout=on_timeout)
    
    # Start some timers
    current_time = 0.0
    print(f"\nTime {current_time}: Starting timers for frames 0-4")
    for i in range(5):
        manager.start_timer(i, current_time, timeout=1.0 + i * 0.5)
    
    print(f"  Active timers: {manager.get_active_count()}")
    print(f"  Next expiry: {manager.get_next_expiry()}")
    
    # Simulate time passing
    for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"\nTime {t}:")
        expired = manager.check_timeouts(t)
        if expired:
            print(f"  Expired frames: {expired}")
    
    # Test adaptive timeout
    print("\n" + "=" * 60)
    print("ADAPTIVE TIMEOUT TEST")
    print("=" * 60)
    
    calculator = AdaptiveTimeoutCalculator(initial_rto=1.0)
    
    # Simulate RTT samples
    rtt_samples = [0.1, 0.12, 0.09, 0.15, 0.11, 0.13, 0.08, 0.14, 0.10, 0.12]
    
    for rtt in rtt_samples:
        calculator.update(rtt)
        stats = calculator.get_statistics()
        print(f"RTT={rtt:.2f}s -> SRTT={stats['srtt']:.4f}s, "
              f"RTTVAR={stats['rttvar']:.4f}s, RTO={stats['rto']:.4f}s")
