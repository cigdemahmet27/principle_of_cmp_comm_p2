"""
ACK Optimization Strategies

This module implements various ACK optimization techniques:
- Piggybacked ACKs
- Delayed ACKs
- ACK pacing
- Silly Window Syndrome prevention
"""

from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PendingACK:
    """A pending ACK waiting to be sent."""
    seq_num: int
    timestamp: float
    is_duplicate: bool = False


class ACKOptimizer:
    """
    ACK Optimization Strategies.
    
    Implements multiple techniques to reduce ACK overhead:
    1. Delayed ACKs: Wait briefly for more data before ACKing
    2. Piggybacked ACKs: Combine ACK with outgoing data
    3. ACK pacing: Rate-limit ACKs to reduce traffic
    4. Cumulative ACKs: Send fewer, cumulative ACKs
    
    Attributes:
        delay_time: Maximum time to delay an ACK
        max_pending: Maximum pending ACKs before forced send
    """
    
    def __init__(
        self,
        delay_time: float = 0.040,       # 40ms delay
        max_pending: int = 2,            # Max pending before force send
        enable_piggybacking: bool = True,
        enable_delayed_ack: bool = True,
        enable_cumulative: bool = True,
        on_ack_ready: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize ACK optimizer.
        
        Args:
            delay_time: Maximum ACK delay in seconds
            max_pending: Maximum number of pending ACKs
            enable_piggybacking: Enable piggybacked ACKs
            enable_delayed_ack: Enable delayed ACKs
            enable_cumulative: Enable cumulative ACKs
            on_ack_ready: Callback when ACK should be sent
        """
        self.delay_time = delay_time
        self.max_pending = max_pending
        self.enable_piggybacking = enable_piggybacking
        self.enable_delayed_ack = enable_delayed_ack
        self.enable_cumulative = enable_cumulative
        self.on_ack_ready = on_ack_ready
        
        # Pending ACKs
        self.pending_acks: deque[PendingACK] = deque()
        
        # Highest in-order ACK (for cumulative)
        self.highest_in_order = -1
        
        # Statistics
        self.acks_delayed = 0
        self.acks_piggybacked = 0
        self.acks_sent = 0
        self.acks_suppressed = 0  # ACKs avoided by cumulative
    
    def receive_data(self, seq_num: int, current_time: float, is_in_order: bool = True):
        """
        Handle received data frame.
        
        Args:
            seq_num: Sequence number of received frame
            current_time: Current simulation time
            is_in_order: Whether frame is in order
        """
        if is_in_order:
            # Update highest in-order
            if seq_num > self.highest_in_order:
                self.highest_in_order = seq_num
            
            if self.enable_cumulative:
                # For cumulative ACK, we'll just update highest and potentially
                # suppress individual ACKs
                pass
        
        # Create pending ACK
        pending = PendingACK(
            seq_num=seq_num,
            timestamp=current_time,
            is_duplicate=False
        )
        
        if self.enable_delayed_ack:
            # Add to pending queue
            self.pending_acks.append(pending)
            self.acks_delayed += 1
            
            # Check if we should send immediately
            if len(self.pending_acks) >= self.max_pending:
                self._send_accumulated_acks(current_time)
        else:
            # Send immediately
            self._send_ack(seq_num)
    
    def check_delayed_acks(self, current_time: float) -> List[int]:
        """
        Check for ACKs that have waited too long.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of ACK numbers to send
        """
        acks_to_send = []
        
        while self.pending_acks:
            oldest = self.pending_acks[0]
            
            if current_time - oldest.timestamp >= self.delay_time:
                # ACK has waited long enough
                self.pending_acks.popleft()
                
                if self.enable_cumulative:
                    # Send cumulative ACK for highest in-order
                    if oldest.seq_num == self.highest_in_order:
                        acks_to_send.append(self.highest_in_order)
                    else:
                        self.acks_suppressed += 1
                else:
                    acks_to_send.append(oldest.seq_num)
            else:
                break
        
        for ack in acks_to_send:
            self._send_ack(ack)
        
        return acks_to_send
    
    def get_piggybacked_ack(self) -> Optional[int]:
        """
        Get ACK to piggyback on outgoing data.
        
        Returns:
            ACK number to piggyback, or None
        """
        if not self.enable_piggybacking:
            return None
        
        if not self.pending_acks:
            return None
        
        # Clear all pending and return cumulative ACK
        self.pending_acks.clear()
        self.acks_piggybacked += 1
        
        return self.highest_in_order
    
    def _send_accumulated_acks(self, current_time: float):
        """Send accumulated pending ACKs."""
        if not self.pending_acks:
            return
        
        if self.enable_cumulative:
            # Send single cumulative ACK
            self.pending_acks.clear()
            self._send_ack(self.highest_in_order)
        else:
            # Send all pending
            while self.pending_acks:
                pending = self.pending_acks.popleft()
                self._send_ack(pending.seq_num)
    
    def _send_ack(self, ack_num: int):
        """Send an ACK."""
        self.acks_sent += 1
        
        if self.on_ack_ready:
            self.on_ack_ready(ack_num)
    
    def get_next_ack_deadline(self) -> Optional[float]:
        """
        Get deadline for next delayed ACK.
        
        Returns:
            Time when oldest pending ACK should be sent
        """
        if not self.pending_acks:
            return None
        
        oldest = self.pending_acks[0]
        return oldest.timestamp + self.delay_time
    
    def get_statistics(self) -> dict:
        """Get optimizer statistics."""
        return {
            'acks_sent': self.acks_sent,
            'acks_delayed': self.acks_delayed,
            'acks_piggybacked': self.acks_piggybacked,
            'acks_suppressed': self.acks_suppressed,
            'pending_acks': len(self.pending_acks),
            'highest_in_order': self.highest_in_order,
            'ack_reduction_rate': (
                self.acks_suppressed / (self.acks_sent + self.acks_suppressed)
                if (self.acks_sent + self.acks_suppressed) > 0 else 0
            )
        }
    
    def reset(self):
        """Reset optimizer state."""
        self.pending_acks.clear()
        self.highest_in_order = -1
        self.acks_delayed = 0
        self.acks_piggybacked = 0
        self.acks_sent = 0
        self.acks_suppressed = 0


class SillyWindowSyndromePreventor:
    """
    Prevents Silly Window Syndrome (SWS).
    
    SWS occurs when small amounts of data are sent, creating
    high overhead. This class ensures minimum segment sizes.
    """
    
    def __init__(
        self,
        mss: int = 1024,            # Maximum Segment Size
        min_send_threshold: float = 0.5  # Min fraction of MSS to send
    ):
        """
        Initialize SWS preventor.
        
        Args:
            mss: Maximum Segment Size
            min_send_threshold: Minimum fraction of MSS to send
        """
        self.mss = mss
        self.min_send_threshold = min_send_threshold
        self.min_send_size = int(mss * min_send_threshold)
        
        # Buffer for small data
        self.buffer = bytearray()
        
        # Statistics
        self.sends_delayed = 0
        self.sends_combined = 0
    
    def should_send(self, data_size: int, window_available: int) -> bool:
        """
        Check if data should be sent or buffered.
        
        Args:
            data_size: Size of data to send
            window_available: Available window space
            
        Returns:
            True if should send now
        """
        # Always send if we have a full MSS
        if data_size >= self.mss:
            return True
        
        # Always send if window is small (receiver constrained)
        if window_available <= self.mss:
            return True
        
        # Don't send small amounts if window is large
        if data_size < self.min_send_size:
            self.sends_delayed += 1
            return False
        
        return True
    
    def buffer_data(self, data: bytes):
        """Buffer small data for later sending."""
        self.buffer.extend(data)
    
    def get_buffered_data(self) -> bytes:
        """Get all buffered data and clear buffer."""
        data = bytes(self.buffer)
        self.buffer.clear()
        self.sends_combined += 1
        return data
    
    def has_buffered_data(self) -> bool:
        """Check if there's buffered data."""
        return len(self.buffer) > 0
    
    def get_statistics(self) -> dict:
        """Get statistics."""
        return {
            'sends_delayed': self.sends_delayed,
            'sends_combined': self.sends_combined,
            'current_buffer_size': len(self.buffer)
        }


if __name__ == "__main__":
    # Test ACK optimizer
    print("=" * 60)
    print("ACK OPTIMIZER TEST")
    print("=" * 60)
    
    acks_sent = []
    
    def on_ack(ack_num):
        acks_sent.append(ack_num)
        print(f"  -> ACK {ack_num} sent")
    
    optimizer = ACKOptimizer(
        delay_time=0.040,
        max_pending=2,
        enable_delayed_ack=True,
        enable_cumulative=True,
        on_ack_ready=on_ack
    )
    
    # Simulate receiving frames
    print("\nReceiving frames in order:")
    current_time = 0.0
    
    for i in range(5):
        print(f"\nTime {current_time:.3f}s: Receive frame {i}")
        optimizer.receive_data(i, current_time, is_in_order=True)
        current_time += 0.010  # 10ms between frames
    
    # Check delayed ACKs
    print(f"\nTime {current_time:.3f}s: Checking delayed ACKs")
    current_time += 0.040  # Wait for delay
    optimizer.check_delayed_acks(current_time)
    
    print(f"\nStatistics: {optimizer.get_statistics()}")
    print(f"ACKs actually sent: {acks_sent}")
