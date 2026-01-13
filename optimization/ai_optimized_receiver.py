"""
AI-Optimized Receiver for Selective Repeat ARQ

This module implements an enhanced receiver that uses AI-based optimizations:

OPTIMIZATIONS IMPLEMENTED:
========================

1. DELAYED ACKs
   - Problem: Sending ACK for every frame creates overhead (many small ACK packets)
   - Solution: Wait briefly (up to 50ms) before ACKing to batch multiple ACKs
   - Benefit: Reduces ACK traffic by ~50%, less reverse channel usage

2. CUMULATIVE ACKs
   - Problem: Individual ACKs waste bandwidth when many frames arrive in order
   - Solution: Single cumulative ACK covers all frames up to ACK number
   - Benefit: One ACK can acknowledge multiple frames

3. SELECTIVE NAK (Negative Acknowledgment)
   - Problem: Sender must wait for timeout to detect lost frames
   - Solution: Send NAK immediately when out-of-order frame detected
   - Benefit: Triggers fast retransmit, faster recovery

4. INTELLIGENT ACK TIMING
   - Problem: Fixed delay may not suit all conditions
   - Solution: Adapt ACK delay based on observed traffic pattern
     * High traffic: Shorter delay (more frames to piggyback)
     * Low traffic: Longer delay (batch more ACKs)
   - Benefit: Optimal ACK timing for current conditions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from collections import deque

from src.arq.frame import Frame, FrameType
from src.arq.receiver import SRReceiver
from optimization.ack_optimization import ACKOptimizer


@dataclass
class ReceiverOptStats:
    """Statistics for receiver optimizations."""
    acks_delayed: int = 0
    acks_batched: int = 0
    naks_sent: int = 0
    cumulative_acks: int = 0
    individual_acks: int = 0
    ack_overhead_saved: int = 0  # Bytes saved by batching


class AIOptimizedReceiver(SRReceiver):
    """
    AI-Optimized Selective Repeat ARQ Receiver.
    
    Inherits from SRReceiver and adds:
    - Delayed ACK generation
    - Cumulative ACK support  
    - Selective NAK for out-of-order detection
    - Intelligent ACK timing
    """
    
    def __init__(
        self,
        window_size: int = 8,
        on_data_delivered: Optional[Callable[[bytes, int], None]] = None,
        # Optimization parameters
        enable_delayed_ack: bool = True,
        enable_cumulative_ack: bool = True,
        enable_nak: bool = True,
        # Delayed ACK params
        ack_delay: float = 0.040,     # Max ACK delay (40ms)
        max_pending_acks: int = 4,    # Max frames before forcing ACK
        # Adaptive ACK params
        enable_adaptive_delay: bool = True,
        min_ack_delay: float = 0.010,  # Minimum delay (10ms)
        max_ack_delay: float = 0.100   # Maximum delay (100ms)
    ):
        """
        Initialize AI-optimized receiver.
        
        Args:
            window_size: Receive window size
            on_data_delivered: Callback when data is delivered in order
            enable_delayed_ack: Enable delayed ACKs
            enable_cumulative_ack: Enable cumulative ACKs
            enable_nak: Enable NAK for out-of-order frames
        """
        super().__init__(window_size=window_size, on_data_delivered=on_data_delivered)
        
        # Optimization flags
        self.enable_delayed_ack = enable_delayed_ack
        self.enable_cumulative_ack = enable_cumulative_ack
        self.enable_nak = enable_nak
        self.enable_adaptive_delay = enable_adaptive_delay
        
        # ACK delay parameters
        self.ack_delay = ack_delay
        self.max_pending_acks = max_pending_acks
        self.min_ack_delay = min_ack_delay
        self.max_ack_delay = max_ack_delay
        
        # Pending ACKs
        self.pending_acks: List[int] = []
        self.last_ack_time: float = 0.0
        self.oldest_pending_time: float = 0.0
        
        # Traffic tracking for adaptive delay
        self.recent_arrivals: deque = deque(maxlen=20)
        self.current_delay = ack_delay
        
        # Statistics
        self.opt_stats = ReceiverOptStats()
        
        # ACK size (for overhead calculation)
        self.ack_frame_size = 24  # bytes
    
    def receive_frame(
        self, 
        frame: Frame, 
        crc_valid: bool = True,
        current_time: float = 0.0
    ) -> Optional[Frame]:
        """
        Process received frame with optimizations.
        
        Args:
            frame: Received frame
            crc_valid: Whether CRC check passed
            current_time: Current simulation time
            
        Returns:
            ACK/NAK frame to send, or None if delayed
        """
        # Track arrival for adaptive delay
        self.recent_arrivals.append(current_time)
        
        # If corrupted, send NAK immediately (no delay for NAKs)
        if not crc_valid:
            return self._create_nak(frame.seq_num)
        
        # Check if this is an out-of-order frame
        is_in_order = (frame.seq_num == self.window.expected_seq)
        
        # Process frame in base class
        self._process_frame(frame)
        
        # Determine ACK response
        if not is_in_order and self.enable_nak:
            # Send NAK immediately for out-of-order (indicates gap)
            self.opt_stats.naks_sent += 1
            return self._create_nak(self.window.expected_seq)
        
        # Queue ACK (possibly delayed)
        if self.enable_delayed_ack:
            return self._queue_delayed_ack(frame.seq_num, current_time)
        else:
            # Send ACK immediately
            self.opt_stats.individual_acks += 1
            return self._create_ack(frame.seq_num)
    
    def _process_frame(self, frame: Frame):
        """Process a valid frame (buffer or deliver)."""
        seq_num = frame.seq_num
        
        if self.window.is_in_window(seq_num):
            if seq_num == self.window.expected_seq:
                # In-order: deliver this and any buffered consecutive frames
                self._deliver_frame(frame)
                self._deliver_buffered_frames()
            else:
                # Out-of-order: buffer for later
                self.out_of_order_buffer[seq_num] = frame
    
    def _deliver_frame(self, frame: Frame):
        """Deliver frame to upper layer."""
        if self.on_data_delivered:
            self.on_data_delivered(frame.payload, frame.seq_num)
        self.window.advance()
    
    def _deliver_buffered_frames(self):
        """Deliver any consecutive buffered frames."""
        while self.window.expected_seq in self.out_of_order_buffer:
            frame = self.out_of_order_buffer.pop(self.window.expected_seq)
            self._deliver_frame(frame)
    
    def _queue_delayed_ack(self, seq_num: int, current_time: float) -> Optional[Frame]:
        """
        Queue ACK for delayed sending.
        
        Returns ACK frame if delay expired or max pending reached.
        """
        # Track pending ACK
        if seq_num not in self.pending_acks:
            self.pending_acks.append(seq_num)
            self.opt_stats.acks_delayed += 1
            
            if len(self.pending_acks) == 1:
                self.oldest_pending_time = current_time
        
        # Update adaptive delay
        if self.enable_adaptive_delay:
            self._update_adaptive_delay()
        
        # Check if we should send now
        delay_expired = (current_time - self.oldest_pending_time) >= self.current_delay
        max_pending_reached = len(self.pending_acks) >= self.max_pending_acks
        
        if delay_expired or max_pending_reached:
            return self._send_pending_acks(current_time)
        
        return None
    
    def _send_pending_acks(self, current_time: float) -> Optional[Frame]:
        """Send pending ACKs (cumulative if enabled)."""
        if not self.pending_acks:
            return None
        
        if self.enable_cumulative_ack and len(self.pending_acks) > 1:
            # Send cumulative ACK for highest sequence number
            highest_ack = max(self.pending_acks)
            self.opt_stats.cumulative_acks += 1
            self.opt_stats.acks_batched += len(self.pending_acks)
            
            # Calculate overhead saved
            acks_saved = len(self.pending_acks) - 1
            self.opt_stats.ack_overhead_saved += acks_saved * self.ack_frame_size
            
            self.pending_acks.clear()
            self.last_ack_time = current_time
            
            return self._create_ack(highest_ack)
        else:
            # Send individual ACK
            ack_num = self.pending_acks.pop(0)
            self.opt_stats.individual_acks += 1
            self.last_ack_time = current_time
            
            return self._create_ack(ack_num)
    
    def _update_adaptive_delay(self):
        """Update ACK delay based on traffic pattern."""
        if len(self.recent_arrivals) < 2:
            return
        
        # Calculate inter-arrival time
        arrivals = list(self.recent_arrivals)
        inter_arrival_times = [
            arrivals[i+1] - arrivals[i] 
            for i in range(len(arrivals)-1)
        ]
        avg_inter_arrival = sum(inter_arrival_times) / len(inter_arrival_times)
        
        # Adapt delay: shorter for high traffic, longer for low traffic
        if avg_inter_arrival < 0.010:  # High traffic (< 10ms between frames)
            self.current_delay = self.min_ack_delay
        elif avg_inter_arrival > 0.050:  # Low traffic (> 50ms between frames)
            self.current_delay = self.max_ack_delay
        else:
            # Linear interpolation
            ratio = (avg_inter_arrival - 0.010) / 0.040
            self.current_delay = self.min_ack_delay + ratio * (self.max_ack_delay - self.min_ack_delay)
    
    def check_delayed_acks(self, current_time: float) -> Optional[Frame]:
        """Check if any delayed ACKs should be sent now."""
        if not self.pending_acks:
            return None
        
        if (current_time - self.oldest_pending_time) >= self.current_delay:
            return self._send_pending_acks(current_time)
        
        return None
    
    def get_next_ack_deadline(self) -> Optional[float]:
        """Get time when next delayed ACK should be sent."""
        if not self.pending_acks:
            return None
        return self.oldest_pending_time + self.current_delay
    
    def _create_ack(self, ack_num: int) -> Frame:
        """Create ACK frame."""
        return Frame(
            frame_type=FrameType.ACK,
            seq_num=0,
            payload=b'',
            ack_num=ack_num
        )
    
    def _create_nak(self, nak_num: int) -> Frame:
        """Create NAK frame."""
        return Frame(
            frame_type=FrameType.NAK,
            seq_num=0,
            payload=b'',
            ack_num=nak_num
        )
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics."""
        total_acks = self.opt_stats.cumulative_acks + self.opt_stats.individual_acks
        
        return {
            'acks_delayed': self.opt_stats.acks_delayed,
            'acks_batched': self.opt_stats.acks_batched,
            'cumulative_acks': self.opt_stats.cumulative_acks,
            'individual_acks': self.opt_stats.individual_acks,
            'naks_sent': self.opt_stats.naks_sent,
            'ack_overhead_saved_bytes': self.opt_stats.ack_overhead_saved,
            'current_delay_ms': self.current_delay * 1000,
            'pending_acks': len(self.pending_acks),
            'batching_ratio': (self.opt_stats.acks_batched / total_acks 
                             if total_acks > 0 else 0)
        }
    
    def reset(self):
        """Reset receiver state."""
        super().reset()
        self.pending_acks.clear()
        self.recent_arrivals.clear()
        self.opt_stats = ReceiverOptStats()
        self.current_delay = self.ack_delay
        self.last_ack_time = 0.0
        self.oldest_pending_time = 0.0


if __name__ == "__main__":
    print("=" * 60)
    print("AI-OPTIMIZED RECEIVER TEST")
    print("=" * 60)
    
    delivered_data = []
    
    def on_deliver(data, seq):
        delivered_data.append((seq, data))
        print(f"  Delivered seq {seq}: {len(data)} bytes")
    
    receiver = AIOptimizedReceiver(
        window_size=16,
        on_data_delivered=on_deliver,
        enable_delayed_ack=True,
        enable_cumulative_ack=True,
        enable_nak=True
    )
    
    print(f"\nInitial State:")
    print(f"  ACK Delay: {receiver.current_delay*1000:.1f} ms")
    print(f"  Max Pending: {receiver.max_pending_acks}")
    
    # Simulate receiving frames
    print("\nReceiving frames...")
    for i in range(5):
        frame = Frame(
            frame_type=FrameType.DATA,
            seq_num=i,
            payload=f"Data {i}".encode()
        )
        ack = receiver.receive_frame(frame, crc_valid=True, current_time=i*0.01)
        if ack:
            print(f"  Frame {i}: ACK sent immediately")
        else:
            print(f"  Frame {i}: ACK delayed")
    
    print(f"\nOptimization Stats: {receiver.get_optimization_stats()}")
