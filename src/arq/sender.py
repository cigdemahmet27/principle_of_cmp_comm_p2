"""
Selective Repeat ARQ Sender

This module implements the sender side of the Selective Repeat ARQ protocol,
including sliding window management, frame buffering, and retransmission.
"""

from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import sys
sys.path.insert(0, '..')

from .frame import Frame, FrameType, FrameBuffer
from .timer import TimerManager, AdaptiveTimeoutCalculator


@dataclass
class SendWindow:
    """
    Sliding window for the sender.
    
    Attributes:
        base: Base of the window (oldest unacknowledged frame)
        next_seq: Next sequence number to use
        size: Window size
        max_seq: Maximum sequence number (for wrapping)
    """
    base: int = 0
    next_seq: int = 0
    size: int = 8
    max_seq: int = 2**16  # 16-bit sequence numbers
    
    @property
    def available_slots(self) -> int:
        """Number of available slots in the window."""
        return self.size - (self.next_seq - self.base)
    
    @property
    def is_full(self) -> bool:
        """Check if window is full."""
        return self.available_slots <= 0
    
    def in_window(self, seq_num: int) -> bool:
        """Check if sequence number is within the window."""
        return self.base <= seq_num < self.base + self.size
    
    def advance_base(self, new_base: int):
        """Advance window base to new position."""
        if new_base > self.base:
            self.base = new_base
    
    def get_next_seq(self) -> int:
        """Get next sequence number and increment counter."""
        seq = self.next_seq % self.max_seq
        self.next_seq += 1
        return seq


class SRSender:
    """
    Selective Repeat ARQ Sender.
    
    Implements the sender side of SR-ARQ with:
    - Sliding window management
    - Per-frame timers
    - Selective retransmission
    - Frame buffering for retransmission
    
    Attributes:
        window_size: Size of the send window
        timeout: Frame timeout in seconds
        window: Send window state
        buffer: Buffer for unacknowledged frames
        timer_manager: Per-frame timer manager
    """
    
    def __init__(
        self,
        window_size: int,
        timeout: float,
        max_retransmissions: int = 10,
        adaptive_timeout: bool = False,
        on_frame_sent: Optional[Callable[[Frame], None]] = None,
        on_timeout: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize SR sender.
        
        Args:
            window_size: Send window size
            timeout: Initial timeout in seconds
            max_retransmissions: Maximum retransmission attempts
            adaptive_timeout: Use adaptive timeout calculation
            on_frame_sent: Callback when frame is sent
            on_timeout: Callback when timeout occurs
        """
        self.window_size = window_size
        self.initial_timeout = timeout
        self.max_retransmissions = max_retransmissions
        self.adaptive_timeout = adaptive_timeout
        
        # Callbacks
        self.on_frame_sent = on_frame_sent
        self.on_timeout = on_timeout
        
        # Window and buffer
        self.window = SendWindow(size=window_size)
        self.buffer = FrameBuffer(max_size=window_size)
        
        # Timer management
        self.timer_manager = TimerManager(
            default_timeout=timeout,
            max_retransmissions=max_retransmissions
        )
        
        # Adaptive timeout calculator
        self.timeout_calculator = AdaptiveTimeoutCalculator(
            initial_rto=timeout
        ) if adaptive_timeout else None
        
        # Pending data queue
        self.pending_data: deque = deque()
        
        # Statistics
        self.frames_sent = 0
        self.frames_acked = 0
        self.retransmissions = 0
        self.total_bytes_sent = 0
        
        # Tracking for RTT measurement
        self.send_times: dict[int, float] = {}
        
        # State
        self.finished = False
        self.all_acked = False
    
    def queue_data(self, data: bytes, is_last: bool = False):
        """
        Queue data for transmission.
        
        Args:
            data: Data bytes to send
            is_last: Whether this is the last piece of data
        """
        self.pending_data.append((data, is_last))
    
    def can_send(self) -> bool:
        """Check if sender can transmit a new frame."""
        return (not self.window.is_full and 
                len(self.pending_data) > 0 and
                not self.finished)
    
    def send_next_frame(self, current_time: float) -> Optional[Frame]:
        """
        Create and send the next frame.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Frame to transmit or None
        """
        if not self.can_send():
            return None
        
        # Get next data from queue
        data, is_last = self.pending_data.popleft()
        
        # Get sequence number
        seq_num = self.window.get_next_seq()
        
        # Create frame
        frame = Frame.create_data_frame(
            seq_num=seq_num,
            payload=data,
            is_last=is_last
        )
        
        # Buffer frame for potential retransmission
        self.buffer.add(frame)
        
        # Start timer
        timeout = self._get_timeout()
        self.timer_manager.start_timer(seq_num, current_time, timeout)
        
        # Track send time for RTT measurement
        self.send_times[seq_num] = current_time
        
        # Update statistics
        self.frames_sent += 1
        self.total_bytes_sent += frame.total_size
        
        # Mark as finished if last frame sent
        if is_last:
            self.finished = True
        
        # Callback
        if self.on_frame_sent:
            self.on_frame_sent(frame)
        
        return frame
    
    def retransmit_frame(self, seq_num: int, current_time: float) -> Optional[Frame]:
        """
        Retransmit a specific frame.
        
        Args:
            seq_num: Sequence number to retransmit
            current_time: Current simulation time
            
        Returns:
            Frame to retransmit or None
        """
        frame = self.buffer.get(seq_num)
        if frame is None:
            return None
        
        # Check max retransmissions
        if self.timer_manager.is_max_retransmissions(seq_num):
            return None
        
        # Mark as retransmission
        frame.set_retransmission()
        
        # Restart timer
        timeout = self._get_timeout()
        self.timer_manager.start_timer(seq_num, current_time, timeout)
        
        # Update send time
        self.send_times[seq_num] = current_time
        
        # Update statistics
        self.retransmissions += 1
        self.total_bytes_sent += frame.total_size
        
        # Callback
        if self.on_frame_sent:
            self.on_frame_sent(frame)
        
        return frame
    
    def process_ack(self, ack_num: int, current_time: float) -> bool:
        """
        Process received ACK.
        
        Args:
            ack_num: Acknowledged sequence number
            current_time: Current simulation time
            
        Returns:
            True if ACK was valid and processed
        """
        # Check if ACK is for frame in window
        if not self.window.in_window(ack_num):
            return False
        
        # Calculate RTT if we have send time
        if ack_num in self.send_times:
            rtt = current_time - self.send_times[ack_num]
            self._update_rtt(rtt, ack_num)
            del self.send_times[ack_num]
        
        # Stop timer
        self.timer_manager.cancel_timer(ack_num)
        
        # Remove from buffer
        self.buffer.remove(ack_num)
        
        # Update statistics
        self.frames_acked += 1
        
        # Slide window
        self._slide_window()
        
        # Check if all frames acknowledged
        if self.finished and self.buffer.is_empty:
            self.all_acked = True
        
        return True
    
    def process_nak(self, nak_num: int, current_time: float) -> Optional[Frame]:
        """
        Process received NAK - trigger immediate retransmission.
        
        Args:
            nak_num: Negative acknowledged sequence number
            current_time: Current simulation time
            
        Returns:
            Frame to retransmit
        """
        return self.retransmit_frame(nak_num, current_time)
    
    def check_timeouts(self, current_time: float) -> List[int]:
        """
        Check for expired timers.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of sequence numbers that timed out
        """
        expired = self.timer_manager.check_timeouts(current_time)
        
        for seq_num in expired:
            if self.on_timeout:
                self.on_timeout(seq_num)
            
            # Apply backoff for adaptive timeout
            if self.timeout_calculator:
                self.timeout_calculator.backoff()
        
        return expired
    
    def get_frames_to_retransmit(self, current_time: float) -> List[Frame]:
        """
        Get all frames that need retransmission due to timeout.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of frames to retransmit
        """
        expired = self.check_timeouts(current_time)
        frames = []
        
        for seq_num in expired:
            frame = self.retransmit_frame(seq_num, current_time)
            if frame:
                frames.append(frame)
        
        return frames
    
    def _slide_window(self):
        """Slide the window forward past consecutive acknowledged frames."""
        while True:
            if self.buffer.contains(self.window.base):
                break
            if self.window.base >= self.window.next_seq:
                break
            self.window.advance_base(self.window.base + 1)
    
    def _get_timeout(self) -> float:
        """Get current timeout value."""
        if self.timeout_calculator:
            return self.timeout_calculator.get_rto()
        return self.timer_manager.default_timeout
    
    def _update_rtt(self, rtt: float, seq_num: int):
        """
        Update RTT estimate.
        
        Args:
            rtt: Measured RTT
            seq_num: Sequence number (not used for retransmits)
        """
        # Don't update RTT for retransmitted frames
        if self.timer_manager.get_retransmit_count(seq_num) > 0:
            return
        
        if self.timeout_calculator:
            self.timeout_calculator.update(rtt)
            # Update timer manager with new timeout
            self.timer_manager.update_timeout(self.timeout_calculator.get_rto())
    
    def get_next_event_time(self) -> Optional[float]:
        """Get time of next timer expiry."""
        return self.timer_manager.get_next_expiry()
    
    def is_complete(self) -> bool:
        """Check if all data has been sent and acknowledged."""
        return self.all_acked
    
    def get_window_state(self) -> dict:
        """Get current window state."""
        return {
            'base': self.window.base,
            'next_seq': self.window.next_seq,
            'size': self.window.size,
            'available': self.window.available_slots,
            'buffered_frames': self.buffer.get_sequence_numbers()
        }
    
    def get_statistics(self) -> dict:
        """Get sender statistics."""
        timer_stats = self.timer_manager.get_statistics()
        
        stats = {
            'frames_sent': self.frames_sent,
            'frames_acked': self.frames_acked,
            'retransmissions': self.retransmissions,
            'total_bytes_sent': self.total_bytes_sent,
            'pending_data': len(self.pending_data),
            'finished': self.finished,
            'all_acked': self.all_acked,
            **timer_stats
        }
        
        if self.timeout_calculator:
            stats['current_rto'] = self.timeout_calculator.get_rto()
            stats['srtt'] = self.timeout_calculator.srtt
        
        return stats
    
    def reset(self):
        """Reset sender to initial state."""
        self.window = SendWindow(size=self.window_size)
        self.buffer = FrameBuffer(max_size=self.window_size)
        self.timer_manager.clear_all()
        self.pending_data.clear()
        self.send_times.clear()
        
        self.frames_sent = 0
        self.frames_acked = 0
        self.retransmissions = 0
        self.total_bytes_sent = 0
        self.finished = False
        self.all_acked = False
        
        if self.timeout_calculator:
            self.timeout_calculator.reset(self.initial_timeout)


if __name__ == "__main__":
    # Test SR sender
    print("=" * 60)
    print("SELECTIVE REPEAT SENDER TEST")
    print("=" * 60)
    
    def on_sent(frame):
        print(f"  [SENT] {frame}")
    
    def on_timeout(seq):
        print(f"  [TIMEOUT] Frame {seq}")
    
    sender = SRSender(
        window_size=4,
        timeout=0.5,
        on_frame_sent=on_sent,
        on_timeout=on_timeout
    )
    
    # Queue some data
    print("\nQueuing data...")
    for i in range(10):
        data = f"Segment {i}".encode()
        sender.queue_data(data, is_last=(i == 9))
    
    # Simulate sending
    current_time = 0.0
    print(f"\nTime {current_time}: Sending frames...")
    
    while sender.can_send():
        frame = sender.send_next_frame(current_time)
    
    print(f"\nWindow state: {sender.get_window_state()}")
    
    # Simulate ACKs
    current_time = 0.1
    print(f"\nTime {current_time}: Processing ACKs for frames 0, 1, 2...")
    for ack in [0, 1, 2]:
        sender.process_ack(ack, current_time)
    
    print(f"Window state: {sender.get_window_state()}")
    
    # Send more frames
    print("\nSending more frames...")
    while sender.can_send():
        frame = sender.send_next_frame(current_time)
    
    print(f"Window state: {sender.get_window_state()}")
    
    # Simulate timeout
    current_time = 0.7
    print(f"\nTime {current_time}: Checking timeouts...")
    timeouts = sender.check_timeouts(current_time)
    print(f"Timed out frames: {timeouts}")
    
    # Get statistics
    print(f"\nStatistics: {sender.get_statistics()}")
