"""
Selective Repeat ARQ Receiver

This module implements the receiver side of the Selective Repeat ARQ protocol,
including sliding window management, out-of-order buffering, and ACK generation.
"""

from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from collections import OrderedDict
import sys
sys.path.insert(0, '..')

from .frame import Frame, FrameType, FrameBuffer


@dataclass
class ReceiveWindow:
    """
    Sliding window for the receiver.
    
    Attributes:
        base: Base of the window (next expected in-order frame)
        size: Window size
        max_seq: Maximum sequence number (for wrapping)
    """
    base: int = 0
    size: int = 8
    max_seq: int = 2**16  # 16-bit sequence numbers
    
    def in_window(self, seq_num: int) -> bool:
        """Check if sequence number is within the receive window."""
        # Handle wrapping
        if seq_num >= self.base and seq_num < self.base + self.size:
            return True
        return False
    
    def is_before_window(self, seq_num: int) -> bool:
        """Check if sequence number is before (already received)."""
        return seq_num < self.base
    
    def advance_base(self, new_base: int):
        """Advance window base to new position."""
        if new_base > self.base:
            self.base = new_base


class SRReceiver:
    """
    Selective Repeat ARQ Receiver.
    
    Implements the receiver side of SR-ARQ with:
    - Sliding window management
    - Out-of-order frame buffering
    - Selective ACK generation
    - In-order delivery to upper layer
    
    Attributes:
        window_size: Size of the receive window
        window: Receive window state
        buffer: Buffer for out-of-order frames
        delivered: Queue of in-order delivered data
    """
    
    def __init__(
        self,
        window_size: int,
        on_data_delivered: Optional[Callable[[bytes, int], None]] = None,
        on_ack_generated: Optional[Callable[[Frame], None]] = None
    ):
        """
        Initialize SR receiver.
        
        Args:
            window_size: Receive window size
            on_data_delivered: Callback when data is delivered in-order
            on_ack_generated: Callback when ACK is generated
        """
        self.window_size = window_size
        
        # Callbacks
        self.on_data_delivered = on_data_delivered
        self.on_ack_generated = on_ack_generated
        
        # Window and buffer
        self.window = ReceiveWindow(size=window_size)
        self.buffer: dict[int, Frame] = {}
        
        # Delivered data tracking
        self.delivered_data: List[Tuple[int, bytes]] = []
        self.total_delivered_bytes = 0
        
        # Statistics
        self.frames_received = 0
        self.duplicate_frames = 0
        self.out_of_order_frames = 0
        self.acks_sent = 0
        self.naks_sent = 0
        
        # State
        self.last_frame_received = False
        self.transfer_complete = False
        
        # For detecting gaps (NAK generation)
        self.highest_received = -1
    
    def receive_frame(self, frame: Frame, crc_valid: bool = True) -> Optional[Frame]:
        """
        Process a received frame.
        
        Args:
            frame: Received frame
            crc_valid: Whether CRC check passed
            
        Returns:
            ACK or NAK frame to send, or None
        """
        # Ignore frames with CRC errors
        if not crc_valid:
            # Optionally generate NAK
            return self._generate_nak(frame.seq_num)
        
        self.frames_received += 1
        seq_num = frame.seq_num
        
        # Check if frame is before window (duplicate of already delivered)
        if self.window.is_before_window(seq_num):
            self.duplicate_frames += 1
            # Still send ACK for duplicate (sender might not have received our ACK)
            return self._generate_ack(seq_num)
        
        # Check if frame is within window
        if not self.window.in_window(seq_num):
            # Frame is outside window - ignore
            return None
        
        # Check if we already have this frame
        if seq_num in self.buffer:
            self.duplicate_frames += 1
            return self._generate_ack(seq_num)
        
        # Buffer the frame
        self.buffer[seq_num] = frame
        
        # Track out-of-order
        if seq_num != self.window.base:
            self.out_of_order_frames += 1
        
        # Track highest received for gap detection
        if seq_num > self.highest_received:
            self.highest_received = seq_num
        
        # Check for last frame
        if frame.is_last_frame():
            self.last_frame_received = True
        
        # Try to deliver in-order frames
        self._deliver_in_order()
        
        # Generate ACK
        return self._generate_ack(seq_num)
    
    def _deliver_in_order(self):
        """Deliver buffered frames that are now in-order."""
        while self.window.base in self.buffer:
            frame = self.buffer.pop(self.window.base)
            
            # Deliver to upper layer
            if self.on_data_delivered:
                self.on_data_delivered(frame.payload, frame.seq_num)
            
            self.delivered_data.append((frame.seq_num, frame.payload))
            self.total_delivered_bytes += len(frame.payload)
            
            # Check if this was the last frame
            if frame.is_last_frame():
                self.transfer_complete = True
            
            # Advance window
            self.window.advance_base(self.window.base + 1)
    
    def _generate_ack(self, seq_num: int) -> Frame:
        """
        Generate ACK frame.
        
        Args:
            seq_num: Sequence number to acknowledge
            
        Returns:
            ACK frame
        """
        ack_frame = Frame.create_ack_frame(seq_num)
        self.acks_sent += 1
        
        if self.on_ack_generated:
            self.on_ack_generated(ack_frame)
        
        return ack_frame
    
    def _generate_nak(self, seq_num: int) -> Frame:
        """
        Generate NAK frame.
        
        Args:
            seq_num: Sequence number to negatively acknowledge
            
        Returns:
            NAK frame
        """
        nak_frame = Frame.create_nak_frame(seq_num)
        self.naks_sent += 1
        
        if self.on_ack_generated:
            self.on_ack_generated(nak_frame)
        
        return nak_frame
    
    def get_missing_frames(self) -> List[int]:
        """
        Get list of missing frame sequence numbers.
        
        Returns:
            List of sequence numbers for frames not yet received
        """
        missing = []
        for seq_num in range(self.window.base, self.highest_received + 1):
            if seq_num not in self.buffer and seq_num >= self.window.base:
                missing.append(seq_num)
        return missing
    
    def generate_selective_naks(self) -> List[Frame]:
        """
        Generate NAKs for all missing frames.
        
        Returns:
            List of NAK frames
        """
        missing = self.get_missing_frames()
        return [self._generate_nak(seq) for seq in missing]
    
    def get_delivered_data(self, clear: bool = True) -> List[Tuple[int, bytes]]:
        """
        Get all delivered data.
        
        Args:
            clear: Whether to clear the delivered data list
            
        Returns:
            List of (sequence_number, data) tuples
        """
        data = self.delivered_data.copy()
        if clear:
            self.delivered_data.clear()
        return data
    
    def get_all_delivered_bytes(self) -> bytes:
        """
        Get all delivered data as a single bytes object.
        
        Returns:
            Concatenated delivered data
        """
        return b''.join(data for _, data in sorted(
            self.delivered_data, key=lambda x: x[0]
        ))
    
    def is_complete(self) -> bool:
        """Check if transfer is complete."""
        return self.transfer_complete
    
    def get_window_state(self) -> dict:
        """Get current window state."""
        return {
            'base': self.window.base,
            'size': self.window.size,
            'buffered_frames': sorted(self.buffer.keys()),
            'highest_received': self.highest_received,
            'missing_frames': self.get_missing_frames()
        }
    
    def get_statistics(self) -> dict:
        """Get receiver statistics."""
        return {
            'frames_received': self.frames_received,
            'duplicate_frames': self.duplicate_frames,
            'out_of_order_frames': self.out_of_order_frames,
            'acks_sent': self.acks_sent,
            'naks_sent': self.naks_sent,
            'total_delivered_bytes': self.total_delivered_bytes,
            'transfer_complete': self.transfer_complete
        }
    
    def reset(self):
        """Reset receiver to initial state."""
        self.window = ReceiveWindow(size=self.window_size)
        self.buffer.clear()
        self.delivered_data.clear()
        self.total_delivered_bytes = 0
        
        self.frames_received = 0
        self.duplicate_frames = 0
        self.out_of_order_frames = 0
        self.acks_sent = 0
        self.naks_sent = 0
        
        self.last_frame_received = False
        self.transfer_complete = False
        self.highest_received = -1


class CumulativeACKReceiver(SRReceiver):
    """
    Variant that generates cumulative ACKs instead of selective ACKs.
    
    The cumulative ACK acknowledges all frames up to and including
    the highest in-order received frame.
    """
    
    def _generate_ack(self, seq_num: int) -> Frame:
        """
        Generate cumulative ACK.
        
        Acknowledges the highest in-order frame (window base - 1).
        """
        # Cumulative ACK is for highest in-order frame
        cumulative_ack = self.window.base - 1
        if cumulative_ack < 0:
            cumulative_ack = seq_num  # First frame
        
        ack_frame = Frame.create_ack_frame(cumulative_ack)
        self.acks_sent += 1
        
        if self.on_ack_generated:
            self.on_ack_generated(ack_frame)
        
        return ack_frame


if __name__ == "__main__":
    # Test SR receiver
    print("=" * 60)
    print("SELECTIVE REPEAT RECEIVER TEST")
    print("=" * 60)
    
    def on_delivered(data, seq):
        print(f"  [DELIVERED] Frame {seq}: {data.decode()}")
    
    def on_ack(frame):
        print(f"  [ACK] Generated: {frame}")
    
    receiver = SRReceiver(
        window_size=4,
        on_data_delivered=on_delivered,
        on_ack_generated=on_ack
    )
    
    # Create test frames (out of order)
    print("\nReceiving frames out of order: 0, 2, 1, 3")
    
    frames = [
        Frame.create_data_frame(0, b"Frame 0"),
        Frame.create_data_frame(2, b"Frame 2"),
        Frame.create_data_frame(1, b"Frame 1"),
        Frame.create_data_frame(3, b"Frame 3", is_last=True),
    ]
    
    for frame in frames:
        print(f"\nReceiving: {frame}")
        ack = receiver.receive_frame(frame)
        print(f"  Window state: {receiver.get_window_state()}")
    
    # Test duplicate
    print("\n\nReceiving duplicate frame 1:")
    dup_frame = Frame.create_data_frame(1, b"Frame 1")
    ack = receiver.receive_frame(dup_frame)
    
    print(f"\nFinal statistics: {receiver.get_statistics()}")
    print(f"Transfer complete: {receiver.is_complete()}")
