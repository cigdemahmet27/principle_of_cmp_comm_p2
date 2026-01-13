"""
Buffer Management

This module provides buffer implementations for the transport layer,
including the receiver buffer with backpressure support.
"""

from typing import Optional, Tuple, List, Callable
from collections import deque
from dataclasses import dataclass
import sys
sys.path.insert(0, '..')
from config import RECEIVER_BUFFER_SIZE


@dataclass
class BufferSegment:
    """Segment stored in buffer."""
    sequence_num: int
    data: bytes
    timestamp: float = 0.0


class ReceiveBuffer:
    """
    Fixed-size receiver buffer with backpressure support.
    
    Implements the 256 KB application-side buffer capacity requirement.
    When full, signals backpressure to the link layer.
    
    Attributes:
        capacity: Maximum buffer size in bytes
        current_size: Current used size in bytes
        buffer: Queue of buffered segments
    """
    
    def __init__(
        self,
        capacity: int = RECEIVER_BUFFER_SIZE,
        on_backpressure: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize receive buffer.
        
        Args:
            capacity: Maximum buffer size in bytes (default 256 KB)
            on_backpressure: Callback when backpressure state changes
        """
        self.capacity = capacity
        self.on_backpressure = on_backpressure
        
        self.buffer: deque[BufferSegment] = deque()
        self.current_size = 0
        self.backpressure_active = False
        
        # High/low watermarks for hysteresis
        self.high_watermark = int(capacity * 0.9)  # 90%
        self.low_watermark = int(capacity * 0.5)   # 50%
        
        # Statistics
        self.total_bytes_buffered = 0
        self.total_segments_buffered = 0
        self.overflow_events = 0
        self.backpressure_events = 0
    
    @property
    def available_space(self) -> int:
        """Get available space in bytes."""
        return self.capacity - self.current_size
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.current_size >= self.capacity
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    @property
    def fill_level(self) -> float:
        """Get buffer fill level (0-1)."""
        return self.current_size / self.capacity if self.capacity > 0 else 0
    
    def can_accept(self, size: int) -> bool:
        """
        Check if buffer can accept data of given size.
        
        Args:
            size: Size of data in bytes
            
        Returns:
            True if data can be accepted
        """
        return self.current_size + size <= self.capacity
    
    def add(
        self,
        data: bytes,
        sequence_num: int = 0,
        timestamp: float = 0.0
    ) -> bool:
        """
        Add data to buffer.
        
        Args:
            data: Data bytes to buffer
            sequence_num: Sequence number for ordering
            timestamp: Timestamp of data
            
        Returns:
            True if data was added, False if buffer full
        """
        size = len(data)
        
        if not self.can_accept(size):
            self.overflow_events += 1
            self._check_backpressure()
            return False
        
        segment = BufferSegment(
            sequence_num=sequence_num,
            data=data,
            timestamp=timestamp
        )
        
        self.buffer.append(segment)
        self.current_size += size
        self.total_bytes_buffered += size
        self.total_segments_buffered += 1
        
        self._check_backpressure()
        return True
    
    def get(self, max_size: Optional[int] = None) -> Optional[BufferSegment]:
        """
        Get next segment from buffer.
        
        Args:
            max_size: Maximum size to return (optional)
            
        Returns:
            BufferSegment or None if empty
        """
        if self.is_empty:
            return None
        
        segment = self.buffer.popleft()
        self.current_size -= len(segment.data)
        
        self._check_backpressure()
        return segment
    
    def peek(self) -> Optional[BufferSegment]:
        """
        Peek at next segment without removing.
        
        Returns:
            BufferSegment or None if empty
        """
        if self.is_empty:
            return None
        return self.buffer[0]
    
    def get_all(self) -> List[BufferSegment]:
        """
        Get all segments from buffer.
        
        Returns:
            List of all buffered segments
        """
        segments = list(self.buffer)
        total_size = sum(len(s.data) for s in segments)
        
        self.buffer.clear()
        self.current_size = 0
        
        self._check_backpressure()
        return segments
    
    def drain(self, max_bytes: int) -> List[BufferSegment]:
        """
        Drain up to max_bytes from buffer.
        
        Args:
            max_bytes: Maximum bytes to drain
            
        Returns:
            List of drained segments
        """
        segments = []
        bytes_drained = 0
        
        while not self.is_empty and bytes_drained < max_bytes:
            segment = self.peek()
            if segment and bytes_drained + len(segment.data) <= max_bytes:
                segments.append(self.get())
                bytes_drained += len(segment.data)
            else:
                break
        
        return segments
    
    def _check_backpressure(self):
        """Check and update backpressure state."""
        new_state = False
        
        if self.current_size >= self.high_watermark:
            new_state = True
        elif self.current_size <= self.low_watermark:
            new_state = False
        else:
            # In hysteresis zone, maintain current state
            new_state = self.backpressure_active
        
        if new_state != self.backpressure_active:
            self.backpressure_active = new_state
            if new_state:
                self.backpressure_events += 1
            
            if self.on_backpressure:
                self.on_backpressure(new_state)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.current_size = 0
        self._check_backpressure()
    
    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        return {
            'capacity': self.capacity,
            'current_size': self.current_size,
            'available_space': self.available_space,
            'fill_level': self.fill_level,
            'segments_buffered': len(self.buffer),
            'backpressure_active': self.backpressure_active,
            'total_bytes_buffered': self.total_bytes_buffered,
            'total_segments_buffered': self.total_segments_buffered,
            'overflow_events': self.overflow_events,
            'backpressure_events': self.backpressure_events
        }


class SendBuffer:
    """
    Send buffer for holding data from application layer.
    
    Data is queued here before being segmented and sent.
    """
    
    def __init__(self, max_segments: int = 1000):
        """
        Initialize send buffer.
        
        Args:
            max_segments: Maximum number of segments to buffer
        """
        self.max_segments = max_segments
        self.buffer: deque[bytes] = deque()
        self.total_bytes = 0
    
    def add(self, data: bytes) -> bool:
        """
        Add data to send buffer.
        
        Args:
            data: Data to buffer
            
        Returns:
            True if added, False if full
        """
        if len(self.buffer) >= self.max_segments:
            return False
        
        self.buffer.append(data)
        self.total_bytes += len(data)
        return True
    
    def get(self) -> Optional[bytes]:
        """
        Get next data from buffer.
        
        Returns:
            Data bytes or None if empty
        """
        if not self.buffer:
            return None
        
        data = self.buffer.popleft()
        self.total_bytes -= len(data)
        return data
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    @property
    def count(self) -> int:
        """Get number of segments in buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.total_bytes = 0


class CircularBuffer:
    """
    Circular buffer for sequence number management.
    
    Used for tracking acknowledged sequence numbers in sliding window.
    """
    
    def __init__(self, size: int):
        """
        Initialize circular buffer.
        
        Args:
            size: Buffer size (typically window size)
        """
        self.size = size
        self.buffer = [False] * size
        self.base = 0
    
    def mark(self, seq_num: int):
        """Mark a sequence number as received/acknowledged."""
        index = seq_num % self.size
        self.buffer[index] = True
    
    def is_marked(self, seq_num: int) -> bool:
        """Check if sequence number is marked."""
        index = seq_num % self.size
        return self.buffer[index]
    
    def advance_base(self):
        """Advance base while marked, clear old marks."""
        while self.is_marked(self.base):
            index = self.base % self.size
            self.buffer[index] = False
            self.base += 1
    
    def get_base(self) -> int:
        """Get current base."""
        return self.base
    
    def clear(self):
        """Clear all marks."""
        self.buffer = [False] * self.size
        self.base = 0


if __name__ == "__main__":
    # Test receive buffer
    print("=" * 60)
    print("RECEIVE BUFFER TEST")
    print("=" * 60)
    
    def on_bp_change(active):
        print(f"  [BACKPRESSURE] {'ACTIVE' if active else 'RELEASED'}")
    
    # Small buffer for testing
    buffer = ReceiveBuffer(capacity=1024, on_backpressure=on_bp_change)
    
    print(f"\nBuffer capacity: {buffer.capacity} bytes")
    print(f"High watermark: {buffer.high_watermark} bytes")
    print(f"Low watermark: {buffer.low_watermark} bytes")
    
    # Fill buffer
    print("\nFilling buffer...")
    for i in range(20):
        data = bytes([i] * 100)
        result = buffer.add(data, sequence_num=i)
        fill = buffer.fill_level * 100
        print(f"  Add segment {i}: {'OK' if result else 'FULL'}, "
              f"fill: {fill:.1f}%")
    
    print(f"\nBuffer stats: {buffer.get_statistics()}")
    
    # Drain buffer
    print("\nDraining buffer...")
    while not buffer.is_empty:
        segment = buffer.get()
        if segment:
            print(f"  Got segment {segment.sequence_num}, "
                  f"fill: {buffer.fill_level*100:.1f}%")
    
    print(f"\nFinal stats: {buffer.get_statistics()}")
