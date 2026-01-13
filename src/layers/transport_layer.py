"""
Transport Layer Implementation

This module implements the transport layer with segmentation,
reassembly, and flow control with backpressure support.
"""

from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import struct
import sys
sys.path.insert(0, '..')

from config import TRANSPORT_HEADER_SIZE, RECEIVER_BUFFER_SIZE
from src.utils.buffer import ReceiveBuffer


@dataclass
class TransportSegment:
    """
    Transport Layer Segment.
    
    Header Layout (8 bytes):
        - Segment Number: 4 bytes (unsigned int)
        - Payload Length: 2 bytes (unsigned short)
        - Flags: 1 byte
        - Checksum: 1 byte (simple XOR checksum)
    
    Total Header: 8 bytes (as specified)
    """
    segment_num: int
    payload: bytes
    flags: int = 0
    
    # Flag definitions
    FLAG_LAST_SEGMENT = 0x01
    FLAG_RETRANSMIT = 0x02
    
    HEADER_SIZE = TRANSPORT_HEADER_SIZE  # 8 bytes
    HEADER_FORMAT = '!IHBB'  # segment_num(4) + length(2) + flags(1) + checksum(1)
    
    def is_last(self) -> bool:
        """Check if this is the last segment."""
        return bool(self.flags & self.FLAG_LAST_SEGMENT)
    
    def set_last(self):
        """Mark as last segment."""
        self.flags |= self.FLAG_LAST_SEGMENT
    
    def calculate_checksum(self) -> int:
        """Calculate simple XOR checksum."""
        checksum = 0
        for byte in self.payload:
            checksum ^= byte
        return checksum & 0xFF
    
    def serialize(self) -> bytes:
        """
        Serialize segment to bytes.
        
        Returns:
            Serialized segment
        """
        checksum = self.calculate_checksum()
        header = struct.pack(
            self.HEADER_FORMAT,
            self.segment_num,
            len(self.payload),
            self.flags,
            checksum
        )
        return header + self.payload
    
    @classmethod
    def deserialize(cls, data: bytes) -> Tuple[Optional['TransportSegment'], bool]:
        """
        Deserialize bytes to segment.
        
        Args:
            data: Serialized segment
            
        Returns:
            Tuple of (segment, checksum_valid)
        """
        if len(data) < cls.HEADER_SIZE:
            return None, False
        
        try:
            header_data = data[:cls.HEADER_SIZE]
            segment_num, length, flags, checksum = struct.unpack(
                cls.HEADER_FORMAT, header_data
            )
            
            payload = data[cls.HEADER_SIZE:cls.HEADER_SIZE + length]
            if len(payload) != length:
                return None, False
            
            segment = cls(
                segment_num=segment_num,
                payload=payload,
                flags=flags
            )
            
            # Verify checksum
            expected_checksum = segment.calculate_checksum()
            checksum_valid = (checksum == expected_checksum)
            
            return segment, checksum_valid
            
        except Exception:
            return None, False


class TransportLayerSender:
    """
    Transport Layer Sender.
    
    Handles segmentation of application data into transport segments.
    
    Attributes:
        segment_size: Maximum payload size per segment
        header_size: Transport header size
    """
    
    def __init__(
        self,
        segment_size: int,
        on_segment_ready: Optional[Callable[[bytes, bool], None]] = None
    ):
        """
        Initialize transport sender.
        
        Args:
            segment_size: Maximum payload size per segment
            on_segment_ready: Callback when segment is ready for link layer
        """
        self.segment_size = segment_size
        self.header_size = TRANSPORT_HEADER_SIZE
        self.on_segment_ready = on_segment_ready
        
        # Segmentation state
        self.next_segment_num = 0
        self.data_buffer = b''
        self.total_bytes_segmented = 0
        self.segments_created = 0
        
        # Transfer state
        self.transfer_started = False
        self.transfer_complete = False
        self.total_data_size = 0
    
    def start_transfer(self, total_size: int):
        """
        Start a new transfer.
        
        Args:
            total_size: Total bytes to transfer
        """
        self.transfer_started = True
        self.transfer_complete = False
        self.total_data_size = total_size
        self.next_segment_num = 0
        self.data_buffer = b''
        self.total_bytes_segmented = 0
        self.segments_created = 0
    
    def add_data(self, data: bytes):
        """
        Add data to be segmented.
        
        Args:
            data: Data bytes from application
        """
        self.data_buffer += data
    
    def get_next_segment(self) -> Optional[Tuple[bytes, bool]]:
        """
        Get next segment for transmission.
        
        Returns:
            Tuple of (segment_bytes, is_last) or None if no data
        """
        if len(self.data_buffer) == 0:
            return None
        
        # Determine segment size
        payload_size = min(self.segment_size, len(self.data_buffer))
        payload = self.data_buffer[:payload_size]
        self.data_buffer = self.data_buffer[payload_size:]
        
        # Check if this is the last segment
        is_last = (self.total_bytes_segmented + payload_size >= self.total_data_size 
                   and len(self.data_buffer) == 0)
        
        # Create segment
        segment = TransportSegment(
            segment_num=self.next_segment_num,
            payload=payload
        )
        
        if is_last:
            segment.set_last()
            self.transfer_complete = True
        
        # Serialize
        segment_bytes = segment.serialize()
        
        # Update state
        self.next_segment_num += 1
        self.total_bytes_segmented += payload_size
        self.segments_created += 1
        
        # Callback
        if self.on_segment_ready:
            self.on_segment_ready(segment_bytes, is_last)
        
        return segment_bytes, is_last
    
    def has_data(self) -> bool:
        """Check if there's data to segment."""
        return len(self.data_buffer) > 0
    
    def get_progress(self) -> float:
        """Get transfer progress (0-1)."""
        if self.total_data_size == 0:
            return 0.0
        return self.total_bytes_segmented / self.total_data_size
    
    def get_statistics(self) -> dict:
        """Get sender statistics."""
        return {
            'segments_created': self.segments_created,
            'bytes_segmented': self.total_bytes_segmented,
            'total_size': self.total_data_size,
            'progress': self.get_progress(),
            'transfer_complete': self.transfer_complete,
            'buffer_size': len(self.data_buffer)
        }


class TransportLayerReceiver:
    """
    Transport Layer Receiver.
    
    Handles reassembly of transport segments and flow control.
    
    Attributes:
        buffer: Receive buffer with backpressure
        expected_segment: Next expected segment number
    """
    
    def __init__(
        self,
        buffer_capacity: int = RECEIVER_BUFFER_SIZE,
        on_data_ready: Optional[Callable[[bytes], None]] = None,
        on_transfer_complete: Optional[Callable[[], None]] = None,
        on_backpressure: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize transport receiver.
        
        Args:
            buffer_capacity: Receive buffer capacity in bytes
            on_data_ready: Callback when data is ready for application
            on_transfer_complete: Callback when transfer is complete
            on_backpressure: Callback when backpressure state changes
        """
        self.buffer_capacity = buffer_capacity
        self.on_data_ready = on_data_ready
        self.on_transfer_complete = on_transfer_complete
        self.on_backpressure = on_backpressure
        
        # Receive buffer
        self.buffer = ReceiveBuffer(
            capacity=buffer_capacity,
            on_backpressure=self._handle_backpressure
        )
        
        # Reassembly state
        self.expected_segment = 0
        self.out_of_order_buffer: dict[int, bytes] = {}
        
        # Transfer state
        self.transfer_complete = False
        self.last_segment_received = False
        self.total_bytes_received = 0
        self.segments_received = 0
    
    def receive_segment(self, segment_data: bytes) -> bool:
        """
        Receive a transport segment.
        
        Args:
            segment_data: Raw segment bytes
            
        Returns:
            True if segment was processed successfully
        """
        # Deserialize
        segment, valid = TransportSegment.deserialize(segment_data)
        if segment is None or not valid:
            return False
        
        self.segments_received += 1
        
        # Check if this is the last segment
        if segment.is_last():
            self.last_segment_received = True
        
        # Check if in-order
        if segment.segment_num == self.expected_segment:
            # Deliver this segment
            self._deliver_segment(segment)
            self.expected_segment += 1
            
            # Deliver any buffered out-of-order segments
            self._deliver_buffered()
            
        elif segment.segment_num > self.expected_segment:
            # Out of order - buffer it
            self.out_of_order_buffer[segment.segment_num] = segment.payload
        
        # Else: duplicate, ignore
        
        # Check transfer complete
        if self.last_segment_received and len(self.out_of_order_buffer) == 0:
            self.transfer_complete = True
            if self.on_transfer_complete:
                self.on_transfer_complete()
        
        return True
    
    def _deliver_segment(self, segment: TransportSegment):
        """Deliver a segment to the buffer."""
        if self.buffer.add(segment.payload, segment.segment_num):
            self.total_bytes_received += len(segment.payload)
            
            if self.on_data_ready:
                self.on_data_ready(segment.payload)
    
    def _deliver_buffered(self):
        """Deliver buffered out-of-order segments that are now in-order."""
        while self.expected_segment in self.out_of_order_buffer:
            payload = self.out_of_order_buffer.pop(self.expected_segment)
            
            # Create temporary segment for delivery
            segment = TransportSegment(
                segment_num=self.expected_segment,
                payload=payload
            )
            self._deliver_segment(segment)
            self.expected_segment += 1
    
    def _handle_backpressure(self, active: bool):
        """Handle backpressure signal."""
        if self.on_backpressure:
            self.on_backpressure(active)
    
    def get_data(self, max_bytes: Optional[int] = None) -> bytes:
        """
        Get received data from buffer.
        
        Args:
            max_bytes: Maximum bytes to retrieve
            
        Returns:
            Data bytes
        """
        if max_bytes is None:
            segments = self.buffer.get_all()
        else:
            segments = self.buffer.drain(max_bytes)
        
        return b''.join(s.data for s in segments)
    
    def is_backpressure_active(self) -> bool:
        """Check if backpressure is active."""
        return self.buffer.backpressure_active
    
    def is_complete(self) -> bool:
        """Check if transfer is complete."""
        return self.transfer_complete
    
    def get_statistics(self) -> dict:
        """Get receiver statistics."""
        buffer_stats = self.buffer.get_statistics()
        
        return {
            'segments_received': self.segments_received,
            'bytes_received': self.total_bytes_received,
            'expected_segment': self.expected_segment,
            'out_of_order_count': len(self.out_of_order_buffer),
            'transfer_complete': self.transfer_complete,
            'buffer': buffer_stats
        }
    
    def reset(self):
        """Reset receiver state."""
        self.buffer.clear()
        self.expected_segment = 0
        self.out_of_order_buffer.clear()
        self.transfer_complete = False
        self.last_segment_received = False
        self.total_bytes_received = 0
        self.segments_received = 0


class TransportLayer:
    """
    Combined Transport Layer with sender and receiver.
    """
    
    def __init__(
        self,
        segment_size: int,
        buffer_capacity: int = RECEIVER_BUFFER_SIZE,
        on_segment_ready: Optional[Callable[[bytes, bool], None]] = None,
        on_data_ready: Optional[Callable[[bytes], None]] = None,
        on_transfer_complete: Optional[Callable[[], None]] = None,
        on_backpressure: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize transport layer.
        
        Args:
            segment_size: Maximum segment payload size
            buffer_capacity: Receive buffer capacity
            on_segment_ready: Callback when segment ready for link layer
            on_data_ready: Callback when data ready for application
            on_transfer_complete: Callback when transfer complete
            on_backpressure: Callback when backpressure changes
        """
        self.sender = TransportLayerSender(
            segment_size=segment_size,
            on_segment_ready=on_segment_ready
        )
        
        self.receiver = TransportLayerReceiver(
            buffer_capacity=buffer_capacity,
            on_data_ready=on_data_ready,
            on_transfer_complete=on_transfer_complete,
            on_backpressure=on_backpressure
        )
    
    def start_send(self, data: bytes):
        """Start sending data."""
        self.sender.start_transfer(len(data))
        self.sender.add_data(data)
    
    def get_next_segment(self) -> Optional[Tuple[bytes, bool]]:
        """Get next segment to send."""
        return self.sender.get_next_segment()
    
    def receive_segment(self, segment_data: bytes) -> bool:
        """Receive a segment."""
        return self.receiver.receive_segment(segment_data)
    
    def get_received_data(self) -> bytes:
        """Get all received data."""
        return self.receiver.get_data()
    
    def is_send_complete(self) -> bool:
        """Check if sending is complete."""
        return self.sender.transfer_complete
    
    def is_receive_complete(self) -> bool:
        """Check if receiving is complete."""
        return self.receiver.transfer_complete
    
    def get_statistics(self) -> dict:
        """Get statistics."""
        return {
            'sender': self.sender.get_statistics(),
            'receiver': self.receiver.get_statistics()
        }


if __name__ == "__main__":
    # Test transport layer
    print("=" * 60)
    print("TRANSPORT LAYER TEST")
    print("=" * 60)
    
    # Test segmentation
    segment_size = 100
    sender = TransportLayerSender(segment_size=segment_size)
    
    # Create test data
    test_data = b"Hello, this is a test message that will be segmented! " * 10
    print(f"\nTest data size: {len(test_data)} bytes")
    print(f"Segment size: {segment_size} bytes")
    
    # Start transfer
    sender.start_transfer(len(test_data))
    sender.add_data(test_data)
    
    # Segment data
    segments = []
    while sender.has_data():
        result = sender.get_next_segment()
        if result:
            seg_data, is_last = result
            segments.append(seg_data)
            print(f"  Segment {len(segments)-1}: {len(seg_data)} bytes, last={is_last}")
    
    print(f"\nCreated {len(segments)} segments")
    print(f"Sender stats: {sender.get_statistics()}")
    
    # Test reassembly
    print("\n--- Testing Reassembly ---")
    
    received_data = []
    def on_data(data):
        received_data.append(data)
    
    def on_complete():
        print("  Transfer complete!")
    
    receiver = TransportLayerReceiver(
        buffer_capacity=1024,
        on_data_ready=on_data,
        on_transfer_complete=on_complete
    )
    
    # Receive segments (in order)
    for seg_data in segments:
        receiver.receive_segment(seg_data)
    
    # Get reassembled data
    reassembled = receiver.get_data()
    print(f"\nReassembled data size: {len(reassembled)} bytes")
    print(f"Data matches original: {reassembled == test_data}")
    print(f"Receiver stats: {receiver.get_statistics()}")
