"""
Frame Structure for Selective Repeat ARQ Protocol

This module defines the frame structure used in the link layer,
including sequence numbering, frame types, and serialization.
"""

import struct
import zlib
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass
import sys
sys.path.insert(0, '..')
from config import LINK_HEADER_SIZE


class FrameType(Enum):
    """Frame type enumeration."""
    DATA = 0x01
    ACK = 0x02
    NAK = 0x03


@dataclass
class Frame:
    """
    Link Layer Frame Structure.
    
    Frame Header Layout (24 bytes):
        - Frame Type: 1 byte
        - Sequence Number: 4 bytes (unsigned int)
        - Payload Length: 2 bytes (unsigned short)
        - Flags: 1 byte
        - ACK Number: 4 bytes (for piggybacked ACKs)
        - Reserved: 8 bytes
        - CRC32: 4 bytes
        
    Total Header: 24 bytes (as specified in requirements)
    
    Attributes:
        frame_type: Type of frame (DATA, ACK, NAK)
        seq_num: Sequence number
        payload: Frame payload data (bytes)
        ack_num: Acknowledgment number (for piggybacked ACKs)
        flags: Frame flags
        crc: CRC32 checksum
    """
    
    frame_type: FrameType
    seq_num: int
    payload: bytes = b''
    ack_num: int = 0
    flags: int = 0
    crc: int = 0
    
    # Flag bit definitions
    FLAG_PIGGYBACK_ACK = 0x01
    FLAG_LAST_FRAME = 0x02
    FLAG_RETRANSMISSION = 0x04
    
    # Header format: type(1) + seq(4) + len(2) + flags(1) + ack(4) + reserved(8) + crc(4)
    HEADER_FORMAT = '!BIHBIQ4sI'  # Network byte order
    HEADER_SIZE = LINK_HEADER_SIZE  # 24 bytes
    
    def __post_init__(self):
        """Validate frame after initialization."""
        if self.seq_num < 0:
            raise ValueError("Sequence number must be non-negative")
        if len(self.payload) > 65535:
            raise ValueError("Payload too large (max 65535 bytes)")
    
    @property
    def total_size(self) -> int:
        """Get total frame size (header + payload)."""
        return self.HEADER_SIZE + len(self.payload)
    
    @property
    def payload_size(self) -> int:
        """Get payload size."""
        return len(self.payload)
    
    def has_piggyback_ack(self) -> bool:
        """Check if frame has piggybacked ACK."""
        return bool(self.flags & self.FLAG_PIGGYBACK_ACK)
    
    def is_last_frame(self) -> bool:
        """Check if this is the last frame in transmission."""
        return bool(self.flags & self.FLAG_LAST_FRAME)
    
    def is_retransmission(self) -> bool:
        """Check if this is a retransmitted frame."""
        return bool(self.flags & self.FLAG_RETRANSMISSION)
    
    def set_piggyback_ack(self, ack_num: int):
        """Set piggybacked ACK."""
        self.ack_num = ack_num
        self.flags |= self.FLAG_PIGGYBACK_ACK
    
    def set_last_frame(self):
        """Mark as last frame."""
        self.flags |= self.FLAG_LAST_FRAME
    
    def set_retransmission(self):
        """Mark as retransmission."""
        self.flags |= self.FLAG_RETRANSMISSION
    
    def calculate_crc(self) -> int:
        """
        Calculate CRC32 for the frame.
        
        Returns:
            CRC32 checksum
        """
        # Create header without CRC
        header_data = struct.pack(
            '!BIHBIQ4s',
            self.frame_type.value,
            self.seq_num,
            len(self.payload),
            self.flags,
            self.ack_num,
            0,  # reserved
            b'\x00\x00\x00\x00'  # reserved padding
        )
        
        # Calculate CRC over header (without CRC field) and payload
        data_to_check = header_data + self.payload
        return zlib.crc32(data_to_check) & 0xFFFFFFFF
    
    def serialize(self) -> bytes:
        """
        Serialize the frame to bytes.
        
        Returns:
            Serialized frame as bytes
        """
        # Calculate CRC
        self.crc = self.calculate_crc()
        
        # Pack header
        header = struct.pack(
            self.HEADER_FORMAT,
            self.frame_type.value,
            self.seq_num,
            len(self.payload),
            self.flags,
            self.ack_num,
            0,  # reserved
            b'\x00\x00\x00\x00',  # reserved padding
            self.crc
        )
        
        return header + self.payload
    
    @classmethod
    def deserialize(cls, data: bytes) -> Tuple[Optional['Frame'], bool]:
        """
        Deserialize bytes to a Frame object.
        
        Args:
            data: Serialized frame bytes
            
        Returns:
            Tuple of (Frame or None, CRC valid)
        """
        if len(data) < cls.HEADER_SIZE:
            return None, False
        
        try:
            # Unpack header
            header_data = data[:cls.HEADER_SIZE]
            unpacked = struct.unpack(cls.HEADER_FORMAT, header_data)
            
            frame_type_val, seq_num, payload_len, flags, ack_num, _, _, crc = unpacked
            
            # Get frame type
            try:
                frame_type = FrameType(frame_type_val)
            except ValueError:
                return None, False
            
            # Extract payload
            payload = data[cls.HEADER_SIZE:cls.HEADER_SIZE + payload_len]
            
            if len(payload) != payload_len:
                return None, False
            
            # Create frame
            frame = cls(
                frame_type=frame_type,
                seq_num=seq_num,
                payload=payload,
                ack_num=ack_num,
                flags=flags,
                crc=crc
            )
            
            # Verify CRC
            expected_crc = frame.calculate_crc()
            crc_valid = (crc == expected_crc)
            
            return frame, crc_valid
            
        except Exception:
            return None, False
    
    @classmethod
    def create_data_frame(
        cls,
        seq_num: int,
        payload: bytes,
        is_last: bool = False,
        is_retransmit: bool = False
    ) -> 'Frame':
        """
        Create a DATA frame.
        
        Args:
            seq_num: Sequence number
            payload: Frame payload
            is_last: Whether this is the last frame
            is_retransmit: Whether this is a retransmission
            
        Returns:
            DATA frame
        """
        frame = cls(
            frame_type=FrameType.DATA,
            seq_num=seq_num,
            payload=payload
        )
        
        if is_last:
            frame.set_last_frame()
        if is_retransmit:
            frame.set_retransmission()
        
        return frame
    
    @classmethod
    def create_ack_frame(cls, ack_num: int) -> 'Frame':
        """
        Create an ACK frame.
        
        Args:
            ack_num: Acknowledgment number
            
        Returns:
            ACK frame
        """
        return cls(
            frame_type=FrameType.ACK,
            seq_num=0,
            ack_num=ack_num
        )
    
    @classmethod
    def create_nak_frame(cls, nak_num: int) -> 'Frame':
        """
        Create a NAK frame.
        
        Args:
            nak_num: Negative acknowledgment number
            
        Returns:
            NAK frame
        """
        return cls(
            frame_type=FrameType.NAK,
            seq_num=nak_num,
            ack_num=nak_num
        )
    
    def __repr__(self) -> str:
        return (f"Frame(type={self.frame_type.name}, seq={self.seq_num}, "
                f"ack={self.ack_num}, payload_len={len(self.payload)}, "
                f"flags=0x{self.flags:02x})")


class FrameBuffer:
    """
    Buffer for storing frames awaiting acknowledgment or delivery.
    
    Used by both sender (for retransmission) and receiver (for out-of-order).
    """
    
    def __init__(self, max_size: int):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to buffer
        """
        self.max_size = max_size
        self.buffer: dict[int, Frame] = {}
    
    def add(self, frame: Frame) -> bool:
        """
        Add a frame to the buffer.
        
        Args:
            frame: Frame to add
            
        Returns:
            True if successful, False if buffer full
        """
        if len(self.buffer) >= self.max_size:
            return False
        
        self.buffer[frame.seq_num] = frame
        return True
    
    def get(self, seq_num: int) -> Optional[Frame]:
        """
        Get a frame by sequence number.
        
        Args:
            seq_num: Sequence number
            
        Returns:
            Frame or None
        """
        return self.buffer.get(seq_num)
    
    def remove(self, seq_num: int) -> Optional[Frame]:
        """
        Remove a frame from the buffer.
        
        Args:
            seq_num: Sequence number
            
        Returns:
            Removed frame or None
        """
        return self.buffer.pop(seq_num, None)
    
    def contains(self, seq_num: int) -> bool:
        """Check if buffer contains a frame with given sequence number."""
        return seq_num in self.buffer
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    @property
    def size(self) -> int:
        """Get number of frames in buffer."""
        return len(self.buffer)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) >= self.max_size
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def get_sequence_numbers(self) -> list:
        """Get list of sequence numbers in buffer."""
        return sorted(self.buffer.keys())


if __name__ == "__main__":
    # Test frame creation and serialization
    print("=" * 60)
    print("FRAME STRUCTURE TEST")
    print("=" * 60)
    
    # Test DATA frame
    payload = b"Hello, World! This is test data."
    data_frame = Frame.create_data_frame(seq_num=42, payload=payload)
    print(f"\nDATA Frame: {data_frame}")
    print(f"  Total size: {data_frame.total_size} bytes")
    print(f"  Header size: {Frame.HEADER_SIZE} bytes")
    
    # Serialize and deserialize
    serialized = data_frame.serialize()
    print(f"  Serialized size: {len(serialized)} bytes")
    
    deserialized, crc_valid = Frame.deserialize(serialized)
    print(f"  Deserialized: {deserialized}")
    print(f"  CRC Valid: {crc_valid}")
    
    # Test ACK frame
    ack_frame = Frame.create_ack_frame(ack_num=42)
    print(f"\nACK Frame: {ack_frame}")
    print(f"  Total size: {ack_frame.total_size} bytes")
    
    # Test NAK frame
    nak_frame = Frame.create_nak_frame(nak_num=10)
    print(f"\nNAK Frame: {nak_frame}")
    
    # Test frame buffer
    print("\n" + "=" * 60)
    print("FRAME BUFFER TEST")
    print("=" * 60)
    
    buffer = FrameBuffer(max_size=5)
    
    for i in range(7):
        frame = Frame.create_data_frame(seq_num=i, payload=f"Data {i}".encode())
        result = buffer.add(frame)
        print(f"Add frame {i}: {'Success' if result else 'Failed (buffer full)'}")
    
    print(f"\nBuffer size: {buffer.size}")
    print(f"Sequence numbers: {buffer.get_sequence_numbers()}")
    
    # Remove some frames
    buffer.remove(2)
    buffer.remove(4)
    print(f"After removing 2 and 4: {buffer.get_sequence_numbers()}")
