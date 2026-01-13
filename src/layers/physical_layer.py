"""
Physical Layer Implementation

This module implements the physical layer with Gilbert-Elliot burst error
channel and propagation delay simulation.
"""

from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import heapq
import sys
sys.path.insert(0, '..')

from config import (
    BIT_RATE,
    FORWARD_PROPAGATION_DELAY,
    REVERSE_PROPAGATION_DELAY,
    PROCESSING_DELAY
)
from src.channel.gilbert_elliot import GilbertElliottChannel, ChannelState
from src.arq.frame import Frame


class TransmissionDirection(Enum):
    """Direction of transmission."""
    FORWARD = 0  # Data frames: sender to receiver
    REVERSE = 1  # ACK frames: receiver to sender


@dataclass(order=True)
class TransmissionEvent:
    """Event for frame arriving at destination."""
    arrival_time: float
    frame_data: bytes = field(compare=False)
    direction: TransmissionDirection = field(compare=False)
    corrupted: bool = field(compare=False, default=False)
    bit_errors: int = field(compare=False, default=0)


class PhysicalLayer:
    """
    Physical Layer Implementation.
    
    Simulates:
    - Bit rate / transmission time
    - Asymmetric propagation delays
    - Processing delays
    - Gilbert-Elliot burst error channel
    
    Attributes:
        bit_rate: Channel bit rate in bps
        forward_delay: Forward path propagation delay
        reverse_delay: Reverse path propagation delay
        processing_delay: Processing delay per frame
        channel: Gilbert-Elliot channel model
    """
    
    def __init__(
        self,
        bit_rate: float = BIT_RATE,
        forward_delay: float = FORWARD_PROPAGATION_DELAY,
        reverse_delay: float = REVERSE_PROPAGATION_DELAY,
        processing_delay: float = PROCESSING_DELAY,
        seed: Optional[int] = None,
        on_frame_corrupted: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize physical layer.
        
        Args:
            bit_rate: Bit rate in bps
            forward_delay: Forward propagation delay in seconds
            reverse_delay: Reverse propagation delay in seconds
            processing_delay: Processing delay in seconds
            seed: Random seed for reproducibility
            on_frame_corrupted: Callback when frame is corrupted
        """
        self.bit_rate = bit_rate
        self.forward_delay = forward_delay
        self.reverse_delay = reverse_delay
        self.processing_delay = processing_delay
        self.on_frame_corrupted = on_frame_corrupted
        
        # Channel model
        self.channel = GilbertElliottChannel(seed=seed)
        
        # Event queue for frames in transit
        self.transit_queue: List[TransmissionEvent] = []
        
        # Statistics
        self.frames_transmitted = 0
        self.frames_corrupted = 0
        self.total_bits_transmitted = 0
        self.total_propagation_time = 0.0
    
    def calculate_transmission_time(self, frame_size_bytes: int) -> float:
        """
        Calculate time to transmit a frame.
        
        Args:
            frame_size_bytes: Frame size in bytes
            
        Returns:
            Transmission time in seconds
        """
        frame_size_bits = frame_size_bytes * 8
        return frame_size_bits / self.bit_rate
    
    def get_propagation_delay(self, direction: TransmissionDirection) -> float:
        """
        Get propagation delay for given direction.
        
        Args:
            direction: Transmission direction
            
        Returns:
            Propagation delay in seconds
        """
        if direction == TransmissionDirection.FORWARD:
            return self.forward_delay
        return self.reverse_delay
    
    def calculate_total_delay(
        self,
        frame_size_bytes: int,
        direction: TransmissionDirection
    ) -> float:
        """
        Calculate total delay for a frame.
        
        Total = Transmission + Propagation + Processing
        
        Args:
            frame_size_bytes: Frame size in bytes
            direction: Transmission direction
            
        Returns:
            Total delay in seconds
        """
        tx_time = self.calculate_transmission_time(frame_size_bytes)
        prop_delay = self.get_propagation_delay(direction)
        
        return tx_time + prop_delay + self.processing_delay
    
    def transmit_frame(
        self,
        frame: Frame,
        current_time: float,
        direction: TransmissionDirection = TransmissionDirection.FORWARD
    ) -> Tuple[float, bool]:
        """
        Transmit a frame through the channel.
        
        Args:
            frame: Frame to transmit
            current_time: Current simulation time
            direction: Transmission direction
            
        Returns:
            Tuple of (arrival_time, corrupted)
        """
        frame_data = frame.serialize()
        frame_size_bytes = len(frame_data)
        frame_size_bits = frame_size_bytes * 8
        
        # Simulate channel errors
        corrupted, bit_errors = self.channel.transmit_frame(frame_size_bits)
        
        # Calculate arrival time
        total_delay = self.calculate_total_delay(frame_size_bytes, direction)
        arrival_time = current_time + total_delay
        
        # Create transmission event
        event = TransmissionEvent(
            arrival_time=arrival_time,
            frame_data=frame_data,
            direction=direction,
            corrupted=corrupted,
            bit_errors=bit_errors
        )
        
        # Add to transit queue
        heapq.heappush(self.transit_queue, event)
        
        # Update statistics
        self.frames_transmitted += 1
        self.total_bits_transmitted += frame_size_bits
        self.total_propagation_time += total_delay
        
        if corrupted:
            self.frames_corrupted += 1
            if self.on_frame_corrupted:
                self.on_frame_corrupted(frame.seq_num)
        
        return arrival_time, corrupted
    
    def transmit_bytes(
        self,
        data: bytes,
        current_time: float,
        direction: TransmissionDirection = TransmissionDirection.FORWARD
    ) -> Tuple[float, bool, int]:
        """
        Transmit raw bytes through the channel.
        
        Args:
            data: Bytes to transmit
            current_time: Current simulation time
            direction: Transmission direction
            
        Returns:
            Tuple of (arrival_time, corrupted, bit_errors)
        """
        frame_size_bytes = len(data)
        frame_size_bits = frame_size_bytes * 8
        
        # Simulate channel errors
        corrupted, bit_errors = self.channel.transmit_frame(frame_size_bits)
        
        # Calculate arrival time
        total_delay = self.calculate_total_delay(frame_size_bytes, direction)
        arrival_time = current_time + total_delay
        
        # Create transmission event
        event = TransmissionEvent(
            arrival_time=arrival_time,
            frame_data=data,
            direction=direction,
            corrupted=corrupted,
            bit_errors=bit_errors
        )
        
        # Add to transit queue
        heapq.heappush(self.transit_queue, event)
        
        # Update statistics
        self.frames_transmitted += 1
        self.total_bits_transmitted += frame_size_bits
        
        if corrupted:
            self.frames_corrupted += 1
        
        return arrival_time, corrupted, bit_errors
    
    def get_next_arrival(self) -> Optional[TransmissionEvent]:
        """
        Get the next frame arrival event without removing it.
        
        Returns:
            Next transmission event or None
        """
        if not self.transit_queue:
            return None
        return self.transit_queue[0]
    
    def get_next_arrival_time(self) -> Optional[float]:
        """
        Get the time of the next frame arrival.
        
        Returns:
            Arrival time or None if no frames in transit
        """
        event = self.get_next_arrival()
        return event.arrival_time if event else None
    
    def receive_frame(
        self,
        current_time: float
    ) -> Optional[Tuple[Frame, bool, TransmissionDirection]]:
        """
        Receive a frame that has arrived.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Tuple of (Frame, crc_valid, direction) or None
        """
        if not self.transit_queue:
            return None
        
        event = self.transit_queue[0]
        if event.arrival_time > current_time:
            return None
        
        # Remove from queue
        heapq.heappop(self.transit_queue)
        
        # If corrupted, CRC will fail
        if event.corrupted:
            # Still try to deserialize, but mark as invalid
            frame, _ = Frame.deserialize(event.frame_data)
            return (frame, False, event.direction) if frame else None
        
        # Deserialize frame and verify CRC
        frame, crc_valid = Frame.deserialize(event.frame_data)
        
        return (frame, crc_valid, event.direction) if frame else None
    
    def receive_all_arrived(
        self,
        current_time: float
    ) -> List[Tuple[Frame, bool, TransmissionDirection]]:
        """
        Receive all frames that have arrived by current time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of (Frame, crc_valid, direction) tuples
        """
        arrived = []
        
        while True:
            result = self.receive_frame(current_time)
            if result is None:
                break
            arrived.append(result)
        
        return arrived
    
    def has_frames_in_transit(self) -> bool:
        """Check if any frames are in transit."""
        return len(self.transit_queue) > 0
    
    def clear_transit_queue(self):
        """Clear all frames in transit."""
        self.transit_queue.clear()
    
    def get_channel_state(self) -> ChannelState:
        """Get current channel state."""
        return self.channel.state
    
    def get_statistics(self) -> dict:
        """Get physical layer statistics."""
        channel_stats = self.channel.get_statistics()
        
        return {
            'frames_transmitted': self.frames_transmitted,
            'frames_corrupted': self.frames_corrupted,
            'frame_error_rate': (self.frames_corrupted / self.frames_transmitted 
                                if self.frames_transmitted > 0 else 0),
            'total_bits_transmitted': self.total_bits_transmitted,
            'total_propagation_time': self.total_propagation_time,
            'frames_in_transit': len(self.transit_queue),
            'channel': channel_stats
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset physical layer."""
        self.channel.reset(seed)
        self.transit_queue.clear()
        self.frames_transmitted = 0
        self.frames_corrupted = 0
        self.total_bits_transmitted = 0
        self.total_propagation_time = 0.0


# Utility function for RTT calculation
def calculate_theoretical_rtt(
    payload_size: int,
    bit_rate: float = BIT_RATE,
    forward_delay: float = FORWARD_PROPAGATION_DELAY,
    reverse_delay: float = REVERSE_PROPAGATION_DELAY,
    processing_delay: float = PROCESSING_DELAY,
    link_header_size: int = 24
) -> float:
    """
    Calculate theoretical RTT for given parameters.
    
    RTT = Tx_data + Prop_forward + Proc + Tx_ack + Prop_reverse + Proc
    
    Args:
        payload_size: Payload size in bytes
        bit_rate: Channel bit rate
        forward_delay: Forward propagation delay
        reverse_delay: Reverse propagation delay
        processing_delay: Processing delay per frame
        link_header_size: Link layer header size
        
    Returns:
        Theoretical RTT in seconds
    """
    # Data frame: header + payload
    data_frame_size = link_header_size + payload_size
    data_tx_time = (data_frame_size * 8) / bit_rate
    
    # ACK frame: just header
    ack_frame_size = link_header_size
    ack_tx_time = (ack_frame_size * 8) / bit_rate
    
    rtt = (data_tx_time + forward_delay + processing_delay +
           ack_tx_time + reverse_delay + processing_delay)
    
    return rtt


if __name__ == "__main__":
    # Test physical layer
    print("=" * 60)
    print("PHYSICAL LAYER TEST")
    print("=" * 60)
    
    def on_corrupt(seq):
        print(f"  [CORRUPTED] Frame {seq}")
    
    phy = PhysicalLayer(seed=42, on_frame_corrupted=on_corrupt)
    
    print(f"\nConfiguration:")
    print(f"  Bit rate: {phy.bit_rate / 1e6:.0f} Mbps")
    print(f"  Forward delay: {phy.forward_delay * 1000:.0f} ms")
    print(f"  Reverse delay: {phy.reverse_delay * 1000:.0f} ms")
    print(f"  Processing delay: {phy.processing_delay * 1000:.0f} ms")
    
    # Calculate theoretical delays
    payload = 1024  # bytes
    print(f"\nFor {payload} byte payload:")
    tx_time = phy.calculate_transmission_time(payload + 24)
    print(f"  Transmission time: {tx_time * 1000:.3f} ms")
    
    total_forward = phy.calculate_total_delay(payload + 24, TransmissionDirection.FORWARD)
    print(f"  Total forward delay: {total_forward * 1000:.3f} ms")
    
    rtt = calculate_theoretical_rtt(payload)
    print(f"  Theoretical RTT: {rtt * 1000:.3f} ms")
    
    # Simulate frame transmission
    print("\nTransmitting 100 frames...")
    
    current_time = 0.0
    arrivals = []
    
    for i in range(100):
        frame = Frame.create_data_frame(i, bytes(payload))
        arrival, corrupted = phy.transmit_frame(
            frame, current_time, TransmissionDirection.FORWARD
        )
        arrivals.append((arrival, corrupted))
        current_time += 0.001  # Small interval between sends
    
    stats = phy.get_statistics()
    print(f"\nStatistics:")
    print(f"  Frames transmitted: {stats['frames_transmitted']}")
    print(f"  Frames corrupted: {stats['frames_corrupted']}")
    print(f"  Frame error rate: {stats['frame_error_rate']:.4f}")
    print(f"  Observed BER: {stats['channel']['observed_ber']:.2e}")
