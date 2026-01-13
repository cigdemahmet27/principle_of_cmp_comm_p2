"""
Link Layer Implementation with Selective Repeat ARQ

This module implements the link layer with Selective Repeat ARQ protocol,
integrating sender, receiver, and physical layer components.
"""

from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import sys
sys.path.insert(0, '..')

from config import LINK_HEADER_SIZE
from src.arq.frame import Frame, FrameType
from src.arq.sender import SRSender
from src.arq.receiver import SRReceiver
from src.layers.physical_layer import PhysicalLayer, TransmissionDirection


class LinkLayerRole(Enum):
    """Role of the link layer instance."""
    SENDER = 0
    RECEIVER = 1
    BIDIRECTIONAL = 2


@dataclass
class LinkLayerConfig:
    """Configuration for link layer."""
    window_size: int = 8
    payload_size: int = 1024
    timeout: float = 0.5
    max_retransmissions: int = 10
    adaptive_timeout: bool = False


class LinkLayer:
    """
    Link Layer with Selective Repeat ARQ.
    
    Provides reliable data transfer over unreliable physical layer
    using Selective Repeat ARQ protocol.
    
    Attributes:
        config: Link layer configuration
        sender: SR-ARQ sender instance
        receiver: SR-ARQ receiver instance
        physical_layer: Physical layer instance
    """
    
    def __init__(
        self,
        config: LinkLayerConfig,
        physical_layer: PhysicalLayer,
        role: LinkLayerRole = LinkLayerRole.BIDIRECTIONAL,
        on_data_delivered: Optional[Callable[[bytes, int], None]] = None,
        on_frame_sent: Optional[Callable[[Frame], None]] = None,
        on_ack_sent: Optional[Callable[[Frame], None]] = None
    ):
        """
        Initialize link layer.
        
        Args:
            config: Link layer configuration
            physical_layer: Physical layer instance
            role: Link layer role
            on_data_delivered: Callback when data delivered to upper layer
            on_frame_sent: Callback when data frame sent
            on_ack_sent: Callback when ACK/NAK sent
        """
        self.config = config
        self.physical_layer = physical_layer
        self.role = role
        
        # Callbacks
        self.on_data_delivered = on_data_delivered
        self.on_frame_sent = on_frame_sent
        self.on_ack_sent = on_ack_sent
        
        # Create sender if needed
        self.sender: Optional[SRSender] = None
        if role in (LinkLayerRole.SENDER, LinkLayerRole.BIDIRECTIONAL):
            self.sender = SRSender(
                window_size=config.window_size,
                timeout=config.timeout,
                max_retransmissions=config.max_retransmissions,
                adaptive_timeout=config.adaptive_timeout,
                on_frame_sent=self._handle_frame_sent,
                on_timeout=self._handle_timeout
            )
        
        # Create receiver if needed
        self.receiver: Optional[SRReceiver] = None
        if role in (LinkLayerRole.RECEIVER, LinkLayerRole.BIDIRECTIONAL):
            self.receiver = SRReceiver(
                window_size=config.window_size,
                on_data_delivered=self._handle_data_delivered,
                on_ack_generated=self._handle_ack_generated
            )
        
        # Backpressure state
        self.backpressure_active = False
        
        # Statistics
        self.data_frames_sent = 0
        self.data_frames_received = 0
        self.ack_frames_sent = 0
        self.ack_frames_received = 0
    
    def send_data(self, data: bytes, is_last: bool = False):
        """
        Queue data for transmission.
        
        Args:
            data: Data payload to send
            is_last: Whether this is the last piece of data
        """
        if self.sender is None:
            raise RuntimeError("Link layer not configured as sender")
        
        if self.backpressure_active:
            # Could queue or reject - for now, still queue
            pass
        
        self.sender.queue_data(data, is_last)
    
    def process_send(self, current_time: float) -> List[Frame]:
        """
        Process sending - transmit frames if possible.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of frames transmitted
        """
        if self.sender is None:
            return []
        
        frames_sent = []
        
        # Send new frames if window has space
        while self.sender.can_send():
            frame = self.sender.send_next_frame(current_time)
            if frame:
                self._transmit_frame(frame, current_time, TransmissionDirection.FORWARD)
                frames_sent.append(frame)
        
        # Check for timeouts and retransmit
        retransmit_frames = self.sender.get_frames_to_retransmit(current_time)
        for frame in retransmit_frames:
            self._transmit_frame(frame, current_time, TransmissionDirection.FORWARD)
            frames_sent.append(frame)
        
        return frames_sent
    
    def process_receive(
        self,
        current_time: float
    ) -> List[Tuple[Frame, bool]]:
        """
        Process receiving - handle arrived frames.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of (frame, was_valid) tuples
        """
        received = []
        
        # Get all arrived frames from physical layer
        arrivals = self.physical_layer.receive_all_arrived(current_time)
        
        for frame, crc_valid, direction in arrivals:
            if frame is None:
                continue
            
            received.append((frame, crc_valid))
            
            if direction == TransmissionDirection.FORWARD:
                # Data frame received at receiver side
                self._handle_data_frame_received(frame, crc_valid, current_time)
            else:
                # ACK/NAK received at sender side
                self._handle_ack_frame_received(frame, crc_valid, current_time)
        
        return received
    
    def _handle_data_frame_received(
        self,
        frame: Frame,
        crc_valid: bool,
        current_time: float
    ):
        """Handle received data frame."""
        if self.receiver is None:
            return
        
        self.data_frames_received += 1
        
        # Process frame at receiver
        ack_frame = self.receiver.receive_frame(frame, crc_valid)
        
        # Send ACK if generated
        if ack_frame:
            self._transmit_ack(ack_frame, current_time)
    
    def _handle_ack_frame_received(
        self,
        frame: Frame,
        crc_valid: bool,
        current_time: float
    ):
        """Handle received ACK/NAK frame."""
        if self.sender is None:
            return
        
        if not crc_valid:
            # Corrupted ACK - ignore
            return
        
        self.ack_frames_received += 1
        
        if frame.frame_type == FrameType.ACK:
            self.sender.process_ack(frame.ack_num, current_time)
        elif frame.frame_type == FrameType.NAK:
            # Immediate retransmission on NAK
            retx_frame = self.sender.process_nak(frame.ack_num, current_time)
            if retx_frame:
                self._transmit_frame(retx_frame, current_time, TransmissionDirection.FORWARD)
    
    def _transmit_frame(
        self,
        frame: Frame,
        current_time: float,
        direction: TransmissionDirection
    ):
        """Transmit a frame through physical layer."""
        self.physical_layer.transmit_frame(frame, current_time, direction)
        self.data_frames_sent += 1
    
    def _transmit_ack(self, frame: Frame, current_time: float):
        """Transmit ACK/NAK frame."""
        self.physical_layer.transmit_frame(
            frame, current_time, TransmissionDirection.REVERSE
        )
        self.ack_frames_sent += 1
    
    def _handle_frame_sent(self, frame: Frame):
        """Callback when frame is sent by sender."""
        if self.on_frame_sent:
            self.on_frame_sent(frame)
    
    def _handle_timeout(self, seq_num: int):
        """Callback when timeout occurs."""
        pass  # Retransmission handled in process_send
    
    def _handle_data_delivered(self, data: bytes, seq_num: int):
        """Callback when data is delivered in-order."""
        if self.on_data_delivered:
            self.on_data_delivered(data, seq_num)
    
    def _handle_ack_generated(self, frame: Frame):
        """Callback when ACK is generated."""
        if self.on_ack_sent:
            self.on_ack_sent(frame)
    
    def set_backpressure(self, active: bool):
        """
        Set backpressure state from upper layer.
        
        Args:
            active: Whether backpressure is active
        """
        self.backpressure_active = active
    
    def is_transfer_complete(self) -> bool:
        """Check if transfer is complete."""
        if self.sender:
            return self.sender.is_complete()
        if self.receiver:
            return self.receiver.is_complete()
        return True
    
    def get_next_event_time(self) -> Optional[float]:
        """Get time of next event (timer or frame arrival)."""
        times = []
        
        if self.sender:
            timer_time = self.sender.get_next_event_time()
            if timer_time is not None:
                times.append(timer_time)
        
        arrival_time = self.physical_layer.get_next_arrival_time()
        if arrival_time is not None:
            times.append(arrival_time)
        
        return min(times) if times else None
    
    def get_sender_window_state(self) -> Optional[dict]:
        """Get sender window state."""
        if self.sender:
            return self.sender.get_window_state()
        return None
    
    def get_receiver_window_state(self) -> Optional[dict]:
        """Get receiver window state."""
        if self.receiver:
            return self.receiver.get_window_state()
        return None
    
    def get_statistics(self) -> dict:
        """Get link layer statistics."""
        stats = {
            'data_frames_sent': self.data_frames_sent,
            'data_frames_received': self.data_frames_received,
            'ack_frames_sent': self.ack_frames_sent,
            'ack_frames_received': self.ack_frames_received,
            'backpressure_active': self.backpressure_active
        }
        
        if self.sender:
            stats['sender'] = self.sender.get_statistics()
        if self.receiver:
            stats['receiver'] = self.receiver.get_statistics()
        
        return stats
    
    def reset(self):
        """Reset link layer."""
        if self.sender:
            self.sender.reset()
        if self.receiver:
            self.receiver.reset()
        
        self.backpressure_active = False
        self.data_frames_sent = 0
        self.data_frames_received = 0
        self.ack_frames_sent = 0
        self.ack_frames_received = 0


class LinkLayerPair:
    """
    Helper class to manage sender and receiver link layers.
    
    Used in simulation to represent both ends of a connection.
    """
    
    def __init__(
        self,
        config: LinkLayerConfig,
        physical_layer_forward: PhysicalLayer,
        physical_layer_reverse: PhysicalLayer,
        on_data_delivered: Optional[Callable[[bytes, int], None]] = None
    ):
        """
        Initialize link layer pair.
        
        Args:
            config: Link layer configuration
            physical_layer_forward: Physical layer for forward direction
            physical_layer_reverse: Physical layer for reverse direction
            on_data_delivered: Callback when data delivered at receiver
        """
        self.config = config
        
        # Create sender-side link layer
        self.sender_link = LinkLayer(
            config=config,
            physical_layer=physical_layer_forward,
            role=LinkLayerRole.SENDER
        )
        
        # Create receiver-side link layer
        self.receiver_link = LinkLayer(
            config=config,
            physical_layer=physical_layer_reverse,
            role=LinkLayerRole.RECEIVER,
            on_data_delivered=on_data_delivered
        )
    
    def send_data(self, data: bytes, is_last: bool = False):
        """Queue data at sender."""
        self.sender_link.send_data(data, is_last)
    
    def process(self, current_time: float):
        """Process both sender and receiver."""
        # Process sender (transmit new frames, retransmissions)
        self.sender_link.process_send(current_time)
        
        # Process receiver (receive data frames, send ACKs)
        self.receiver_link.process_receive(current_time)
        
        # Process sender (receive ACKs)
        self.sender_link.process_receive(current_time)


if __name__ == "__main__":
    # Test link layer
    print("=" * 60)
    print("LINK LAYER TEST")
    print("=" * 60)
    
    # Create physical layer
    phy = PhysicalLayer(seed=42)
    
    # Create link layer config
    config = LinkLayerConfig(
        window_size=4,
        payload_size=256,
        timeout=0.2
    )
    
    delivered_data = []
    
    def on_delivered(data, seq):
        delivered_data.append((seq, data))
        print(f"  [DELIVERED] Frame {seq}: {len(data)} bytes")
    
    # Create link layer
    link = LinkLayer(
        config=config,
        physical_layer=phy,
        role=LinkLayerRole.BIDIRECTIONAL,
        on_data_delivered=on_delivered
    )
    
    # Queue some data
    print("\nQueuing 10 segments...")
    for i in range(10):
        data = f"Segment {i:02d} data".encode().ljust(256, b'\x00')
        link.send_data(data, is_last=(i == 9))
    
    # Simulate
    print("\nSimulating transfer...")
    current_time = 0.0
    max_time = 5.0
    
    while current_time < max_time and not link.is_transfer_complete():
        # Process sending
        link.process_send(current_time)
        
        # Advance to next event
        next_time = link.get_next_event_time()
        if next_time is None:
            break
        
        current_time = next_time
        
        # Process receiving
        link.process_receive(current_time)
    
    print(f"\nSimulation ended at t={current_time:.4f}s")
    print(f"Transfer complete: {link.is_transfer_complete()}")
    print(f"Delivered {len(delivered_data)} segments")
    print(f"\nStatistics: {link.get_statistics()}")
