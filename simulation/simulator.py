"""
Main Simulator - Event-Driven Network Simulation

This module implements the main simulation engine that orchestrates
all network layers and runs the complete transfer simulation.
"""

from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BIT_RATE, FORWARD_PROPAGATION_DELAY, REVERSE_PROPAGATION_DELAY,
    PROCESSING_DELAY, LINK_HEADER_SIZE, TRANSPORT_HEADER_SIZE,
    RECEIVER_BUFFER_SIZE, MAX_SIMULATION_TIME
)
from src.channel.gilbert_elliot import GilbertElliottChannel
from src.arq.frame import Frame, FrameType
from src.arq.sender import SRSender
from src.arq.receiver import SRReceiver
from src.layers.transport_layer import TransportSegment
from src.layers.application_layer import TestDataGenerator, DataVerifier
from src.layers.physical_layer import calculate_theoretical_rtt
from src.utils.metrics import MetricsCollector
from src.utils.logger import SimulationLogger, LogLevel


class EventType(Enum):
    """Types of simulation events."""
    DATA_ARRIVAL = 0      # Data frame arrives at receiver
    ACK_ARRIVAL = 1       # ACK arrives at sender
    TIMER_CHECK = 2       # Check for timeouts


@dataclass(order=True)
class SimEvent:
    """Simulation event."""
    time: float
    event_type: EventType = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)


@dataclass
class SimulatorConfig:
    """Configuration for the simulator."""
    # ARQ parameters
    window_size: int = 8
    payload_size: int = 1024
    
    # Timeout (auto-calculated if None)
    timeout: Optional[float] = None
    timeout_multiplier: float = 2.0  # RTT multiplier for timeout
    
    # Protocol options
    adaptive_timeout: bool = False
    max_retransmissions: int = 10
    
    # Data parameters
    data_size: int = 1024 * 1024  # 1 MB default
    
    # Simulation parameters
    seed: int = 42
    max_time: float = MAX_SIMULATION_TIME
    log_level: int = LogLevel.WARNING
    
    def get_timeout(self) -> float:
        """Calculate timeout value."""
        if self.timeout is not None:
            return self.timeout
        # Calculate based on RTT
        rtt = calculate_theoretical_rtt(self.payload_size)
        return rtt * self.timeout_multiplier


class Simulator:
    """
    Main Event-Driven Simulator.
    
    Uses a single physical channel model and simulates both 
    forward (data) and reverse (ACK) paths.
    """
    
    def __init__(self, config: SimulatorConfig):
        """Initialize simulator."""
        self.config = config
        
        # Create logger
        self.logger = SimulationLogger(
            name="Sim",
            level=config.log_level
        )
        
        # Channel model for forward path
        self.forward_channel = GilbertElliottChannel(seed=config.seed)
        # Channel model for reverse path (ACKs)
        self.reverse_channel = GilbertElliottChannel(seed=config.seed + 1000)
        
        # Calculate transport payload size
        self.transport_payload_size = config.payload_size - TRANSPORT_HEADER_SIZE
        
        # Create ARQ sender
        self.sender = SRSender(
            window_size=config.window_size,
            timeout=config.get_timeout(),
            max_retransmissions=config.max_retransmissions,
            adaptive_timeout=config.adaptive_timeout
        )
        
        # Create ARQ receiver
        self.receiver = SRReceiver(
            window_size=config.window_size,
            on_data_delivered=self._on_data_delivered
        )
        
        # Metrics
        self.metrics = MetricsCollector()
        
        # Simulation state
        self.current_time = 0.0
        self.event_queue: List[SimEvent] = []
        
        # Data tracking
        self.sent_data: bytes = b''
        self.received_data = bytearray()
        self.segments_to_send: List[Tuple[bytes, bool]] = []
        self.current_segment_idx = 0
        
        # Stats
        self.frames_sent = 0
        self.frames_corrupted = 0
        self.acks_sent = 0
        self.acks_corrupted = 0
    
    def _calculate_forward_delay(self, frame_bytes: int) -> float:
        """Calculate total forward path delay."""
        tx_time = (frame_bytes * 8) / BIT_RATE
        return tx_time + FORWARD_PROPAGATION_DELAY + PROCESSING_DELAY
    
    def _calculate_reverse_delay(self, frame_bytes: int) -> float:
        """Calculate total reverse path delay."""
        tx_time = (frame_bytes * 8) / BIT_RATE
        return tx_time + REVERSE_PROPAGATION_DELAY + PROCESSING_DELAY
    
    def _setup_data(self, data: bytes):
        """Setup data for transfer."""
        self.sent_data = data
        self.received_data = bytearray()
        self.segments_to_send = []
        
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + self.transport_payload_size]
            is_last = (offset + len(chunk) >= len(data))
            
            # Create transport segment
            segment = TransportSegment(
                segment_num=len(self.segments_to_send),
                payload=chunk
            )
            if is_last:
                segment.set_last()
            
            seg_bytes = segment.serialize()
            self.segments_to_send.append((seg_bytes, is_last))
            offset += len(chunk)
        
        self.current_segment_idx = 0
        self.logger.info(
            f"Data setup: {len(data)} bytes, {len(self.segments_to_send)} segments",
            "SETUP"
        )
    
    def _schedule_event(self, time: float, event_type: EventType, data: dict = None):
        """Schedule an event."""
        event = SimEvent(time=time, event_type=event_type, data=data or {})
        heapq.heappush(self.event_queue, event)
    
    def _on_data_delivered(self, data: bytes, seq_num: int):
        """Callback when receiver delivers data in-order."""
        # Deserialize transport segment
        segment, valid = TransportSegment.deserialize(data)
        if segment and valid:
            self.received_data.extend(segment.payload)
            self.metrics.record_data_delivered(len(segment.payload))
            self.logger.debug(f"Delivered segment {seq_num}", "RX")
    
    def _send_frames(self):
        """Send new frames if possible."""
        # Queue segments to sender
        while (self.current_segment_idx < len(self.segments_to_send) and
               not self.sender.window.is_full):
            seg_data, is_last = self.segments_to_send[self.current_segment_idx]
            self.sender.queue_data(seg_data, is_last)
            self.current_segment_idx += 1
        
        # Send queued frames
        while self.sender.can_send():
            frame = self.sender.send_next_frame(self.current_time)
            if frame is None:
                break
            
            self.frames_sent += 1
            frame_bytes = frame.total_size
            
            # Simulate channel errors
            corrupted, _ = self.forward_channel.transmit_frame(frame_bytes * 8)
            if corrupted:
                self.frames_corrupted += 1
                self.metrics.record_frame_corrupted()
            
            # Calculate arrival time
            delay = self._calculate_forward_delay(frame_bytes)
            arrival_time = self.current_time + delay
            
            # Schedule arrival event
            self._schedule_event(
                arrival_time,
                EventType.DATA_ARRIVAL,
                {'frame': frame, 'corrupted': corrupted}
            )
            
            # Record metrics
            payload_size = len(frame.payload) - TRANSPORT_HEADER_SIZE
            if payload_size > 0:
                self.metrics.record_data_sent(payload_size, frame_bytes)
            
            self.logger.debug(f"Send frame {frame.seq_num}", "TX")
    
    def _handle_data_arrival(self, event_data: dict):
        """Handle data frame arriving at receiver."""
        frame = event_data['frame']
        corrupted = event_data['corrupted']
        
        # Process at receiver
        ack_frame = self.receiver.receive_frame(frame, crc_valid=not corrupted)
        
        if ack_frame:
            self.acks_sent += 1
            
            # Simulate ACK channel
            ack_bytes = ack_frame.total_size
            ack_corrupted, _ = self.reverse_channel.transmit_frame(ack_bytes * 8)
            if ack_corrupted:
                self.acks_corrupted += 1
            
            # Schedule ACK arrival
            delay = self._calculate_reverse_delay(ack_bytes)
            arrival_time = self.current_time + delay
            
            self._schedule_event(
                arrival_time, 
                EventType.ACK_ARRIVAL,
                {'ack_frame': ack_frame, 'corrupted': ack_corrupted}
            )
            
            self.metrics.record_ack_sent(ack_bytes)
    
    def _handle_ack_arrival(self, event_data: dict):
        """Handle ACK arriving at sender."""
        ack_frame = event_data['ack_frame']
        corrupted = event_data['corrupted']
        
        if corrupted:
            # Ignore corrupted ACK
            return
        
        ack_num = ack_frame.ack_num
        
        # Record RTT
        if ack_num in self.sender.send_times:
            send_time = self.sender.send_times[ack_num]
            rtt = self.current_time - send_time
            self.metrics.record_rtt(rtt)
        
        # Process ACK at sender
        if ack_frame.frame_type == FrameType.ACK:
            self.sender.process_ack(ack_num, self.current_time)
        elif ack_frame.frame_type == FrameType.NAK:
            retx_frame = self.sender.process_nak(ack_num, self.current_time)
            if retx_frame:
                self._retransmit_frame(retx_frame)
    
    def _retransmit_frame(self, frame: Frame):
        """Retransmit a frame."""
        frame_bytes = frame.total_size
        
        # Simulate channel
        corrupted, _ = self.forward_channel.transmit_frame(frame_bytes * 8)
        if corrupted:
            self.frames_corrupted += 1
            self.metrics.record_frame_corrupted()
        
        # Calculate arrival
        delay = self._calculate_forward_delay(frame_bytes)
        arrival_time = self.current_time + delay
        
        self._schedule_event(
            arrival_time,
            EventType.DATA_ARRIVAL,
            {'frame': frame, 'corrupted': corrupted}
        )
        
        self.metrics.record_retransmission(frame_bytes)
        self.logger.debug(f"Retransmit frame {frame.seq_num}", "RTX")
    
    def _handle_timeouts(self):
        """Handle timer expirations."""
        frames_to_retx = self.sender.get_frames_to_retransmit(self.current_time)
        for frame in frames_to_retx:
            self._retransmit_frame(frame)
    
    def _is_complete(self) -> bool:
        """Check if transfer is complete."""
        return (self.sender.is_complete() and 
                len(self.received_data) >= len(self.sent_data))
    
    def run(self, data: Optional[bytes] = None) -> Dict:
        """Run the simulation."""
        # Generate data if needed
        if data is None:
            data = TestDataGenerator.generate_test_data(
                self.config.data_size,
                pattern="sequential"
            )
        
        # Setup
        self._setup_data(data)
        self.event_queue.clear()
        self.current_time = 0.0
        self.frames_sent = 0
        self.frames_corrupted = 0
        self.acks_sent = 0
        self.acks_corrupted = 0
        
        # Reset components
        self.sender.reset()
        self.receiver.reset()
        self.forward_channel.reset(self.config.seed)
        self.reverse_channel.reset(self.config.seed + 1000)
        
        # Start metrics
        self.metrics.reset()
        self.metrics.start(0.0)
        sim_start_real = time.time()
        
        # Initial send
        self._send_frames()
        
        # Schedule initial timer check
        next_timer = self.sender.get_next_event_time()
        if next_timer:
            self._schedule_event(next_timer, EventType.TIMER_CHECK)
        
        # Main loop
        max_iterations = 10000000  # 10 million iterations for large transfers
        iterations = 0
        
        while (self.current_time < self.config.max_time and 
               not self._is_complete() and
               iterations < max_iterations):
            
            if not self.event_queue:
                # Try to send more
                self._send_frames()
                
                # Schedule timer if needed
                next_timer = self.sender.get_next_event_time()
                if next_timer:
                    self._schedule_event(next_timer, EventType.TIMER_CHECK)
                
                if not self.event_queue:
                    break
            
            # Get next event
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            # Process event
            if event.event_type == EventType.DATA_ARRIVAL:
                self._handle_data_arrival(event.data)
                self._send_frames()
                
            elif event.event_type == EventType.ACK_ARRIVAL:
                self._handle_ack_arrival(event.data)
                self._send_frames()
                
            elif event.event_type == EventType.TIMER_CHECK:
                self._handle_timeouts()
            
            # Schedule next timer check
            next_timer = self.sender.get_next_event_time()
            if next_timer and next_timer > self.current_time:
                # Check if we already have a timer check scheduled
                has_timer = any(e.event_type == EventType.TIMER_CHECK for e in self.event_queue)
                if not has_timer:
                    self._schedule_event(next_timer, EventType.TIMER_CHECK)
            
            iterations += 1
        
        # Finish
        self.metrics.finish(self.current_time)
        sim_end_real = time.time()
        
        # Verify
        valid, verify_details = DataVerifier.verify_data(
            self.sent_data,
            bytes(self.received_data)
        )
        
        metrics_summary = self.metrics.get_summary()
        
        return {
            'config': {
                'window_size': self.config.window_size,
                'payload_size': self.config.payload_size,
                'data_size': self.config.data_size,
                'seed': self.config.seed,
                'timeout': self.config.get_timeout()
            },
            'metrics': metrics_summary,
            'verification': {'valid': valid, **verify_details},
            'real_time': sim_end_real - sim_start_real,
            'simulation_time': self.current_time,
            'complete': self._is_complete()
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset simulator."""
        if seed is not None:
            self.config.seed = seed
        self.sender.reset()
        self.receiver.reset()
        self.metrics.reset()
        self.current_time = 0.0
        self.event_queue.clear()


if __name__ == "__main__":
    print("=" * 60)
    print("SIMULATOR TEST")
    print("=" * 60)
    
    config = SimulatorConfig(
        window_size=8,
        payload_size=1024,
        data_size=10 * 1024,  # 10 KB
        seed=42,
        log_level=LogLevel.INFO
    )
    
    print(f"\nConfiguration:")
    print(f"  Window size: {config.window_size}")
    print(f"  Payload size: {config.payload_size} bytes")
    print(f"  Data size: {config.data_size} bytes")
    print(f"  Timeout: {config.get_timeout()*1000:.2f} ms")
    
    sim = Simulator(config)
    print("\nRunning simulation...")
    
    results = sim.run()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nTransfer:")
    print(f"  Complete: {results['complete']}")
    print(f"  Data valid: {results['verification']['valid']}")
    print(f"  Simulation time: {results['simulation_time']:.4f} s")
    print(f"  Real time: {results['real_time']:.4f} s")
    
    metrics = results['metrics']
    print(f"\nMetrics:")
    print(f"  Goodput: {metrics['goodput']:.2f} B/s ({metrics['goodput_mbps']:.4f} Mbps)")
    print(f"  Efficiency: {metrics['efficiency']*100:.2f}%")
    print(f"  Retransmissions: {metrics['retransmissions']}")
