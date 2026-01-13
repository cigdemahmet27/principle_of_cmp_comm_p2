"""
Metrics Collection and Calculation

This module provides utilities for calculating and tracking
performance metrics including Goodput, utilization, and RTT.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import deque
import statistics
import sys
sys.path.insert(0, '..')
from config import BIT_RATE


@dataclass
class MetricsSample:
    """Single sample of metrics at a point in time."""
    timestamp: float
    bytes_delivered: int = 0
    bytes_sent: int = 0
    frames_sent: int = 0
    frames_acked: int = 0
    retransmissions: int = 0
    current_window_size: int = 0


class MetricsCollector:
    """
    Collects and calculates performance metrics for the simulation.
    
    Primary metric: Goodput = Delivered Application Bytes / Total Transmission Time
    
    Attributes:
        start_time: Simulation start time
        end_time: Simulation end time
        bit_rate: Channel bit rate
    """
    
    def __init__(self, bit_rate: float = BIT_RATE):
        """
        Initialize metrics collector.
        
        Args:
            bit_rate: Channel bit rate in bps
        """
        self.bit_rate = bit_rate
        
        # Time tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Byte counters
        self.application_bytes_sent = 0  # Original data bytes
        self.application_bytes_delivered = 0
        self.total_bytes_transmitted = 0  # Including headers and retransmits
        self.header_bytes = 0
        
        # Frame counters
        self.data_frames_sent = 0
        self.data_frames_delivered = 0
        self.ack_frames_sent = 0
        self.nak_frames_sent = 0
        self.retransmissions = 0
        
        # Error tracking
        self.frames_corrupted = 0
        self.frames_lost = 0
        
        # RTT samples
        self.rtt_samples: List[float] = []
        
        # Buffer events
        self.buffer_full_events = 0
        self.backpressure_events = 0
        
        # Per-interval samples
        self.samples: List[MetricsSample] = []
        self.sample_interval = 0.1  # seconds
        self.last_sample_time = 0.0
        
        # Window utilization tracking
        self.window_utilization_samples: List[float] = []
    
    def start(self, time: float):
        """
        Mark simulation start.
        
        Args:
            time: Start time
        """
        self.start_time = time
        self.last_sample_time = time
    
    def finish(self, time: float):
        """
        Mark simulation end.
        
        Args:
            time: End time
        """
        self.end_time = time
        self._take_sample(time)
    
    def record_data_sent(self, payload_bytes: int, total_frame_bytes: int):
        """
        Record data frame sent.
        
        Args:
            payload_bytes: Original payload bytes
            total_frame_bytes: Total frame bytes including headers
        """
        self.application_bytes_sent += payload_bytes
        self.total_bytes_transmitted += total_frame_bytes
        self.header_bytes += (total_frame_bytes - payload_bytes)
        self.data_frames_sent += 1
    
    def record_data_delivered(self, payload_bytes: int):
        """
        Record data successfully delivered.
        
        Args:
            payload_bytes: Delivered payload bytes
        """
        self.application_bytes_delivered += payload_bytes
        self.data_frames_delivered += 1
    
    def record_retransmission(self, total_frame_bytes: int):
        """
        Record frame retransmission.
        
        Args:
            total_frame_bytes: Retransmitted frame bytes
        """
        self.retransmissions += 1
        self.total_bytes_transmitted += total_frame_bytes
    
    def record_ack_sent(self, frame_bytes: int):
        """Record ACK frame sent."""
        self.ack_frames_sent += 1
        self.total_bytes_transmitted += frame_bytes
    
    def record_nak_sent(self, frame_bytes: int):
        """Record NAK frame sent."""
        self.nak_frames_sent += 1
        self.total_bytes_transmitted += frame_bytes
    
    def record_frame_corrupted(self):
        """Record frame corrupted by channel."""
        self.frames_corrupted += 1
    
    def record_frame_lost(self):
        """Record frame lost (timeout without ACK)."""
        self.frames_lost += 1
    
    def record_rtt(self, rtt: float):
        """
        Record RTT sample.
        
        Args:
            rtt: Round-trip time in seconds
        """
        self.rtt_samples.append(rtt)
    
    def record_buffer_full(self):
        """Record buffer full event."""
        self.buffer_full_events += 1
    
    def record_backpressure(self):
        """Record backpressure event."""
        self.backpressure_events += 1
    
    def record_window_utilization(self, used_slots: int, total_slots: int):
        """
        Record window utilization.
        
        Args:
            used_slots: Number of slots in use
            total_slots: Total window size
        """
        if total_slots > 0:
            utilization = used_slots / total_slots
            self.window_utilization_samples.append(utilization)
    
    def _take_sample(self, time: float):
        """Take a periodic sample of metrics."""
        sample = MetricsSample(
            timestamp=time,
            bytes_delivered=self.application_bytes_delivered,
            bytes_sent=self.total_bytes_transmitted,
            frames_sent=self.data_frames_sent,
            frames_acked=self.data_frames_delivered,
            retransmissions=self.retransmissions
        )
        self.samples.append(sample)
    
    def update(self, current_time: float):
        """
        Update metrics with current time.
        
        Args:
            current_time: Current simulation time
        """
        if current_time - self.last_sample_time >= self.sample_interval:
            self._take_sample(current_time)
            self.last_sample_time = current_time
    
    def calculate_goodput(self) -> float:
        """
        Calculate Goodput.
        
        Goodput = Delivered Application Bytes / Total Transmission Time
        
        Returns:
            Goodput in bytes per second
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        
        total_time = self.end_time - self.start_time
        if total_time <= 0:
            return 0.0
        
        return self.application_bytes_delivered / total_time
    
    def calculate_goodput_bps(self) -> float:
        """
        Calculate Goodput in bits per second.
        
        Returns:
            Goodput in bps
        """
        return self.calculate_goodput() * 8
    
    def calculate_throughput(self) -> float:
        """
        Calculate raw throughput.
        
        Throughput = Total Bytes Transmitted / Total Transmission Time
        
        Returns:
            Throughput in bytes per second
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        
        total_time = self.end_time - self.start_time
        if total_time <= 0:
            return 0.0
        
        return self.total_bytes_transmitted / total_time
    
    def calculate_efficiency(self) -> float:
        """
        Calculate transmission efficiency.
        
        Efficiency = Application Bytes Delivered / Total Bytes Transmitted
        
        Returns:
            Efficiency ratio (0-1)
        """
        if self.total_bytes_transmitted <= 0:
            return 0.0
        
        return self.application_bytes_delivered / self.total_bytes_transmitted
    
    def calculate_utilization(self) -> float:
        """
        Calculate channel utilization.
        
        Utilization = (Bytes Transmitted * 8) / (Bit Rate * Time)
        
        Returns:
            Utilization ratio (0-1)
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        
        total_time = self.end_time - self.start_time
        if total_time <= 0:
            return 0.0
        
        bits_transmitted = self.total_bytes_transmitted * 8
        capacity = self.bit_rate * total_time
        
        return bits_transmitted / capacity
    
    def calculate_frame_error_rate(self) -> float:
        """
        Calculate frame error rate.
        
        FER = Corrupted Frames / Total Frames Sent
        
        Returns:
            Frame error rate (0-1)
        """
        total_frames = self.data_frames_sent + self.retransmissions
        if total_frames <= 0:
            return 0.0
        
        return self.frames_corrupted / total_frames
    
    def calculate_retransmission_rate(self) -> float:
        """
        Calculate retransmission rate.
        
        Returns:
            Retransmissions / Original Frames Sent
        """
        if self.data_frames_sent <= 0:
            return 0.0
        
        return self.retransmissions / self.data_frames_sent
    
    def get_rtt_statistics(self) -> Dict[str, float]:
        """
        Get RTT statistics.
        
        Returns:
            Dictionary with min, max, mean, median, stdev RTT
        """
        if not self.rtt_samples:
            return {
                'min': 0, 'max': 0, 'mean': 0, 
                'median': 0, 'stdev': 0, 'samples': 0
            }
        
        return {
            'min': min(self.rtt_samples),
            'max': max(self.rtt_samples),
            'mean': statistics.mean(self.rtt_samples),
            'median': statistics.median(self.rtt_samples),
            'stdev': statistics.stdev(self.rtt_samples) if len(self.rtt_samples) > 1 else 0,
            'samples': len(self.rtt_samples)
        }
    
    def get_window_utilization_stats(self) -> Dict[str, float]:
        """
        Get window utilization statistics.
        
        Returns:
            Dictionary with utilization stats
        """
        if not self.window_utilization_samples:
            return {'mean': 0, 'max': 0, 'min': 0}
        
        return {
            'mean': statistics.mean(self.window_utilization_samples),
            'max': max(self.window_utilization_samples),
            'min': min(self.window_utilization_samples)
        }
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        total_time = 0
        if self.start_time is not None and self.end_time is not None:
            total_time = self.end_time - self.start_time
        
        return {
            # Time
            'total_time': total_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            
            # Primary metric
            'goodput': self.calculate_goodput(),
            'goodput_bps': self.calculate_goodput_bps(),
            'goodput_mbps': self.calculate_goodput_bps() / 1e6,
            
            # Secondary metrics
            'throughput': self.calculate_throughput(),
            'efficiency': self.calculate_efficiency(),
            'utilization': self.calculate_utilization(),
            
            # Byte counts
            'application_bytes_sent': self.application_bytes_sent,
            'application_bytes_delivered': self.application_bytes_delivered,
            'total_bytes_transmitted': self.total_bytes_transmitted,
            'header_overhead': self.header_bytes,
            
            # Frame counts
            'data_frames_sent': self.data_frames_sent,
            'data_frames_delivered': self.data_frames_delivered,
            'ack_frames_sent': self.ack_frames_sent,
            'nak_frames_sent': self.nak_frames_sent,
            'retransmissions': self.retransmissions,
            
            # Error metrics
            'frames_corrupted': self.frames_corrupted,
            'frame_error_rate': self.calculate_frame_error_rate(),
            'retransmission_rate': self.calculate_retransmission_rate(),
            
            # RTT
            'rtt': self.get_rtt_statistics(),
            
            # Buffer events
            'buffer_full_events': self.buffer_full_events,
            'backpressure_events': self.backpressure_events,
            
            # Window utilization
            'window_utilization': self.get_window_utilization_stats()
        }
    
    def to_csv_row(self) -> Dict:
        """
        Get metrics as a flat dictionary suitable for CSV export.
        
        Returns:
            Dictionary with flattened metrics
        """
        summary = self.get_summary()
        rtt = summary.pop('rtt')
        window_util = summary.pop('window_utilization')
        
        # Flatten nested dicts
        flat = {**summary}
        for key, value in rtt.items():
            flat[f'rtt_{key}'] = value
        for key, value in window_util.items():
            flat[f'window_util_{key}'] = value
        
        return flat
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.application_bytes_sent = 0
        self.application_bytes_delivered = 0
        self.total_bytes_transmitted = 0
        self.header_bytes = 0
        self.data_frames_sent = 0
        self.data_frames_delivered = 0
        self.ack_frames_sent = 0
        self.nak_frames_sent = 0
        self.retransmissions = 0
        self.frames_corrupted = 0
        self.frames_lost = 0
        self.rtt_samples.clear()
        self.buffer_full_events = 0
        self.backpressure_events = 0
        self.samples.clear()
        self.window_utilization_samples.clear()
        self.last_sample_time = 0.0


if __name__ == "__main__":
    # Test metrics collector
    print("=" * 60)
    print("METRICS COLLECTOR TEST")
    print("=" * 60)
    
    metrics = MetricsCollector()
    
    # Simulate a transfer
    metrics.start(0.0)
    
    # Simulate sending frames
    for i in range(100):
        metrics.record_data_sent(payload_bytes=1024, total_frame_bytes=1056)
        metrics.record_rtt(0.1 + i * 0.001)  # Varying RTT
        
        # Occasional retransmission
        if i % 10 == 0:
            metrics.record_retransmission(1056)
            metrics.record_frame_corrupted()
    
    # Simulate deliveries
    for i in range(100):
        metrics.record_data_delivered(1024)
    
    metrics.finish(10.0)
    
    # Print summary
    summary = metrics.get_summary()
    print("\nMetrics Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6f}")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
