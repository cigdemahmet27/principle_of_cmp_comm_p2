"""
Parameter Sweep Configuration

This module defines the parameter space for the exhaustive search
and provides utilities for parameter sweep analysis.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WINDOW_SIZES, PAYLOAD_SIZES, RUNS_PER_CONFIGURATION


@dataclass
class ParameterPoint:
    """A single point in the parameter space."""
    window_size: int
    payload_size: int
    
    @property
    def key(self) -> Tuple[int, int]:
        return (self.window_size, self.payload_size)


class ParameterSweep:
    """
    Parameter sweep configuration and analysis.
    
    Defines the parameter space:
    - W ∈ {2, 4, 8, 16, 32, 64}
    - L ∈ {128, 256, 512, 1024, 2048, 4096} bytes
    - 10 runs per configuration
    - Total = 6 × 6 × 10 = 360 simulations
    """
    
    def __init__(
        self,
        window_sizes: List[int] = None,
        payload_sizes: List[int] = None,
        runs_per_config: int = RUNS_PER_CONFIGURATION
    ):
        """
        Initialize parameter sweep.
        
        Args:
            window_sizes: List of window sizes
            payload_sizes: List of payload sizes
            runs_per_config: Number of runs per configuration
        """
        self.window_sizes = window_sizes or WINDOW_SIZES
        self.payload_sizes = payload_sizes or PAYLOAD_SIZES
        self.runs_per_config = runs_per_config
    
    @property
    def total_configurations(self) -> int:
        """Total number of (W, L) configurations."""
        return len(self.window_sizes) * len(self.payload_sizes)
    
    @property
    def total_simulations(self) -> int:
        """Total number of simulation runs."""
        return self.total_configurations * self.runs_per_config
    
    def get_all_points(self) -> List[ParameterPoint]:
        """Get all parameter points."""
        points = []
        for w in self.window_sizes:
            for l in self.payload_sizes:
                points.append(ParameterPoint(w, l))
        return points
    
    def analyze_tradeoffs(self) -> Dict:
        """
        Document the theoretical trade-offs for analysis.
        
        Returns:
            Dictionary with trade-off documentation
        """
        return {
            'frame_size_tradeoff': {
                'description': (
                    "Trade-off between large frame size (lower overhead) "
                    "and burst error sensitivity"
                ),
                'large_frame_pros': [
                    "Lower header overhead ratio",
                    "Fewer frames to transmit",
                    "Better utilization for good channels"
                ],
                'large_frame_cons': [
                    "Higher probability of frame corruption",
                    "More data lost per corrupted frame",
                    "Larger retransmission penalty"
                ],
                'analysis': (
                    "With burst errors, larger frames are more likely to encounter "
                    "the bad channel state during transmission. The probability of "
                    "at least one bit error increases with frame size. When bursts "
                    "occur, losing a large frame means retransmitting more data."
                )
            },
            'window_size_tradeoff': {
                'description': (
                    "Trade-off between large windows (pipeline utilization) "
                    "and burst-loss amplification"
                ),
                'large_window_pros': [
                    "Better pipeline utilization",
                    "Higher throughput potential",
                    "More frames in flight during propagation"
                ],
                'large_window_cons': [
                    "More frames lost during burst errors",
                    "Larger buffer requirements",
                    "Potentially more retransmissions"
                ],
                'analysis': (
                    "A larger window keeps more frames in flight, which is good "
                    "for utilizing the bandwidth-delay product. However, during "
                    "burst errors, multiple consecutive frames may be corrupted, "
                    "requiring many retransmissions and reducing effective throughput."
                )
            },
            'optimal_region': {
                'description': (
                    "The optimal (W, L) pair balances these trade-offs"
                ),
                'factors': [
                    "RTT determines minimum window for full utilization",
                    "BER and burst length affect optimal frame size",
                    "Buffer capacity limits practical window size",
                    "Retransmission overhead affects efficiency"
                ],
                'expected_optimal': (
                    "Expected optimal region: moderate window size (8-16) that "
                    "covers the bandwidth-delay product without excessive burst "
                    "sensitivity, and moderate frame size (512-1024) that balances "
                    "overhead with error probability."
                )
            }
        }
    
    def get_bandwidth_delay_product(self, bit_rate: float, rtt: float) -> float:
        """
        Calculate bandwidth-delay product.
        
        BDP = Bit Rate × RTT (in bits)
        
        Args:
            bit_rate: Channel bit rate in bps
            rtt: Round-trip time in seconds
            
        Returns:
            BDP in bits
        """
        return bit_rate * rtt
    
    def calculate_optimal_window(
        self,
        bit_rate: float,
        rtt: float,
        frame_size: int
    ) -> int:
        """
        Calculate optimal window size for full pipeline utilization.
        
        W_opt = BDP / Frame Size (in frames)
        
        Args:
            bit_rate: Channel bit rate in bps
            rtt: Round-trip time in seconds
            frame_size: Frame size in bytes
            
        Returns:
            Optimal window size
        """
        bdp_bits = self.get_bandwidth_delay_product(bit_rate, rtt)
        bdp_bytes = bdp_bits / 8
        optimal = bdp_bytes / frame_size
        return max(1, int(optimal))
    
    def estimate_frame_error_probability(
        self,
        frame_size_bits: int,
        ber: float
    ) -> float:
        """
        Estimate frame error probability for independent errors.
        
        P_frame_error = 1 - (1 - BER)^frame_size
        
        Args:
            frame_size_bits: Frame size in bits
            ber: Bit error rate
            
        Returns:
            Frame error probability
        """
        return 1 - (1 - ber) ** frame_size_bits
    
    @staticmethod
    def load_results(filepath: str) -> List[Dict]:
        """
        Load results from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of result dictionaries
        """
        results = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    try:
                        if '.' in row[key]:
                            row[key] = float(row[key])
                        else:
                            row[key] = int(row[key])
                    except (ValueError, TypeError):
                        pass
                results.append(row)
        return results
    
    @staticmethod
    def create_goodput_matrix(results: List[Dict]) -> Dict:
        """
        Create a matrix of mean goodput values.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with goodput matrix data
        """
        import statistics
        
        # Group by (W, L)
        grouped = {}
        for r in results:
            key = (r['window_size'], r['payload_size'])
            if key not in grouped:
                grouped[key] = []
            if isinstance(r.get('goodput'), (int, float)):
                grouped[key].append(r['goodput'])
        
        # Calculate means
        matrix = {}
        for key, goodputs in grouped.items():
            if goodputs:
                matrix[key] = {
                    'mean': statistics.mean(goodputs),
                    'std': statistics.stdev(goodputs) if len(goodputs) > 1 else 0,
                    'n': len(goodputs)
                }
        
        return matrix


if __name__ == "__main__":
    # Document parameter sweep
    print("=" * 60)
    print("PARAMETER SWEEP CONFIGURATION")
    print("=" * 60)
    
    sweep = ParameterSweep()
    
    print(f"\nParameter Space:")
    print(f"  Window Sizes (W): {sweep.window_sizes}")
    print(f"  Payload Sizes (L): {sweep.payload_sizes}")
    print(f"  Runs per config: {sweep.runs_per_config}")
    print(f"  Total configurations: {sweep.total_configurations}")
    print(f"  Total simulations: {sweep.total_simulations}")
    
    print("\n" + "=" * 60)
    print("TRADE-OFF ANALYSIS")
    print("=" * 60)
    
    tradeoffs = sweep.analyze_tradeoffs()
    
    for key, analysis in tradeoffs.items():
        print(f"\n{key.upper()}:")
        print(f"  {analysis['description']}")
        
        if 'pros' in str(analysis):
            print(f"\n  Pros (large values):")
            for pro in analysis.get('large_frame_pros') or analysis.get('large_window_pros', []):
                print(f"    + {pro}")
            print(f"\n  Cons (large values):")
            for con in analysis.get('large_frame_cons') or analysis.get('large_window_cons', []):
                print(f"    - {con}")
        
        if 'analysis' in analysis:
            print(f"\n  Analysis: {analysis['analysis']}")
        
        if 'expected_optimal' in analysis:
            print(f"\n  {analysis['expected_optimal']}")
    
    # Calculate some theoretical values
    print("\n" + "=" * 60)
    print("THEORETICAL CALCULATIONS")
    print("=" * 60)
    
    from config import BIT_RATE
    from src.layers.physical_layer import calculate_theoretical_rtt
    
    for payload in sweep.payload_sizes:
        rtt = calculate_theoretical_rtt(payload)
        bdp = sweep.get_bandwidth_delay_product(BIT_RATE, rtt) / 8  # bytes
        opt_window = sweep.calculate_optimal_window(BIT_RATE, rtt, payload + 32)  # +headers
        
        print(f"\n  Payload {payload} bytes:")
        print(f"    RTT: {rtt*1000:.2f} ms")
        print(f"    BDP: {bdp:.0f} bytes")
        print(f"    Optimal window: ~{opt_window} frames")
