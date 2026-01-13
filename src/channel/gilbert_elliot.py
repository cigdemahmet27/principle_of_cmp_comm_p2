"""
Gilbert-Elliott Burst Error Channel Model

This module implements the two-state Markov chain model for simulating
burst errors in communication channels. The channel alternates between
a "Good" state (low BER) and a "Bad" state (high BER).
"""

import numpy as np
from enum import Enum
from typing import Tuple, List, Optional
import sys
sys.path.insert(0, '..')
from config import (
    GOOD_STATE_BER, BAD_STATE_BER,
    P_GOOD_TO_BAD, P_BAD_TO_GOOD
)


class ChannelState(Enum):
    """Channel state enumeration."""
    GOOD = 0
    BAD = 1


class GilbertElliottChannel:
    """
    Gilbert-Elliott two-state Markov channel model.
    
    The channel transitions between Good and Bad states with specified
    probabilities. Each state has its own bit error rate (BER).
    
    Attributes:
        pg: Bit error rate in Good state
        pb: Bit error rate in Bad state
        p_gb: Transition probability from Good to Bad
        p_bg: Transition probability from Bad to Good
        state: Current channel state
        rng: Random number generator
    """
    
    def __init__(
        self,
        pg: float = GOOD_STATE_BER,
        pb: float = BAD_STATE_BER,
        p_gb: float = P_GOOD_TO_BAD,
        p_bg: float = P_BAD_TO_GOOD,
        seed: Optional[int] = None
    ):
        """
        Initialize the Gilbert-Elliott channel.
        
        Args:
            pg: Bit error rate in Good state (default from config)
            pb: Bit error rate in Bad state (default from config)
            p_gb: Probability of transitioning from Good to Bad
            p_bg: Probability of transitioning from Bad to Good
            seed: Random seed for reproducibility
        """
        self.pg = pg
        self.pb = pb
        self.p_gb = p_gb
        self.p_bg = p_bg
        
        # Initialize RNG
        self.rng = np.random.default_rng(seed)
        
        # Start in steady-state (probabilistically)
        self._initialize_state()
        
        # Statistics tracking
        self.total_bits_transmitted = 0
        self.total_bit_errors = 0
        self.state_transitions = 0
        self.time_in_good = 0
        self.time_in_bad = 0
        
    def _initialize_state(self):
        """Initialize channel state based on steady-state probabilities."""
        pi_good, pi_bad = self.get_steady_state_probabilities()
        if self.rng.random() < pi_good:
            self.state = ChannelState.GOOD
        else:
            self.state = ChannelState.BAD
    
    def get_steady_state_probabilities(self) -> Tuple[float, float]:
        """
        Calculate steady-state probabilities for Good and Bad states.
        
        Returns:
            Tuple of (π_Good, π_Bad)
        """
        sum_transitions = self.p_gb + self.p_bg
        pi_good = self.p_bg / sum_transitions
        pi_bad = self.p_gb / sum_transitions
        return pi_good, pi_bad
    
    def get_average_ber(self) -> float:
        """
        Calculate average BER based on steady-state probabilities.
        
        Returns:
            Average bit error rate
        """
        pi_good, pi_bad = self.get_steady_state_probabilities()
        return pi_good * self.pg + pi_bad * self.pb
    
    def get_current_ber(self) -> float:
        """Get the BER for the current channel state."""
        return self.pg if self.state == ChannelState.GOOD else self.pb
    
    def transition_state(self):
        """
        Perform a state transition based on transition probabilities.
        
        This should be called once per bit or per frame depending on
        the desired granularity.
        """
        if self.state == ChannelState.GOOD:
            self.time_in_good += 1
            if self.rng.random() < self.p_gb:
                self.state = ChannelState.BAD
                self.state_transitions += 1
        else:
            self.time_in_bad += 1
            if self.rng.random() < self.p_bg:
                self.state = ChannelState.GOOD
                self.state_transitions += 1
    
    def transmit_bit(self) -> bool:
        """
        Simulate transmission of a single bit through the channel.
        
        Returns:
            True if bit was corrupted (error), False if successful
        """
        # Get current BER
        ber = self.get_current_ber()
        
        # Determine if error occurred
        error = self.rng.random() < ber
        
        # Update statistics
        self.total_bits_transmitted += 1
        if error:
            self.total_bit_errors += 1
        
        # Transition state after each bit
        self.transition_state()
        
        return error
    
    def transmit_frame(self, frame_size_bits: int) -> Tuple[bool, int]:
        """
        Simulate transmission of a frame through the channel.
        
        Uses TRUE per-bit Markov chain:
        - Each bit: check for error based on current state BER
        - Each bit: possibly transition to other state
        
        This models burst errors correctly:
        - When in BAD state, consecutive bits are corrupted
        - State transitions happen bit-by-bit
        
        Args:
            frame_size_bits: Size of the frame in bits
            
        Returns:
            Tuple of (frame_corrupted, number_of_bit_errors)
        """
        bit_errors = 0
        
        # Simulate bit-by-bit (optimized with batching)
        for _ in range(frame_size_bits):
            # Get current BER
            ber = self.pg if self.state == ChannelState.GOOD else self.pb
            
            # Check if this bit has an error
            if self.rng.random() < ber:
                bit_errors += 1
            
            # Update statistics
            self.total_bits_transmitted += 1
            if self.state == ChannelState.GOOD:
                self.time_in_good += 1
            else:
                self.time_in_bad += 1
            
            # State transition for this bit
            if self.state == ChannelState.GOOD:
                if self.rng.random() < self.p_gb:
                    self.state = ChannelState.BAD
                    self.state_transitions += 1
            else:
                if self.rng.random() < self.p_bg:
                    self.state = ChannelState.GOOD
                    self.state_transitions += 1
        
        self.total_bit_errors += bit_errors
        frame_corrupted = bit_errors > 0
        return frame_corrupted, bit_errors
    
    def transmit_frame_fast(self, frame_size_bits: int) -> Tuple[bool, int]:
        """
        Fast vectorized frame transmission simulation.
        
        More efficient for large frames but less accurate for state
        transitions (transitions happen per-frame instead of per-bit).
        
        Args:
            frame_size_bits: Size of the frame in bits
            
        Returns:
            Tuple of (frame_corrupted, number_of_bit_errors)
        """
        # Estimate number of bits in each state based on average durations
        avg_good_duration = 1 / self.p_gb if self.p_gb > 0 else float('inf')
        avg_bad_duration = 1 / self.p_bg if self.p_bg > 0 else float('inf')
        
        bits_remaining = frame_size_bits
        bit_errors = 0
        
        while bits_remaining > 0:
            # Determine how many bits to process in current state
            if self.state == ChannelState.GOOD:
                expected_bits = min(bits_remaining, int(self.rng.exponential(avg_good_duration) + 1))
                ber = self.pg
            else:
                expected_bits = min(bits_remaining, int(self.rng.exponential(avg_bad_duration) + 1))
                ber = self.pb
            
            # Generate errors for this burst
            errors_in_burst = self.rng.binomial(expected_bits, ber)
            bit_errors += errors_in_burst
            bits_remaining -= expected_bits
            
            # Update statistics
            self.total_bits_transmitted += expected_bits
            self.total_bit_errors += errors_in_burst
            
            # Transition state
            self.transition_state()
        
        frame_corrupted = bit_errors > 0
        return frame_corrupted, bit_errors
    
    def check_frame_crc(self, frame_size_bytes: int) -> bool:
        """
        Simulate CRC check for a frame.
        
        Args:
            frame_size_bytes: Size of the frame in bytes
            
        Returns:
            True if frame passed CRC (no errors), False if corrupted
        """
        frame_corrupted, _ = self.transmit_frame(frame_size_bytes * 8)
        return not frame_corrupted
    
    def get_statistics(self) -> dict:
        """
        Get channel statistics.
        
        Returns:
            Dictionary with transmission statistics
        """
        total_time = self.time_in_good + self.time_in_bad
        
        return {
            'total_bits': self.total_bits_transmitted,
            'bit_errors': self.total_bit_errors,
            'observed_ber': (self.total_bit_errors / self.total_bits_transmitted 
                           if self.total_bits_transmitted > 0 else 0),
            'state_transitions': self.state_transitions,
            'time_in_good': self.time_in_good,
            'time_in_bad': self.time_in_bad,
            'fraction_in_good': (self.time_in_good / total_time 
                                if total_time > 0 else 0),
            'theoretical_avg_ber': self.get_average_ber()
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.total_bits_transmitted = 0
        self.total_bit_errors = 0
        self.state_transitions = 0
        self.time_in_good = 0
        self.time_in_bad = 0
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the channel to initial state.
        
        Args:
            seed: New random seed (optional)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._initialize_state()
        self.reset_statistics()


# Utility functions
def simulate_burst_pattern(
    channel: GilbertElliottChannel,
    num_frames: int,
    frame_size_bytes: int
) -> List[bool]:
    """
    Simulate transmission of multiple frames and return error pattern.
    
    Args:
        channel: Gilbert-Elliott channel instance
        num_frames: Number of frames to simulate
        frame_size_bytes: Size of each frame in bytes
        
    Returns:
        List of booleans (True = frame corrupted)
    """
    frame_size_bits = frame_size_bytes * 8
    error_pattern = []
    
    for _ in range(num_frames):
        corrupted, _ = channel.transmit_frame(frame_size_bits)
        error_pattern.append(corrupted)
    
    return error_pattern


def analyze_burst_lengths(error_pattern: List[bool]) -> dict:
    """
    Analyze burst lengths in an error pattern.
    
    Args:
        error_pattern: List of frame error indicators
        
    Returns:
        Dictionary with burst statistics
    """
    if not error_pattern:
        return {'avg_burst_length': 0, 'max_burst_length': 0, 'num_bursts': 0}
    
    bursts = []
    current_burst = 0
    
    for error in error_pattern:
        if error:
            current_burst += 1
        elif current_burst > 0:
            bursts.append(current_burst)
            current_burst = 0
    
    if current_burst > 0:
        bursts.append(current_burst)
    
    if bursts:
        return {
            'avg_burst_length': np.mean(bursts),
            'max_burst_length': max(bursts),
            'num_bursts': len(bursts),
            'burst_lengths': bursts
        }
    return {'avg_burst_length': 0, 'max_burst_length': 0, 'num_bursts': 0}


if __name__ == "__main__":
    # Test the channel model
    print("=" * 60)
    print("GILBERT-ELLIOT CHANNEL MODEL TEST")
    print("=" * 60)
    
    channel = GilbertElliottChannel(seed=42)
    
    print(f"\nChannel Parameters:")
    print(f"  Good State BER: {channel.pg:.2e}")
    print(f"  Bad State BER: {channel.pb:.2e}")
    print(f"  P(G→B): {channel.p_gb}")
    print(f"  P(B→G): {channel.p_bg}")
    
    pi_good, pi_bad = channel.get_steady_state_probabilities()
    print(f"\nSteady-State Probabilities:")
    print(f"  π(Good): {pi_good:.4f}")
    print(f"  π(Bad): {pi_bad:.4f}")
    print(f"  Theoretical Avg BER: {channel.get_average_ber():.2e}")
    
    # Simulate frame transmissions
    num_frames = 10000
    frame_size = 1024  # bytes
    
    print(f"\nSimulating {num_frames} frames of {frame_size} bytes each...")
    
    error_pattern = simulate_burst_pattern(channel, num_frames, frame_size)
    
    stats = channel.get_statistics()
    burst_stats = analyze_burst_lengths(error_pattern)
    
    print(f"\nResults:")
    print(f"  Total bits transmitted: {stats['total_bits']:,}")
    print(f"  Total bit errors: {stats['bit_errors']:,}")
    print(f"  Observed BER: {stats['observed_ber']:.2e}")
    print(f"  Theoretical BER: {stats['theoretical_avg_ber']:.2e}")
    print(f"  Frame Error Rate: {sum(error_pattern)/len(error_pattern):.4f}")
    
    print(f"\nBurst Analysis:")
    print(f"  Number of error bursts: {burst_stats['num_bursts']}")
    print(f"  Average burst length: {burst_stats['avg_burst_length']:.2f} frames")
    print(f"  Maximum burst length: {burst_stats['max_burst_length']} frames")
