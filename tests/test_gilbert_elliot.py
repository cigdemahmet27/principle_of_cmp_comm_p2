"""
Unit tests for the Gilbert-Elliot channel model.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.channel.gilbert_elliot import (
    GilbertElliottChannel, ChannelState,
    simulate_burst_pattern, analyze_burst_lengths
)


class TestGilbertElliottChannel:
    """Tests for Gilbert-Elliot channel model."""
    
    def test_initialization(self):
        """Test channel initialization with default parameters."""
        channel = GilbertElliottChannel(seed=42)
        
        assert channel.pg == 1e-6
        assert channel.pb == 5e-3
        assert channel.p_gb == 0.002
        assert channel.p_bg == 0.05
        assert channel.state in [ChannelState.GOOD, ChannelState.BAD]
    
    def test_steady_state_probabilities(self):
        """Test steady-state probability calculation."""
        channel = GilbertElliottChannel()
        
        pi_good, pi_bad = channel.get_steady_state_probabilities()
        
        # Should sum to 1
        assert abs(pi_good + pi_bad - 1.0) < 1e-10
        
        # Expected values: pi_good ≈ 0.962, pi_bad ≈ 0.038
        assert 0.95 < pi_good < 0.98
        assert 0.02 < pi_bad < 0.05
    
    def test_average_ber(self):
        """Test average BER calculation."""
        channel = GilbertElliottChannel()
        
        avg_ber = channel.get_average_ber()
        
        # Should be approximately 1e-4 as specified
        assert 5e-5 < avg_ber < 5e-4
    
    def test_single_bit_transmission(self):
        """Test single bit transmission."""
        channel = GilbertElliottChannel(seed=42)
        
        # Transmit many bits and check statistics
        errors = 0
        total = 10000
        
        for _ in range(total):
            if channel.transmit_bit():
                errors += 1
        
        # Observed BER should be in reasonable range
        observed_ber = errors / total
        assert 0 < observed_ber < 0.01  # Should be low but not zero
    
    def test_frame_transmission(self):
        """Test frame transmission."""
        channel = GilbertElliottChannel(seed=42)
        
        frame_size_bits = 1024 * 8  # 1 KB frame
        corrupted, bit_errors = channel.transmit_frame(frame_size_bits)
        
        assert isinstance(corrupted, bool)
        assert isinstance(bit_errors, int)
        assert bit_errors >= 0
        assert corrupted == (bit_errors > 0)
    
    def test_state_transitions(self):
        """Test that state transitions occur."""
        channel = GilbertElliottChannel(seed=42)
        
        # Force many transitions
        states_seen = set()
        for _ in range(1000):
            channel.transition_state()
            states_seen.add(channel.state)
        
        # Should see both states
        assert len(states_seen) == 2
    
    def test_burst_pattern_simulation(self):
        """Test burst pattern simulation."""
        channel = GilbertElliottChannel(seed=42)
        
        error_pattern = simulate_burst_pattern(channel, 100, 1024)
        
        assert len(error_pattern) == 100
        assert all(isinstance(e, bool) for e in error_pattern)
    
    def test_burst_analysis(self):
        """Test burst length analysis."""
        # Create a pattern with known bursts
        pattern = [False, False, True, True, True, False, True, False]
        
        stats = analyze_burst_lengths(pattern)
        
        assert stats['num_bursts'] == 2
        assert stats['max_burst_length'] == 3
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        channel = GilbertElliottChannel(seed=42)
        
        # Transmit some frames
        for _ in range(100):
            channel.transmit_frame(1024 * 8)
        
        stats = channel.get_statistics()
        
        assert stats['total_bits'] > 0
        assert 'observed_ber' in stats
        assert 'state_transitions' in stats
    
    def test_reset(self):
        """Test channel reset."""
        channel = GilbertElliottChannel(seed=42)
        
        # Transmit and accumulate stats
        for _ in range(100):
            channel.transmit_frame(1024 * 8)
        
        # Reset
        channel.reset(seed=123)
        
        stats = channel.get_statistics()
        assert stats['total_bits'] == 0
        assert stats['bit_errors'] == 0
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        channel1 = GilbertElliottChannel(seed=42)
        channel2 = GilbertElliottChannel(seed=42)
        
        results1 = [channel1.transmit_frame(1024 * 8) for _ in range(10)]
        results2 = [channel2.transmit_frame(1024 * 8) for _ in range(10)]
        
        assert results1 == results2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
