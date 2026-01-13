"""
Unit tests for the Selective Repeat ARQ protocol.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arq.frame import Frame, FrameType, FrameBuffer
from src.arq.sender import SRSender
from src.arq.receiver import SRReceiver
from src.arq.timer import TimerManager, FrameTimer, AdaptiveTimeoutCalculator


class TestFrame:
    """Tests for Frame class."""
    
    def test_data_frame_creation(self):
        """Test creating a data frame."""
        payload = b"Hello, World!"
        frame = Frame.create_data_frame(seq_num=42, payload=payload)
        
        assert frame.frame_type == FrameType.DATA
        assert frame.seq_num == 42
        assert frame.payload == payload
    
    def test_ack_frame_creation(self):
        """Test creating an ACK frame."""
        frame = Frame.create_ack_frame(ack_num=10)
        
        assert frame.frame_type == FrameType.ACK
        assert frame.ack_num == 10
    
    def test_nak_frame_creation(self):
        """Test creating a NAK frame."""
        frame = Frame.create_nak_frame(nak_num=5)
        
        assert frame.frame_type == FrameType.NAK
        assert frame.ack_num == 5
    
    def test_serialization_deserialization(self):
        """Test frame serialization and deserialization."""
        payload = b"Test payload data"
        original = Frame.create_data_frame(seq_num=100, payload=payload)
        
        serialized = original.serialize()
        deserialized, crc_valid = Frame.deserialize(serialized)
        
        assert deserialized is not None
        assert crc_valid
        assert deserialized.seq_num == original.seq_num
        assert deserialized.payload == original.payload
    
    def test_crc_detection(self):
        """Test that CRC detects corruption."""
        frame = Frame.create_data_frame(seq_num=1, payload=b"data")
        serialized = bytearray(frame.serialize())
        
        # Corrupt a byte
        serialized[10] ^= 0xFF
        
        deserialized, crc_valid = Frame.deserialize(bytes(serialized))
        
        assert not crc_valid
    
    def test_frame_flags(self):
        """Test frame flag operations."""
        frame = Frame.create_data_frame(seq_num=1, payload=b"data")
        
        assert not frame.is_last_frame()
        frame.set_last_frame()
        assert frame.is_last_frame()
        
        assert not frame.is_retransmission()
        frame.set_retransmission()
        assert frame.is_retransmission()
    
    def test_total_size(self):
        """Test frame size calculation."""
        payload = b"x" * 100
        frame = Frame.create_data_frame(seq_num=1, payload=payload)
        
        assert frame.total_size == Frame.HEADER_SIZE + 100


class TestFrameBuffer:
    """Tests for FrameBuffer class."""
    
    def test_buffer_add_get(self):
        """Test adding and getting frames."""
        buffer = FrameBuffer(max_size=5)
        
        frame = Frame.create_data_frame(seq_num=1, payload=b"data")
        buffer.add(frame)
        
        retrieved = buffer.get(1)
        assert retrieved is not None
        assert retrieved.seq_num == 1
    
    def test_buffer_full(self):
        """Test buffer full detection."""
        buffer = FrameBuffer(max_size=3)
        
        for i in range(3):
            frame = Frame.create_data_frame(seq_num=i, payload=b"data")
            assert buffer.add(frame)
        
        assert buffer.is_full
        
        frame = Frame.create_data_frame(seq_num=10, payload=b"data")
        assert not buffer.add(frame)
    
    def test_buffer_remove(self):
        """Test removing frames."""
        buffer = FrameBuffer(max_size=5)
        
        frame = Frame.create_data_frame(seq_num=1, payload=b"data")
        buffer.add(frame)
        
        removed = buffer.remove(1)
        assert removed is not None
        assert not buffer.contains(1)


class TestSRSender:
    """Tests for Selective Repeat Sender."""
    
    def test_queue_data(self):
        """Test queuing data for transmission."""
        sender = SRSender(window_size=4, timeout=0.5)
        
        sender.queue_data(b"segment1")
        sender.queue_data(b"segment2")
        
        assert sender.can_send()
    
    def test_send_frame(self):
        """Test sending a frame."""
        sender = SRSender(window_size=4, timeout=0.5)
        sender.queue_data(b"data")
        
        frame = sender.send_next_frame(current_time=0.0)
        
        assert frame is not None
        assert frame.seq_num == 0
        assert sender.frames_sent == 1
    
    def test_window_full(self):
        """Test window becoming full."""
        sender = SRSender(window_size=2, timeout=0.5)
        
        for i in range(5):
            sender.queue_data(f"segment{i}".encode())
        
        # Send until window is full
        sender.send_next_frame(0.0)
        sender.send_next_frame(0.0)
        
        assert not sender.can_send() or sender.window.is_full
    
    def test_ack_processing(self):
        """Test ACK processing."""
        sender = SRSender(window_size=4, timeout=0.5)
        sender.queue_data(b"data")
        sender.send_next_frame(0.0)
        
        # Process ACK
        result = sender.process_ack(0, 0.1)
        
        assert result
        assert sender.frames_acked == 1
    
    def test_timeout_detection(self):
        """Test timeout detection."""
        sender = SRSender(window_size=4, timeout=0.1)
        sender.queue_data(b"data")
        sender.send_next_frame(0.0)
        
        # Check timeouts after timeout period
        expired = sender.check_timeouts(0.2)
        
        assert len(expired) > 0


class TestSRReceiver:
    """Tests for Selective Repeat Receiver."""
    
    def test_receive_in_order(self):
        """Test receiving frames in order."""
        delivered = []
        
        def on_delivered(data, seq):
            delivered.append((seq, data))
        
        receiver = SRReceiver(window_size=4, on_data_delivered=on_delivered)
        
        for i in range(3):
            frame = Frame.create_data_frame(seq_num=i, payload=f"data{i}".encode())
            receiver.receive_frame(frame)
        
        assert len(delivered) == 3
        assert delivered[0][0] == 0
    
    def test_receive_out_of_order(self):
        """Test receiving frames out of order."""
        delivered = []
        
        def on_delivered(data, seq):
            delivered.append((seq, data))
        
        receiver = SRReceiver(window_size=4, on_data_delivered=on_delivered)
        
        # Receive frame 2 first (out of order)
        receiver.receive_frame(Frame.create_data_frame(2, b"data2"))
        assert len(delivered) == 0  # Not delivered yet
        
        # Receive frame 0
        receiver.receive_frame(Frame.create_data_frame(0, b"data0"))
        assert len(delivered) == 1  # Frame 0 delivered
        
        # Receive frame 1
        receiver.receive_frame(Frame.create_data_frame(1, b"data1"))
        assert len(delivered) == 3  # All three now delivered
    
    def test_ack_generation(self):
        """Test ACK generation."""
        receiver = SRReceiver(window_size=4)
        
        frame = Frame.create_data_frame(seq_num=0, payload=b"data")
        ack = receiver.receive_frame(frame)
        
        assert ack is not None
        assert ack.frame_type == FrameType.ACK
        assert ack.ack_num == 0
    
    def test_duplicate_detection(self):
        """Test duplicate frame detection."""
        receiver = SRReceiver(window_size=4)
        
        frame = Frame.create_data_frame(seq_num=0, payload=b"data")
        
        receiver.receive_frame(frame)
        receiver.receive_frame(frame)  # Duplicate
        
        assert receiver.duplicate_frames == 1


class TestTimerManager:
    """Tests for timer management."""
    
    def test_timer_start(self):
        """Test starting a timer."""
        manager = TimerManager(default_timeout=1.0)
        
        manager.start_timer(seq_num=0, current_time=0.0)
        
        assert manager.get_active_count() == 1
    
    def test_timer_expiry(self):
        """Test timer expiry detection."""
        manager = TimerManager(default_timeout=1.0)
        
        manager.start_timer(seq_num=0, current_time=0.0)
        
        # Check before expiry
        expired = manager.check_timeouts(0.5)
        assert len(expired) == 0
        
        # Check after expiry
        expired = manager.check_timeouts(1.5)
        assert 0 in expired
    
    def test_timer_stop(self):
        """Test stopping a timer."""
        manager = TimerManager(default_timeout=1.0)
        
        manager.start_timer(seq_num=0, current_time=0.0)
        manager.stop_timer(0)
        
        # Should not expire after stopping
        expired = manager.check_timeouts(1.5)
        assert len(expired) == 0


class TestAdaptiveTimeout:
    """Tests for adaptive timeout calculator."""
    
    def test_first_sample(self):
        """Test first RTT sample."""
        calc = AdaptiveTimeoutCalculator(initial_rto=1.0)
        
        calc.update(0.1)
        
        assert calc.srtt == 0.1
        assert calc.rttvar == 0.05
    
    def test_multiple_samples(self):
        """Test multiple RTT samples."""
        calc = AdaptiveTimeoutCalculator(initial_rto=1.0)
        
        for rtt in [0.1, 0.12, 0.09, 0.11]:
            calc.update(rtt)
        
        # SRTT should be around the average
        assert 0.08 < calc.srtt < 0.14
    
    def test_backoff(self):
        """Test exponential backoff."""
        calc = AdaptiveTimeoutCalculator(initial_rto=1.0)
        calc.update(0.1)
        
        initial_rto = calc.get_rto()
        calc.backoff()
        
        assert calc.get_rto() >= initial_rto * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
