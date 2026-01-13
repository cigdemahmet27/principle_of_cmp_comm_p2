"""
Simulation Logger

This module provides logging utilities for the simulation,
with configurable verbosity levels and structured output.
"""

from typing import Optional, TextIO
from datetime import datetime
from enum import IntEnum
import sys
import os
sys.path.insert(0, '..')
from config import DEFAULT_LOG_LEVEL, OUTPUT_DIR


class LogLevel(IntEnum):
    """Log level enumeration."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class SimulationLogger:
    """
    Logger for simulation events.
    
    Provides structured logging with timestamps and categories.
    
    Attributes:
        name: Logger name
        level: Minimum log level
        file: Optional file for logging
    """
    
    # Color codes for terminal output
    COLORS = {
        LogLevel.DEBUG: '\033[36m',     # Cyan
        LogLevel.INFO: '\033[32m',      # Green
        LogLevel.WARNING: '\033[33m',   # Yellow
        LogLevel.ERROR: '\033[31m',     # Red
        LogLevel.CRITICAL: '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(
        self,
        name: str = "Simulator",
        level: int = DEFAULT_LOG_LEVEL,
        log_file: Optional[str] = None,
        use_colors: bool = True,
        include_timestamp: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Minimum log level
            log_file: Optional file path for logging
            use_colors: Use ANSI colors in output
            include_timestamp: Include timestamps in log messages
        """
        self.name = name
        self.level = level
        self.use_colors = use_colors
        self.include_timestamp = include_timestamp
        
        self.file: Optional[TextIO] = None
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.file = open(log_file, 'w')
        
        # Simulation time tracking
        self.sim_time: Optional[float] = None
        
        # Message counts
        self.message_counts = {level: 0 for level in LogLevel}
    
    def set_sim_time(self, time: float):
        """Set current simulation time for log messages."""
        self.sim_time = time
    
    def set_level(self, level: int):
        """Set minimum log level."""
        self.level = level
    
    def _format_message(
        self,
        level: LogLevel,
        message: str,
        category: Optional[str] = None
    ) -> str:
        """Format a log message."""
        parts = []
        
        # Timestamp
        if self.include_timestamp:
            if self.sim_time is not None:
                parts.append(f"[{self.sim_time:10.6f}s]")
            else:
                parts.append(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]")
        
        # Level
        level_str = level.name.ljust(8)
        if self.use_colors:
            level_str = f"{self.COLORS[level]}{level_str}{self.RESET}"
        parts.append(level_str)
        
        # Name
        parts.append(f"[{self.name}]")
        
        # Category
        if category:
            parts.append(f"[{category}]")
        
        # Message
        parts.append(message)
        
        return " ".join(parts)
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        category: Optional[str] = None
    ):
        """Log a message."""
        if level < self.level:
            return
        
        self.message_counts[level] += 1
        formatted = self._format_message(level, message, category)
        
        print(formatted)
        
        if self.file:
            # Strip color codes for file
            clean = formatted
            for color in self.COLORS.values():
                clean = clean.replace(color, '')
            clean = clean.replace(self.RESET, '')
            self.file.write(clean + '\n')
            self.file.flush()
    
    def debug(self, message: str, category: Optional[str] = None):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, category)
    
    def info(self, message: str, category: Optional[str] = None):
        """Log info message."""
        self._log(LogLevel.INFO, message, category)
    
    def warning(self, message: str, category: Optional[str] = None):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, category)
    
    def error(self, message: str, category: Optional[str] = None):
        """Log error message."""
        self._log(LogLevel.ERROR, message, category)
    
    def critical(self, message: str, category: Optional[str] = None):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, category)
    
    # Convenience methods for simulation events
    def frame_sent(self, seq_num: int, frame_type: str, size: int):
        """Log frame sent event."""
        self.debug(f"Frame {seq_num} ({frame_type}) sent, size={size}B", "TX")
    
    def frame_received(self, seq_num: int, frame_type: str, valid: bool):
        """Log frame received event."""
        status = "OK" if valid else "CORRUPTED"
        self.debug(f"Frame {seq_num} ({frame_type}) received, {status}", "RX")
    
    def ack_sent(self, ack_num: int):
        """Log ACK sent event."""
        self.debug(f"ACK {ack_num} sent", "ACK")
    
    def ack_received(self, ack_num: int):
        """Log ACK received event."""
        self.debug(f"ACK {ack_num} received", "ACK")
    
    def timeout(self, seq_num: int, retransmit_count: int):
        """Log timeout event."""
        self.warning(f"Timeout for frame {seq_num} (retx #{retransmit_count})", "TIMEOUT")
    
    def retransmit(self, seq_num: int):
        """Log retransmission event."""
        self.info(f"Retransmitting frame {seq_num}", "RETX")
    
    def channel_state(self, state: str, ber: float):
        """Log channel state change."""
        self.debug(f"Channel state: {state}, BER={ber:.2e}", "CHANNEL")
    
    def buffer_event(self, event_type: str, fill_level: float):
        """Log buffer event."""
        level = LogLevel.WARNING if fill_level > 0.9 else LogLevel.DEBUG
        self._log(level, f"Buffer {event_type}, fill={fill_level*100:.1f}%", "BUFFER")
    
    def window_update(self, base: int, next_seq: int, size: int):
        """Log window update."""
        self.debug(f"Window: base={base}, next={next_seq}, size={size}", "WINDOW")
    
    def progress(self, bytes_sent: int, total_bytes: int):
        """Log transfer progress."""
        pct = (bytes_sent / total_bytes * 100) if total_bytes > 0 else 0
        self.info(f"Progress: {bytes_sent}/{total_bytes} bytes ({pct:.1f}%)", "PROGRESS")
    
    def simulation_start(self, params: dict):
        """Log simulation start."""
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        self.info(f"Simulation started: {param_str}", "SIM")
    
    def simulation_end(self, metrics: dict):
        """Log simulation end."""
        self.info(f"Simulation ended: Goodput={metrics.get('goodput', 0):.2f} B/s", "SIM")
    
    def get_summary(self) -> dict:
        """Get logging summary."""
        return {
            'message_counts': dict(self.message_counts),
            'total_messages': sum(self.message_counts.values())
        }
    
    def close(self):
        """Close log file if open."""
        if self.file:
            self.file.close()
            self.file = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global logger instance
_global_logger: Optional[SimulationLogger] = None


def get_logger() -> SimulationLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = SimulationLogger()
    return _global_logger


def set_logger(logger: SimulationLogger):
    """Set global logger instance."""
    global _global_logger
    _global_logger = logger


if __name__ == "__main__":
    # Test logger
    print("=" * 60)
    print("SIMULATION LOGGER TEST")
    print("=" * 60)
    
    logger = SimulationLogger(name="Test", level=LogLevel.DEBUG)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test simulation events
    print("\n--- Simulation Events ---")
    logger.set_sim_time(0.0)
    logger.simulation_start({'window_size': 8, 'payload_size': 1024})
    
    logger.set_sim_time(0.001)
    logger.frame_sent(0, "DATA", 1048)
    
    logger.set_sim_time(0.050)
    logger.frame_received(0, "DATA", True)
    logger.ack_sent(0)
    
    logger.set_sim_time(0.100)
    logger.ack_received(0)
    
    logger.set_sim_time(0.600)
    logger.timeout(1, 1)
    logger.retransmit(1)
    
    logger.set_sim_time(1.0)
    logger.buffer_event("FULL", 0.95)
    
    logger.set_sim_time(10.0)
    logger.simulation_end({'goodput': 1234567.89})
    
    print(f"\nLogger summary: {logger.get_summary()}")
