"""
Application Layer Implementation

This module implements the application layer for file I/O,
handling the 100 MB data file reading and verification.
"""

import os
import hashlib
from typing import Optional, Callable, Generator, Tuple
from dataclasses import dataclass
import sys
sys.path.insert(0, '..')

from config import FILE_SIZE, INPUT_DIR


@dataclass
class TransferInfo:
    """Information about a file transfer."""
    filename: str
    size: int
    checksum: str
    chunk_size: int
    total_chunks: int


class ApplicationLayer:
    """
    Application Layer for file handling.
    
    Handles:
    - Reading large data files (100 MB)
    - Chunked data delivery to transport layer
    - Received data verification
    
    Attributes:
        chunk_size: Size of chunks to deliver to transport layer
    """
    
    def __init__(
        self,
        chunk_size: int = 8192,  # 8 KB chunks
        on_chunk_ready: Optional[Callable[[bytes, bool], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None
    ):
        """
        Initialize application layer.
        
        Args:
            chunk_size: Size of chunks of deliver to transport
            on_chunk_ready: Callback when chunk is ready
            on_progress: Callback for progress updates (bytes_read, total)
        """
        self.chunk_size = chunk_size
        self.on_chunk_ready = on_chunk_ready
        self.on_progress = on_progress
        
        # File state
        self.current_file: Optional[str] = None
        self.file_size = 0
        self.file_checksum = ""
        self.bytes_read = 0
        
        # Receive state
        self.received_data = bytearray()
        self.receive_complete = False
    
    def load_file(self, filepath: str) -> TransferInfo:
        """
        Load a file for transfer.
        
        Args:
            filepath: Path to the file
            
        Returns:
            TransferInfo with file details
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.current_file = filepath
        self.file_size = os.path.getsize(filepath)
        self.bytes_read = 0
        
        # Calculate checksum
        self.file_checksum = self._calculate_file_checksum(filepath)
        
        total_chunks = (self.file_size + self.chunk_size - 1) // self.chunk_size
        
        return TransferInfo(
            filename=os.path.basename(filepath),
            size=self.file_size,
            checksum=self.file_checksum,
            chunk_size=self.chunk_size,
            total_chunks=total_chunks
        )
    
    def _calculate_file_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of file."""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                md5.update(chunk)
        return md5.hexdigest()
    
    def read_chunks(self) -> Generator[bytes, None, None]:
        """
        Generator to read file in chunks.
        
        Yields:
            Data chunks
        """
        if self.current_file is None:
            return
        
        with open(self.current_file, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                self.bytes_read += len(chunk)
                
                if self.on_progress:
                    self.on_progress(self.bytes_read, self.file_size)
                
                yield chunk
    
    def get_all_data(self) -> bytes:
        """
        Read entire file at once.
        
        Returns:
            File contents
        """
        if self.current_file is None:
            return b''
        
        with open(self.current_file, 'rb') as f:
            return f.read()
    
    def receive_data(self, data: bytes):
        """
        Receive data from transport layer.
        
        Args:
            data: Received data bytes
        """
        self.received_data.extend(data)
    
    def finish_receive(self):
        """Mark receive as complete."""
        self.receive_complete = True
    
    def verify_received_data(self, expected_checksum: str) -> bool:
        """
        Verify received data integrity.
        
        Args:
            expected_checksum: Expected MD5 checksum
            
        Returns:
            True if data is valid
        """
        actual_checksum = hashlib.md5(self.received_data).hexdigest()
        return actual_checksum == expected_checksum
    
    def save_received_data(self, filepath: str):
        """
        Save received data to file.
        
        Args:
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(self.received_data)
    
    def get_received_data(self) -> bytes:
        """Get all received data."""
        return bytes(self.received_data)
    
    def get_received_size(self) -> int:
        """Get size of received data."""
        return len(self.received_data)
    
    def get_read_progress(self) -> float:
        """Get read progress (0-1)."""
        if self.file_size == 0:
            return 0.0
        return self.bytes_read / self.file_size
    
    def reset_receive(self):
        """Reset receive state."""
        self.received_data = bytearray()
        self.receive_complete = False
    
    def reset_send(self):
        """Reset send state."""
        self.current_file = None
        self.file_size = 0
        self.file_checksum = ""
        self.bytes_read = 0


class TestDataGenerator:
    """
    Generates test data for simulation.
    
    Used when no actual file is available.
    """
    
    @staticmethod
    def generate_test_data(size: int, pattern: str = "random") -> bytes:
        """
        Generate test data of specified size.
        
        Args:
            size: Size in bytes
            pattern: Pattern type ("random", "sequential", "zeros")
            
        Returns:
            Generated data
        """
        if pattern == "random":
            import random
            return bytes(random.getrandbits(8) for _ in range(size))
        elif pattern == "sequential":
            return bytes(i % 256 for i in range(size))
        elif pattern == "zeros":
            return bytes(size)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    @staticmethod
    def generate_test_file(filepath: str, size: int, pattern: str = "random"):
        """
        Generate a test file.
        
        Args:
            filepath: Output file path
            size: File size in bytes
            pattern: Data pattern
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        chunk_size = 1024 * 1024  # 1 MB
        
        with open(filepath, 'wb') as f:
            remaining = size
            while remaining > 0:
                chunk_len = min(chunk_size, remaining)
                chunk = TestDataGenerator.generate_test_data(chunk_len, pattern)
                f.write(chunk)
                remaining -= chunk_len
        
        print(f"Generated test file: {filepath} ({size} bytes)")
    
    @staticmethod
    def create_100mb_test_file(output_dir: str = INPUT_DIR) -> str:
        """
        Create the 100 MB test file required by the assignment.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to created file
        """
        filepath = os.path.join(output_dir, "test_100mb.bin")
        
        if os.path.exists(filepath):
            existing_size = os.path.getsize(filepath)
            if existing_size == FILE_SIZE:
                print(f"Test file already exists: {filepath}")
                return filepath
        
        TestDataGenerator.generate_test_file(filepath, FILE_SIZE, "sequential")
        return filepath


class DataVerifier:
    """
    Utility for verifying data integrity.
    """
    
    @staticmethod
    def calculate_checksum(data: bytes) -> str:
        """Calculate MD5 checksum of data."""
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def verify_data(original: bytes, received: bytes) -> Tuple[bool, dict]:
        """
        Verify received data against original.
        
        Args:
            original: Original data
            received: Received data
            
        Returns:
            Tuple of (match, details)
        """
        original_len = len(original)
        received_len = len(received)
        
        size_match = original_len == received_len
        
        if size_match:
            content_match = original == received
        else:
            content_match = False
        
        original_checksum = DataVerifier.calculate_checksum(original)
        received_checksum = DataVerifier.calculate_checksum(received)
        checksum_match = original_checksum == received_checksum
        
        # Find first mismatch if any
        first_mismatch = -1
        if not content_match:
            min_len = min(original_len, received_len)
            for i in range(min_len):
                if original[i] != received[i]:
                    first_mismatch = i
                    break
            if first_mismatch == -1 and original_len != received_len:
                first_mismatch = min_len
        
        details = {
            'size_match': size_match,
            'content_match': content_match,
            'checksum_match': checksum_match,
            'original_size': original_len,
            'received_size': received_len,
            'original_checksum': original_checksum,
            'received_checksum': received_checksum,
            'first_mismatch_byte': first_mismatch
        }
        
        return size_match and content_match, details




if __name__ == "__main__":
    # Test application layer
    print("=" * 60)
    print("APPLICATION LAYER TEST")
    print("=" * 60)
    
    # Generate small test data
    print("\nGenerating test data...")
    test_data = TestDataGenerator.generate_test_data(1024, "sequential")
    print(f"Generated {len(test_data)} bytes")
    
    # Test transfer simulation
    print("\nSimulating transfer...")
    
    # Sender side
    def on_progress(read, total):
        pct = read / total * 100
        print(f"  Progress: {read}/{total} ({pct:.1f}%)")
    
    # Create a temporary test file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(test_data)
        temp_path = f.name
    
    try:
        # Load file
        app = ApplicationLayer(chunk_size=256, on_progress=on_progress)
        info = app.load_file(temp_path)
        
        print(f"\nFile info:")
        print(f"  Size: {info.size} bytes")
        print(f"  Checksum: {info.checksum}")
        print(f"  Total chunks: {info.total_chunks}")
        
        # Simulate receiving
        for chunk in app.read_chunks():
            app.receive_data(chunk)  # Echo back
        
        app.finish_receive()
        
        # Verify
        received = app.get_received_data()
        valid, details = DataVerifier.verify_data(test_data, received)
        
        print(f"\nVerification:")
        print(f"  Valid: {valid}")
        for key, value in details.items():
            print(f"  {key}: {value}")
        
    finally:
        os.unlink(temp_path)
    
    print("\nTest complete!")
