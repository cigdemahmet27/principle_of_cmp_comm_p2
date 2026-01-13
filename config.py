"""
Configuration file for the Selective Repeat ARQ Protocol Simulator.
Contains all fixed baseline parameters as specified in the assignment.
"""

# =============================================================================
# PHYSICAL LAYER PARAMETERS
# =============================================================================

# Bit Rate (bits per second)
BIT_RATE = 10_000_000  # 10 Mbps

# Asymmetric Propagation Delays (in seconds)
FORWARD_PROPAGATION_DELAY = 0.040  # 40 ms - Data frames
REVERSE_PROPAGATION_DELAY = 0.010  # 10 ms - ACK frames
PROCESSING_DELAY = 0.002  # 2 ms per frame

# =============================================================================
# GILBERT-ELLIOT BURST ERROR MODEL PARAMETERS
# =============================================================================

# Bit Error Rates
GOOD_STATE_BER = 1e-6   # pg = 1 × 10^-6
BAD_STATE_BER = 5e-3    # pb = 5 × 10^-3

# State Transition Probabilities
P_GOOD_TO_BAD = 0.002   # P(G → B)
P_BAD_TO_GOOD = 0.05    # P(B → G)

# Average target BER ≈ 1 × 10^-4

# =============================================================================
# PROTOCOL HEADER SIZES (in bytes)
# =============================================================================

TRANSPORT_HEADER_SIZE = 8   # bytes
LINK_HEADER_SIZE = 24       # bytes

# =============================================================================
# BUFFER CONFIGURATION
# =============================================================================

# Application-side receiver buffer capacity
RECEIVER_BUFFER_SIZE = 256 * 1024  # 256 KB

# =============================================================================
# APPLICATION LAYER PARAMETERS
# =============================================================================

# Fixed input file size
FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# =============================================================================
# PARAMETER SWEEP CONFIGURATION
# =============================================================================

# Send Window Sizes to evaluate
WINDOW_SIZES = [2, 4, 8, 16, 32, 64]

# Frame Payload Sizes to evaluate (in bytes)
PAYLOAD_SIZES = [128, 256, 512, 1024, 2048, 4096]

# Number of simulation runs per (W, L) pair
RUNS_PER_CONFIGURATION = 10

# Total simulations = 6 × 6 × 10 = 360

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

# Default RNG seed base (actual seed = base + run_id)
RNG_SEED_BASE = 42

# Simulation time limit (seconds) - failsafe
MAX_SIMULATION_TIME = 3600  # 1 hour

# Logging verbosity levels
LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3

DEFAULT_LOG_LEVEL = LOG_LEVEL_INFO

# =============================================================================
# OUTPUT PATHS
# =============================================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
AI_LOGS_DIR = os.path.join(BASE_DIR, "ai_logs")

# Results CSV filename
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")
IMPROVED_RESULTS_CSV = os.path.join(OUTPUT_DIR, "improved_results.csv")

# =============================================================================
# DERIVED PARAMETERS (calculated from fixed parameters)
# =============================================================================

def calculate_transmission_time(frame_size_bytes):
    """Calculate transmission time for a frame of given size."""
    frame_size_bits = frame_size_bytes * 8
    return frame_size_bits / BIT_RATE

def calculate_total_frame_size(payload_size):
    """Calculate total frame size including all headers."""
    return payload_size + TRANSPORT_HEADER_SIZE + LINK_HEADER_SIZE

def calculate_rtt_estimate(payload_size):
    """
    Estimate round-trip time for a given payload size.
    RTT = Tx_data + Prop_forward + Processing + Tx_ack + Prop_reverse + Processing
    """
    data_frame_size = calculate_total_frame_size(payload_size)
    ack_frame_size = LINK_HEADER_SIZE  # ACK is just header, no payload
    
    tx_data = calculate_transmission_time(data_frame_size)
    tx_ack = calculate_transmission_time(ack_frame_size)
    
    rtt = (tx_data + FORWARD_PROPAGATION_DELAY + PROCESSING_DELAY +
           tx_ack + REVERSE_PROPAGATION_DELAY + PROCESSING_DELAY)
    
    return rtt

def calculate_steady_state_probabilities():
    """
    Calculate steady-state probabilities for Good and Bad states.
    π_G = P(B→G) / (P(G→B) + P(B→G))
    π_B = P(G→B) / (P(G→B) + P(B→G))
    """
    sum_transitions = P_GOOD_TO_BAD + P_BAD_TO_GOOD
    pi_good = P_BAD_TO_GOOD / sum_transitions
    pi_bad = P_GOOD_TO_BAD / sum_transitions
    return pi_good, pi_bad

def calculate_average_ber():
    """
    Calculate average BER based on steady-state probabilities.
    BER_avg = π_G * pg + π_B * pb
    """
    pi_good, pi_bad = calculate_steady_state_probabilities()
    return pi_good * GOOD_STATE_BER + pi_bad * BAD_STATE_BER


# Print configuration summary
if __name__ == "__main__":
    print("=" * 60)
    print("SELECTIVE REPEAT ARQ SIMULATOR - CONFIGURATION")
    print("=" * 60)
    print(f"\nPhysical Layer:")
    print(f"  Bit Rate: {BIT_RATE / 1e6:.0f} Mbps")
    print(f"  Forward Delay: {FORWARD_PROPAGATION_DELAY * 1000:.0f} ms")
    print(f"  Reverse Delay: {REVERSE_PROPAGATION_DELAY * 1000:.0f} ms")
    print(f"  Processing Delay: {PROCESSING_DELAY * 1000:.0f} ms")
    
    print(f"\nGilbert-Elliot Model:")
    print(f"  Good State BER: {GOOD_STATE_BER:.2e}")
    print(f"  Bad State BER: {BAD_STATE_BER:.2e}")
    print(f"  P(G->B): {P_GOOD_TO_BAD}")
    print(f"  P(B->G): {P_BAD_TO_GOOD}")
    
    pi_good, pi_bad = calculate_steady_state_probabilities()
    avg_ber = calculate_average_ber()
    print(f"  Steady-state P(Good): {pi_good:.4f}")
    print(f"  Steady-state P(Bad): {pi_bad:.4f}")
    print(f"  Average BER: {avg_ber:.2e}")
    
    print(f"\nProtocol Headers:")
    print(f"  Transport Header: {TRANSPORT_HEADER_SIZE} bytes")
    print(f"  Link Header: {LINK_HEADER_SIZE} bytes")
    
    print(f"\nBuffer & File:")
    print(f"  Receiver Buffer: {RECEIVER_BUFFER_SIZE / 1024:.0f} KB")
    print(f"  Input File Size: {FILE_SIZE / (1024*1024):.0f} MB")
    
    print(f"\nParameter Sweep:")
    print(f"  Window Sizes: {WINDOW_SIZES}")
    print(f"  Payload Sizes: {PAYLOAD_SIZES}")
    print(f"  Runs per config: {RUNS_PER_CONFIGURATION}")
    print(f"  Total simulations: {len(WINDOW_SIZES) * len(PAYLOAD_SIZES) * RUNS_PER_CONFIGURATION}")
    
    print(f"\nSample RTT Estimates:")
    for payload in PAYLOAD_SIZES:
        rtt = calculate_rtt_estimate(payload)
        print(f"  Payload {payload:4d} bytes: RTT = {rtt*1000:.2f} ms")
