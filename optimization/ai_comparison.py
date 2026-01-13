"""
AI Optimization Comparison - Simplified Version

This script compares BASELINE vs OPTIMIZED SR-ARQ performance using the existing simulator.

The optimization is implemented by adjusting simulation parameters that the protocol can control:
- Window size selection
- Timeout values
- Payload size selection

OPTIMIZATIONS DEMONSTRATED:
==========================
1. OPTIMAL PARAMETER SELECTION - Use analysis to select best (W, L) combination
2. ADAPTIVE TIMEOUT - Use better initial timeout based on RTT estimate
3. LARGER WINDOW FOR HIGH BDP - Better pipeline utilization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulator import Simulator
from config import (
    BIT_RATE, FORWARD_PROPAGATION_DELAY, REVERSE_PROPAGATION_DELAY,
    PROCESSING_DELAY
)


def calculate_optimal_timeout():
    """
    AI OPTIMIZATION 1: Calculate optimal initial timeout based on RTT estimate.
    
    Instead of using arbitrary fixed timeout, calculate based on channel parameters.
    RTO = RTT_estimate + safety_margin
    
    This is what TCP's Jacobson/Karels algorithm would converge to.
    """
    # Estimate RTT from physical parameters
    rtt_estimate = (FORWARD_PROPAGATION_DELAY + REVERSE_PROPAGATION_DELAY + 
                    2 * PROCESSING_DELAY)
    
    # Add safety margin (K=4 like in TCP)
    rtt_variance_estimate = rtt_estimate * 0.1  # 10% variance estimate
    optimal_rto = rtt_estimate + 4 * rtt_variance_estimate
    
    return optimal_rto


def calculate_optimal_window(payload_size: int):
    """
    AI OPTIMIZATION 2: Calculate optimal window size based on Bandwidth-Delay Product.
    
    Window should be large enough to keep the pipe full:
    W >= BDP / frame_size
    
    Where BDP = Bit_Rate * RTT
    """
    rtt = FORWARD_PROPAGATION_DELAY + REVERSE_PROPAGATION_DELAY + 2 * PROCESSING_DELAY
    bdp_bytes = BIT_RATE * rtt / 8  # Bandwidth-Delay Product in bytes
    
    # Frame size includes header
    frame_size = payload_size + 24  # Link header
    
    # Optimal window to fill the pipe
    optimal_window = int(bdp_bytes / frame_size) + 1
    
    # Cap at reasonable maximum
    return min(optimal_window, 64)


def run_simulation(window_size: int, payload_size: int, timeout: float, 
                   data_size: int, seed: int):
    """Run a single simulation and return results."""
    from simulation.simulator import Simulator, SimulatorConfig
    
    config = SimulatorConfig(
        window_size=window_size,
        payload_size=payload_size,
        timeout=timeout,
        data_size=data_size,
        seed=seed
    )
    
    sim = Simulator(config)
    results = sim.run()
    return results


def run_comparison():
    """Run comparison between baseline and AI-optimized configurations."""
    print()
    print("*" * 70)
    print("*" + " AI-OPTIMIZED SR-ARQ COMPARISON ".center(68) + "*")
    print("*" * 70)
    print()
    
    print("=" * 70)
    print("AI OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Calculate optimal parameters
    optimal_timeout = calculate_optimal_timeout()
    print("OPTIMIZATION 1: Adaptive Timeout")
    print(f"  Estimated RTT: {(FORWARD_PROPAGATION_DELAY + REVERSE_PROPAGATION_DELAY + 2*PROCESSING_DELAY)*1000:.1f} ms")
    print(f"  Optimal RTO: {optimal_timeout*1000:.1f} ms (vs typical fixed 500ms)")
    print()
    
    # Test configurations
    data_size = 50 * 1024  # 50 KB
    runs = 2
    
    # Baseline: typical default parameters
    baseline_configs = [
        {'name': 'Small Window', 'W': 4, 'L': 512, 'timeout': 0.5},
        {'name': 'Medium Window', 'W': 16, 'L': 512, 'timeout': 0.5},
        {'name': 'Large Window', 'W': 32, 'L': 512, 'timeout': 0.5},
    ]
    
    # AI-Optimized: calculated optimal parameters
    optimal_L = 256  # From our heatmap analysis - best payload for this channel
    optimal_W = calculate_optimal_window(optimal_L)
    
    print("OPTIMIZATION 2: Optimal Window Size")
    print(f"  Bandwidth-Delay Product: {BIT_RATE * 0.054 / 8:.0f} bytes")
    print(f"  Optimal Window (L={optimal_L}): W={optimal_W}")
    print()
    
    print("OPTIMIZATION 3: Optimal Payload Selection")
    print(f"  From heatmap analysis: L={optimal_L} bytes provides best Goodput")
    print(f"  (Balances header overhead vs frame error probability)")
    print()
    
    optimized_configs = [
        {'name': 'AI-Optimized (W=opt, L=256)', 'W': optimal_W, 'L': 256, 'timeout': optimal_timeout},
        {'name': 'AI-Optimized (W=64, L=256)', 'W': 64, 'L': 256, 'timeout': optimal_timeout},
        {'name': 'AI-Optimized (W=32, L=512)', 'W': 32, 'L': 512, 'timeout': optimal_timeout},
    ]
    
    print("=" * 70)
    print("RUNNING SIMULATIONS")
    print("=" * 70)
    print()
    
    # Run baseline simulations
    print("BASELINE CONFIGURATIONS (Fixed parameters):")
    baseline_results = []
    for config in baseline_configs:
        goodputs = []
        for run in range(runs):
            seed = 42 + run * 100
            result = run_simulation(
                window_size=config['W'],
                payload_size=config['L'],
                timeout=config['timeout'],
                data_size=data_size,
                seed=seed
            )
            if result.get('metrics', {}).get('goodput_mbps'):
                goodputs.append(result['metrics']['goodput_mbps'])
        
        avg_goodput = sum(goodputs) / len(goodputs) if goodputs else 0
        baseline_results.append({'config': config, 'goodput': avg_goodput})
        print(f"  {config['name']:30s} W={config['W']:2d}, L={config['L']:4d}, RTO={config['timeout']*1000:.0f}ms -> {avg_goodput:.3f} Mbps")
    
    print()
    print("AI-OPTIMIZED CONFIGURATIONS (Calculated parameters):")
    optimized_results = []
    for config in optimized_configs:
        goodputs = []
        for run in range(runs):
            seed = 42 + run * 100
            result = run_simulation(
                window_size=config['W'],
                payload_size=config['L'],
                timeout=config['timeout'],
                data_size=data_size,
                seed=seed
            )
            if result.get('metrics', {}).get('goodput_mbps'):
                goodputs.append(result['metrics']['goodput_mbps'])
        
        avg_goodput = sum(goodputs) / len(goodputs) if goodputs else 0
        optimized_results.append({'config': config, 'goodput': avg_goodput})
        print(f"  {config['name']:30s} W={config['W']:2d}, L={config['L']:4d}, RTO={config['timeout']*1000:.0f}ms -> {avg_goodput:.3f} Mbps")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_baseline = max(baseline_results, key=lambda x: x['goodput'])
    best_optimized = max(optimized_results, key=lambda x: x['goodput'])
    
    print()
    print(f"  Best Baseline:  {best_baseline['config']['name']}")
    print(f"                  Goodput = {best_baseline['goodput']:.3f} Mbps")
    print()
    print(f"  Best Optimized: {best_optimized['config']['name']}")
    print(f"                  Goodput = {best_optimized['goodput']:.3f} Mbps")
    print()
    
    if best_baseline['goodput'] > 0:
        improvement = ((best_optimized['goodput'] - best_baseline['goodput']) / 
                      best_baseline['goodput']) * 100
        print(f"  IMPROVEMENT: {improvement:+.1f}%")
    
    print()
    print("=" * 70)
    print("AI OPTIMIZATIONS APPLIED:")
    print("=" * 70)
    print("""
1. ADAPTIVE TIMEOUT CALCULATION
   - Calculates optimal RTO based on channel RTT estimate
   - Uses formula: RTO = RTT + 4 × RTT_variance
   - Avoids wasteful fixed timeouts that are too long

2. BANDWIDTH-DELAY PRODUCT WINDOW SIZING  
   - Calculates window needed to keep channel fully utilized
   - W >= BDP / frame_size
   - Prevents pipeline stalls from undersized window

3. OPTIMAL PAYLOAD SELECTION
   - Analyzes tradeoff between header overhead and error probability
   - Smaller payloads have more header overhead
   - Larger payloads have higher corruption probability
   - Selects optimal balance from empirical analysis

These optimizations are protocol-level only - they do NOT modify:
   - Physical layer (bit rate, delays)
   - Channel model (BER, transition probabilities)
   - Fundamental ARQ mechanism
""")


if __name__ == "__main__":
    run_comparison()
