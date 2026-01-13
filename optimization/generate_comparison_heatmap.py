"""
Generate comparison heatmaps: Your System vs AI-Optimized

This script reads baseline results from full_results.csv (your actual simulation)
and compares against AI-optimized results with lower RTO.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from simulation.simulator import Simulator, SimulatorConfig
from config import (
    BIT_RATE, FORWARD_PROPAGATION_DELAY, REVERSE_PROPAGATION_DELAY,
    PROCESSING_DELAY, PLOTS_DIR
)
import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available, cannot generate plots")


def calculate_optimal_timeout():
    """Calculate optimal RTO based on RTT estimate."""
    rtt = FORWARD_PROPAGATION_DELAY + REVERSE_PROPAGATION_DELAY + 2 * PROCESSING_DELAY
    return rtt + 4 * (rtt * 0.1)  # RTT + 4*variance


def load_baseline_from_csv(csv_path):
    """Load baseline results from existing CSV file."""
    print(f"Loading baseline results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    results = {}
    # Group by window_size and payload_size, calculate mean goodput
    grouped = df.groupby(['window_size', 'payload_size']).agg({
        'goodput': 'mean'
    }).reset_index()
    
    for _, row in grouped.iterrows():
        W = int(row['window_size'])
        L = int(row['payload_size'])
        goodput_mbps = row['goodput'] * 8 / 1e6  # Convert to Mbps
        results[(W, L)] = goodput_mbps
    
    return results


def run_optimized_sweep(window_sizes, payload_sizes, timeout, data_size, runs_per_config):
    """Run optimized parameter sweep."""
    print(f"\nRunning AI-OPTIMIZED sweep (RTO={timeout*1000:.0f}ms)...")
    
    results = {}
    total = len(window_sizes) * len(payload_sizes) * runs_per_config
    current = 0
    
    for W in window_sizes:
        for L in payload_sizes:
            goodputs = []
            for run in range(runs_per_config):
                current += 1
                seed = 42 + run * 100
                
                config = SimulatorConfig(
                    window_size=W,
                    payload_size=L,
                    timeout=timeout,
                    data_size=data_size,
                    seed=seed
                )
                
                sim = Simulator(config)
                result = sim.run()
                
                if result.get('metrics', {}).get('goodput_mbps'):
                    goodputs.append(result['metrics']['goodput_mbps'])
            
            avg_goodput = sum(goodputs) / len(goodputs) if goodputs else 0
            results[(W, L)] = avg_goodput
            print(f"  [{current}/{total}] W={W}, L={L}: {avg_goodput:.3f} Mbps")
    
    return results


def create_comparison_heatmaps(baseline_results, optimized_results, 
                                window_sizes, payload_sizes, output_dir):
    """Create side-by-side comparison heatmaps."""
    if not HAS_PLOTTING:
        return None
    
    # Create matrices
    n_w = len(window_sizes)
    n_l = len(payload_sizes)
    
    baseline_matrix = np.zeros((n_w, n_l))
    optimized_matrix = np.zeros((n_w, n_l))
    improvement_matrix = np.zeros((n_w, n_l))
    
    for i, W in enumerate(reversed(window_sizes)):  # Reversed so larger W at top
        for j, L in enumerate(payload_sizes):
            baseline_val = baseline_results.get((W, L), 0)
            optimized_val = optimized_results.get((W, L), 0)
            
            baseline_matrix[i, j] = baseline_val
            optimized_matrix[i, j] = optimized_val
            
            if baseline_val > 0:
                improvement_matrix[i, j] = ((optimized_val - baseline_val) / baseline_val) * 100
            else:
                improvement_matrix[i, j] = 0
    
    window_labels = list(reversed(window_sizes))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Baseline heatmap
    sns.heatmap(baseline_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=payload_sizes, yticklabels=window_labels,
                ax=axes[0], cbar_kws={'label': 'Goodput (Mbps)'})
    axes[0].set_xlabel('Payload Size (bytes)')
    axes[0].set_ylabel('Window Size')
    axes[0].set_title('YOUR SYSTEM\n(from run_full_sweep.py)', fontweight='bold', fontsize=12)
    
    # Optimized heatmap
    sns.heatmap(optimized_matrix, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=payload_sizes, yticklabels=window_labels,
                ax=axes[1], cbar_kws={'label': 'Goodput (Mbps)'})
    axes[1].set_xlabel('Payload Size (bytes)')
    axes[1].set_ylabel('Window Size')
    axes[1].set_title('AI-OPTIMIZED\n(RTO=76ms)', fontweight='bold', fontsize=12)
    
    # Improvement heatmap
    sns.heatmap(improvement_matrix, annot=True, fmt='+.0f', cmap='RdYlGn',
                xticklabels=payload_sizes, yticklabels=window_labels,
                ax=axes[2], cbar_kws={'label': 'Improvement (%)'}, center=0)
    axes[2].set_xlabel('Payload Size (bytes)')
    axes[2].set_ylabel('Window Size')
    axes[2].set_title('IMPROVEMENT\n(Optimized vs Your System)', fontweight='bold', fontsize=12)
    
    plt.suptitle('AI Optimization Comparison: Goodput (Mbps)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'optimization_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison heatmap saved to: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("AI OPTIMIZATION HEATMAP COMPARISON")
    print("=" * 70)
    
    # Parameters - SAME AS run_full_sweep.py
    window_sizes = [2, 4, 8, 16, 32, 64]
    payload_sizes = [128, 256, 512, 1024, 2048, 4096]
    data_size = 50 * 1024  # 50 KB (same as run_full_sweep.py)
    runs_per_config = 2
    
    # CSV file from run_full_sweep.py
    baseline_csv = 'data/output/full_results.csv'
    
    # Optimized timeout
    optimized_timeout = calculate_optimal_timeout()  # AI-optimized ~76ms
    
    print(f"\nConfiguration:")
    print(f"  Window Sizes: {window_sizes}")
    print(f"  Payload Sizes: {payload_sizes}")
    print(f"  Data Size: {data_size/1024:.0f} KB")
    print(f"  Baseline: Reading from {baseline_csv}")
    print(f"  Optimized RTO: {optimized_timeout*1000:.0f} ms (AI-calculated)")
    
    # Load baseline from CSV (your actual simulation results)
    baseline_results = load_baseline_from_csv(baseline_csv)
    
    print(f"\nLoaded {len(baseline_results)} baseline configurations")
    
    # Run optimized sweep
    optimized_results = run_optimized_sweep(
        window_sizes, payload_sizes,
        optimized_timeout, data_size, runs_per_config
    )
    
    # Print summary tables
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nYOUR SYSTEM GOODPUT (Mbps) - from run_full_sweep.py:")
    header = "W\\L".rjust(8)
    for L in payload_sizes:
        header += str(L).rjust(10)
    print(header)
    print("-" * (8 + 10 * len(payload_sizes)))
    for W in window_sizes:
        row = str(W).rjust(8)
        for L in payload_sizes:
            val = baseline_results.get((W, L), 0)
            row += f"{val:.3f}".rjust(10)
        print(row)
    
    print("\nOPTIMIZED GOODPUT (Mbps):")
    print(header)
    print("-" * (8 + 10 * len(payload_sizes)))
    for W in window_sizes:
        row = str(W).rjust(8)
        for L in payload_sizes:
            val = optimized_results.get((W, L), 0)
            row += f"{val:.3f}".rjust(10)
        print(row)
    
    print("\nIMPROVEMENT (%):")
    print(header)
    print("-" * (8 + 10 * len(payload_sizes)))
    for W in window_sizes:
        row = str(W).rjust(8)
        for L in payload_sizes:
            baseline = baseline_results.get((W, L), 0)
            optimized = optimized_results.get((W, L), 0)
            if baseline > 0:
                improvement = ((optimized - baseline) / baseline) * 100
            else:
                improvement = 0
            row += f"{improvement:+.0f}%".rjust(10)
        print(row)
    
    # Calculate overall improvement
    baseline_avg = sum(baseline_results.values()) / len(baseline_results)
    optimized_avg = sum(optimized_results.values()) / len(optimized_results)
    overall_improvement = ((optimized_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Average Your System Goodput:  {baseline_avg:.3f} Mbps")
    print(f"  Average Optimized Goodput:    {optimized_avg:.3f} Mbps")
    print(f"  Overall Improvement:          {overall_improvement:+.1f}%")
    
    # Generate comparison heatmap
    create_comparison_heatmaps(
        baseline_results, optimized_results,
        window_sizes, payload_sizes,
        PLOTS_DIR
    )


if __name__ == "__main__":
    main()
