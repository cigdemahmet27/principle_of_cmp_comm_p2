#!/usr/bin/env python3
"""
Run full parameter sweep and display results as a table.
"""

from simulation.runner import BatchRunner
import os

# Full parameter sweep with 2 runs per config for speed
runner = BatchRunner(
    window_sizes=[2, 4, 8, 16, 32, 64],
    payload_sizes=[128, 256, 512, 1024, 2048, 4096],
    runs_per_config=2,  # 2 runs per config for speed
    data_size=50 * 1024,  # 50 KB for faster test
    output_file='data/output/full_results.csv'
)

print(f'Total simulations: {runner.total_runs}')
print('Running simulations...')
results = runner.run_sequential()
runner.save_results()

# Print aggregated results as table
aggregated = runner.get_aggregated_results()
print()
print('=' * 80)
print('GOODPUT RESULTS (Mbps) - HEATMAP DATA')
print('=' * 80)

# Create table header
header_str = "W\\L".rjust(6)
for L in [128, 256, 512, 1024, 2048, 4096]:
    header_str += str(L).rjust(10)
print()
print(header_str)
print('-' * 66)

# Print rows
for W in [2, 4, 8, 16, 32, 64]:
    row_str = str(W).rjust(6)
    for L in [128, 256, 512, 1024, 2048, 4096]:
        key = (W, L)
        if key in aggregated:
            goodput_mbps = aggregated[key].get('goodput_mean', 0) * 8 / 1e6
            row_str += f'{goodput_mbps:.3f}'.rjust(10)
        else:
            row_str += 'N/A'.rjust(10)
    print(row_str)

print()
opt = runner.get_optimal_configuration()
print(f"OPTIMAL: W={opt['optimal_window_size']}, L={opt['optimal_payload_size']}, Goodput={opt['mean_goodput']*8/1e6:.3f} Mbps")
print()

# Also print efficiency table
print('=' * 80)
print('EFFICIENCY RESULTS (%) - How much useful data vs total transmitted')
print('=' * 80)

header_str = "W\\L".rjust(6)
for L in [128, 256, 512, 1024, 2048, 4096]:
    header_str += str(L).rjust(10)
print()
print(header_str)
print('-' * 66)

for W in [2, 4, 8, 16, 32, 64]:
    row_str = str(W).rjust(6)
    for L in [128, 256, 512, 1024, 2048, 4096]:
        key = (W, L)
        if key in aggregated:
            efficiency = aggregated[key].get('efficiency_mean', 0) * 100
            row_str += f'{efficiency:.1f}%'.rjust(10)
        else:
            row_str += 'N/A'.rjust(10)
    print(row_str)
