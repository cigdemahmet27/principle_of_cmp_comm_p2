#!/usr/bin/env python3
"""
Generate heatmap visualizations from the full sweep results.
"""

from visualization.heatmap import GoodputHeatmap

# Load results and generate heatmaps
print("Loading results from data/output/full_results.csv...")
heatmap = GoodputHeatmap(csv_file='data/output/full_results.csv')

# Generate Goodput heatmap
print("\nGenerating Goodput heatmap...")
output1 = heatmap.plot(
    output_file='data/output/plots/goodput_heatmap_360.png',
    title='Goodput vs Window Size and Payload Size (360 simulations)',
    show_values=True,
    highlight_optimal=False,
    unit='Mbps'
)
print(f"Saved: {output1}")

print("\nDone! Heatmap saved to data/output/plots/")
