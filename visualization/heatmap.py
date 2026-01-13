"""
Goodput Heatmap Visualization

This module generates 2D heatmaps showing Goodput = f(W, L).
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import csv

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WINDOW_SIZES, PAYLOAD_SIZES, PLOTS_DIR

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class GoodputHeatmap:
    """
    Generates 2D heatmaps of Goodput(W, L).
    
    Creates publication-quality heatmaps with optimal point highlighted.
    """
    
    def __init__(
        self,
        results: Optional[List[Dict]] = None,
        csv_file: Optional[str] = None
    ):
        """
        Initialize heatmap generator.
        
        Args:
            results: List of result dictionaries
            csv_file: Path to CSV file with results
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        
        if results:
            self.results = results
        elif csv_file:
            self.results = self._load_csv(csv_file)
        else:
            self.results = []
        
        self.window_sizes = sorted(set(r['window_size'] for r in self.results)) if self.results else WINDOW_SIZES
        self.payload_sizes = sorted(set(r['payload_size'] for r in self.results)) if self.results else PAYLOAD_SIZES
    
    def _load_csv(self, filepath: str) -> List[Dict]:
        """Load results from CSV file."""
        results = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    try:
                        if '.' in str(row[key]):
                            row[key] = float(row[key])
                        else:
                            row[key] = int(row[key])
                    except (ValueError, TypeError):
                        pass
                results.append(row)
        return results
    
    def _create_goodput_matrix(self) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Create matrix of mean goodput values.
        
        Returns:
            Tuple of (matrix, max_value, optimal_indices)
        """
        # Group results by (W, L)
        grouped = {}
        for r in self.results:
            key = (r['window_size'], r['payload_size'])
            if key not in grouped:
                grouped[key] = []
            if isinstance(r.get('goodput'), (int, float)):
                grouped[key].append(r['goodput'])
        
        # Create matrix
        n_windows = len(self.window_sizes)
        n_payloads = len(self.payload_sizes)
        matrix = np.zeros((n_windows, n_payloads))
        
        max_val = 0
        max_idx = (0, 0)
        
        for i, w in enumerate(self.window_sizes):
            for j, l in enumerate(self.payload_sizes):
                key = (w, l)
                if key in grouped and grouped[key]:
                    mean_val = np.mean(grouped[key])
                    matrix[i, j] = mean_val
                    if mean_val > max_val:
                        max_val = mean_val
                        max_idx = (i, j)
        
        return matrix, max_val, max_idx
    
    def plot(
        self,
        output_file: Optional[str] = None,
        title: str = "Goodput vs Window Size and Payload Size",
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "viridis",
        show_values: bool = True,
        highlight_optimal: bool = False,
        unit: str = "Mbps"
    ) -> str:
        """
        Generate and save heatmap.
        
        Args:
            output_file: Output file path (auto-generated if None)
            title: Plot title
            figsize: Figure size (width, height)
            cmap: Colormap name
            show_values: Show values in cells
            highlight_optimal: Highlight optimal point
            unit: Unit for display
            
        Returns:
            Path to saved figure
        """
        if not self.results:
            raise ValueError("No results to plot")
        
        matrix, max_val, max_idx = self._create_goodput_matrix()
        
        # Convert to Mbps for display
        if unit == "Mbps":
            matrix_display = matrix * 8 / 1e6  # bytes/s to Mbps
            unit_label = "Goodput (Mbps)"
        elif unit == "MB/s":
            matrix_display = matrix / 1e6
            unit_label = "Goodput (MB/s)"
        else:
            matrix_display = matrix
            unit_label = "Goodput (B/s)"
        
        # Reverse the matrix and window_sizes so larger W is at top
        matrix_display = np.flipud(matrix_display)
        window_sizes_display = list(reversed(self.window_sizes))
        # Adjust max_idx for flipped matrix
        max_idx = (len(self.window_sizes) - 1 - max_idx[0], max_idx[1])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use seaborn if available
        if HAS_SEABORN:
            sns.heatmap(
                matrix_display,
                annot=show_values,
                fmt='.3f',
                cmap=cmap,
                xticklabels=self.payload_sizes,
                yticklabels=window_sizes_display,
                ax=ax,
                cbar_kws={'label': unit_label}
            )
        else:
            # Matplotlib fallback
            im = ax.imshow(matrix_display, cmap=cmap, aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(unit_label)
            
            # Add tick labels
            ax.set_xticks(range(len(self.payload_sizes)))
            ax.set_xticklabels(self.payload_sizes)
            ax.set_yticks(range(len(self.window_sizes)))
            ax.set_yticklabels(window_sizes_display)
            
            # Add values
            if show_values:
                for i in range(len(self.window_sizes)):
                    for j in range(len(self.payload_sizes)):
                        value = matrix_display[i, j]
                        color = 'white' if value < matrix_display.max() / 2 else 'black'
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                               color=color, fontsize=8)
        
        # Highlight optimal
        if highlight_optimal:
            opt_i, opt_j = max_idx
            rect = plt.Rectangle(
                (opt_j - 0.5, opt_i - 0.5), 1, 1,
                fill=False, edgecolor='red', linewidth=3
            )
            ax.add_patch(rect)
            
            # Add annotation
            opt_w = window_sizes_display[opt_i]
            opt_l = self.payload_sizes[opt_j]
            opt_val = matrix_display[opt_i, opt_j]
            ax.annotate(
                f'Optimal\nW={opt_w}, L={opt_l}\n{opt_val:.2f} {unit}',
                xy=(opt_j, opt_i),
                xytext=(opt_j + 1.5, opt_i - 1),
                fontsize=10,
                color='red',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Labels
        ax.set_xlabel('Payload Size (bytes)', fontsize=12)
        ax.set_ylabel('Window Size', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        if output_file is None:
            os.makedirs(PLOTS_DIR, exist_ok=True)
            output_file = os.path.join(PLOTS_DIR, 'goodput_heatmap.png')
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to: {output_file}")
        return output_file
    
    def plot_comparison(
        self,
        other_results: List[Dict],
        output_file: Optional[str] = None,
        titles: Tuple[str, str] = ("Before Optimization", "After Optimization")
    ) -> str:
        """
        Generate side-by-side comparison heatmaps.
        
        Args:
            other_results: Results to compare with
            output_file: Output file path
            titles: Titles for each subplot
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (results, title) in enumerate([(self.results, titles[0]), 
                                                 (other_results, titles[1])]):
            # Create matrix for these results
            grouped = {}
            for r in results:
                key = (r['window_size'], r['payload_size'])
                if key not in grouped:
                    grouped[key] = []
                if isinstance(r.get('goodput'), (int, float)):
                    grouped[key].append(r['goodput'])
            
            matrix = np.zeros((len(self.window_sizes), len(self.payload_sizes)))
            for i, w in enumerate(self.window_sizes):
                for j, l in enumerate(self.payload_sizes):
                    key = (w, l)
                    if key in grouped and grouped[key]:
                        matrix[i, j] = np.mean(grouped[key]) / 1e6  # MB/s
            
            ax = axes[idx]
            if HAS_SEABORN:
                sns.heatmap(
                    matrix, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=self.payload_sizes,
                    yticklabels=self.window_sizes,
                    ax=ax
                )
            else:
                im = ax.imshow(matrix, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax)
            
            ax.set_xlabel('Payload Size (bytes)')
            ax.set_ylabel('Window Size')
            ax.set_title(title)
        
        plt.suptitle("Goodput Comparison (MB/s)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_file is None:
            os.makedirs(PLOTS_DIR, exist_ok=True)
            output_file = os.path.join(PLOTS_DIR, 'goodput_comparison.png')
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("HEATMAP GENERATOR TEST")
    print("=" * 60)
    
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, cannot run test")
        exit(1)
    
    # Generate fake test data
    import random
    
    test_results = []
    for w in WINDOW_SIZES:
        for l in PAYLOAD_SIZES:
            for run in range(3):
                # Simulate some goodput pattern
                base = 500000 + w * 10000 + l * 100
                noise = random.gauss(0, 50000)
                goodput = max(0, base + noise)
                
                test_results.append({
                    'window_size': w,
                    'payload_size': l,
                    'run_id': run,
                    'goodput': goodput
                })
    
    print(f"Generated {len(test_results)} test results")
    
    # Create heatmap
    heatmap = GoodputHeatmap(results=test_results)
    output = heatmap.plot(
        title="Test Goodput Heatmap",
        show_values=True,
        highlight_optimal=True
    )
    
    print(f"Test complete: {output}")
