"""
3D Surface Plot Visualization

This module generates 3D surface plots showing Goodput = f(W, L).
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
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GoodputSurfacePlot:
    """
    Generates 3D surface plots of Goodput(W, L).
    
    Creates interactive 3D visualizations with optimal point highlighted.
    """
    
    def __init__(
        self,
        results: Optional[List[Dict]] = None,
        csv_file: Optional[str] = None
    ):
        """
        Initialize surface plot generator.
        
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
    
    def _create_mesh_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple]:
        """
        Create mesh grid for 3D surface.
        
        Returns:
            Tuple of (X, Y, Z, optimal_point)
        """
        # Group results by (W, L)
        grouped = {}
        for r in self.results:
            key = (r['window_size'], r['payload_size'])
            if key not in grouped:
                grouped[key] = []
            if isinstance(r.get('goodput'), (int, float)):
                grouped[key].append(r['goodput'])
        
        # Create mesh grid
        W = np.array(self.window_sizes)
        L = np.array(self.payload_sizes)
        Wm, Lm = np.meshgrid(W, L, indexing='ij')
        
        # Fill Z values
        Z = np.zeros_like(Wm, dtype=float)
        max_val = 0
        max_point = (0, 0, 0)
        
        for i, w in enumerate(self.window_sizes):
            for j, l in enumerate(self.payload_sizes):
                key = (w, l)
                if key in grouped and grouped[key]:
                    mean_val = np.mean(grouped[key])
                    Z[i, j] = mean_val
                    if mean_val > max_val:
                        max_val = mean_val
                        max_point = (w, l, mean_val)
        
        return Wm, Lm, Z, max_point
    
    def plot(
        self,
        output_file: Optional[str] = None,
        title: str = "Goodput Surface: f(Window Size, Payload Size)",
        figsize: Tuple[int, int] = (12, 9),
        cmap: str = "viridis",
        elevation: float = 25,
        azimuth: float = 45,
        highlight_optimal: bool = True,
        unit: str = "MB/s"
    ) -> str:
        """
        Generate and save 3D surface plot.
        
        Args:
            output_file: Output file path
            title: Plot title
            figsize: Figure size
            cmap: Colormap name
            elevation: View elevation angle
            azimuth: View azimuth angle
            highlight_optimal: Highlight optimal point
            unit: Unit for display
            
        Returns:
            Path to saved figure
        """
        if not self.results:
            raise ValueError("No results to plot")
        
        W, L, Z, optimal = self._create_mesh_grid()
        
        # Convert units
        if unit == "MB/s":
            Z_display = Z / 1e6
            optimal = (optimal[0], optimal[1], optimal[2] / 1e6)
            z_label = "Goodput (MB/s)"
        else:
            Z_display = Z
            z_label = "Goodput (B/s)"
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(
            W, L, Z_display,
            cmap=cmap,
            edgecolor='none',
            alpha=0.8,
            antialiased=True
        )
        
        # Add wireframe for clarity
        ax.plot_wireframe(
            W, L, Z_display,
            color='black',
            alpha=0.1,
            linewidth=0.5
        )
        
        # Highlight optimal point
        if highlight_optimal:
            ax.scatter(
                [optimal[0]], [optimal[1]], [optimal[2]],
                color='red', s=200, marker='*',
                label=f'Optimal: W={optimal[0]}, L={optimal[1]}\n{optimal[2]:.2f} {unit}'
            )
            ax.legend(loc='upper left', fontsize=10)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(z_label)
        
        # Labels
        ax.set_xlabel('Window Size (W)', fontsize=11, labelpad=10)
        ax.set_ylabel('Payload Size (L, bytes)', fontsize=11, labelpad=10)
        ax.set_zlabel(z_label, fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # View angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Save figure
        if output_file is None:
            os.makedirs(PLOTS_DIR, exist_ok=True)
            output_file = os.path.join(PLOTS_DIR, 'goodput_surface.png')
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Surface plot saved to: {output_file}")
        return output_file
    
    def plot_multiple_views(
        self,
        output_file: Optional[str] = None,
        title: str = "Goodput Surface - Multiple Views"
    ) -> str:
        """
        Generate multiple view angles of the surface.
        
        Args:
            output_file: Output file path
            title: Overall title
            
        Returns:
            Path to saved figure
        """
        if not self.results:
            raise ValueError("No results to plot")
        
        W, L, Z, optimal = self._create_mesh_grid()
        Z_display = Z / 1e6  # MB/s
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        views = [
            (30, 45, "Standard View"),
            (30, 135, "Rotated 90°"),
            (60, 45, "Top-Down View"),
            (10, 45, "Side View")
        ]
        
        for idx, (elev, azim, view_title) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d')
            
            surf = ax.plot_surface(
                W, L, Z_display,
                cmap='viridis',
                edgecolor='none',
                alpha=0.8
            )
            
            ax.scatter(
                [optimal[0]], [optimal[1]], [optimal[2] / 1e6],
                color='red', s=100, marker='*'
            )
            
            ax.set_xlabel('W')
            ax.set_ylabel('L')
            ax.set_zlabel('Goodput')
            ax.set_title(view_title)
            ax.view_init(elev=elev, azim=azim)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_file is None:
            os.makedirs(PLOTS_DIR, exist_ok=True)
            output_file = os.path.join(PLOTS_DIR, 'goodput_surface_views.png')
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def plot_contour(
        self,
        output_file: Optional[str] = None,
        title: str = "Goodput Contour Plot"
    ) -> str:
        """
        Generate contour plot (top-down view of surface).
        
        Args:
            output_file: Output file path
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        W, L, Z, optimal = self._create_mesh_grid()
        Z_display = Z / 1e6  # MB/s
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Contour plot
        contour = ax.contourf(W, L, Z_display, levels=20, cmap='viridis')
        ax.contour(W, L, Z_display, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Goodput (MB/s)')
        
        # Optimal point
        ax.scatter(
            [optimal[0]], [optimal[1]],
            color='red', s=200, marker='*',
            label=f'Optimal: W={optimal[0]}, L={optimal[1]}'
        )
        ax.legend()
        
        ax.set_xlabel('Window Size (W)')
        ax.set_ylabel('Payload Size (L, bytes)')
        ax.set_title(title)
        
        if output_file is None:
            os.makedirs(PLOTS_DIR, exist_ok=True)
            output_file = os.path.join(PLOTS_DIR, 'goodput_contour.png')
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file


def generate_all_visualizations(
    results: List[Dict],
    output_dir: str = PLOTS_DIR
) -> Dict[str, str]:
    """
    Generate all visualization types.
    
    Args:
        results: Simulation results
        output_dir: Output directory
        
    Returns:
        Dictionary mapping plot type to file path
    """
    from visualization.heatmap import GoodputHeatmap
    
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}
    
    # Heatmap
    heatmap = GoodputHeatmap(results=results)
    outputs['heatmap'] = heatmap.plot(
        output_file=os.path.join(output_dir, 'goodput_heatmap.png')
    )
    
    # 3D Surface
    surface = GoodputSurfacePlot(results=results)
    outputs['surface'] = surface.plot(
        output_file=os.path.join(output_dir, 'goodput_surface.png')
    )
    
    # Multiple views
    outputs['surface_views'] = surface.plot_multiple_views(
        output_file=os.path.join(output_dir, 'goodput_surface_views.png')
    )
    
    # Contour
    outputs['contour'] = surface.plot_contour(
        output_file=os.path.join(output_dir, 'goodput_contour.png')
    )
    
    return outputs


if __name__ == "__main__":
    print("=" * 60)
    print("SURFACE PLOT GENERATOR TEST")
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
                # Create a pattern with a clear optimal region
                # Optimal around W=16, L=1024
                w_factor = -((w - 16) ** 2) / 100
                l_factor = -((l - 1024) ** 2) / 500000
                base = 800000 + w_factor * 10000 + l_factor * 1000
                noise = random.gauss(0, 20000)
                goodput = max(100000, base + noise)
                
                test_results.append({
                    'window_size': w,
                    'payload_size': l,
                    'run_id': run,
                    'goodput': goodput
                })
    
    print(f"Generated {len(test_results)} test results")
    
    # Create surface plot
    surface = GoodputSurfacePlot(results=test_results)
    
    # Single view
    output1 = surface.plot(title="Test 3D Surface Plot")
    print(f"Single view: {output1}")
    
    # Multiple views
    output2 = surface.plot_multiple_views()
    print(f"Multiple views: {output2}")
    
    # Contour
    output3 = surface.plot_contour()
    print(f"Contour: {output3}")
    
    print("\nTest complete!")
