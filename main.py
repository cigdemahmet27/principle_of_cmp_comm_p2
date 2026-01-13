#!/usr/bin/env python3
"""
Selective Repeat ARQ Protocol Simulator - Main Entry Point

This is the main CLI interface for the ARQ protocol simulator.
It provides options for:
- Single simulation runs
- Full parameter sweep (360 simulations)
- Visualization generation
- Optimized protocol comparison

Usage:
    python main.py --single --window 16 --payload 512
    python main.py --sweep --runs 10
    python main.py --visualize --csv results.csv
"""

import argparse
import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WINDOW_SIZES, PAYLOAD_SIZES, RUNS_PER_CONFIGURATION,
    FILE_SIZE, OUTPUT_DIR, RESULTS_CSV, PLOTS_DIR
)


def run_single_simulation(args):
    """Run a single simulation with specified parameters."""
    from simulation.simulator import Simulator, SimulatorConfig
    from src.utils.logger import LogLevel
    
    config = SimulatorConfig(
        window_size=args.window,
        payload_size=args.payload,
        data_size=args.data_size,
        seed=args.seed,
        adaptive_timeout=args.adaptive_timeout,
        log_level=LogLevel.INFO if args.verbose else LogLevel.WARNING
    )
    
    print("=" * 60)
    print("SELECTIVE REPEAT ARQ SIMULATOR")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Window size: {config.window_size}")
    print(f"  Payload size: {config.payload_size} bytes")
    print(f"  Data size: {config.data_size / 1024:.1f} KB")
    print(f"  Timeout: {config.get_timeout() * 1000:.2f} ms")
    print(f"  Adaptive timeout: {config.adaptive_timeout}")
    print(f"  Seed: {config.seed}")
    
    print("\nRunning simulation...")
    
    sim = Simulator(config)
    start_time = time.time()
    results = sim.run()
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nTransfer Status:")
    print(f"  Complete: {results['complete']}")
    print(f"  Data Valid: {results['verification']['valid']}")
    print(f"  Simulation Time: {results['simulation_time']:.4f} s")
    print(f"  Real Time: {elapsed:.2f} s")
    
    metrics = results['metrics']
    print(f"\nPerformance Metrics:")
    print(f"  Goodput: {metrics['goodput']:.2f} B/s ({metrics['goodput_mbps']:.4f} Mbps)")
    print(f"  Efficiency: {metrics['efficiency'] * 100:.2f}%")
    print(f"  Utilization: {metrics['utilization'] * 100:.2f}%")
    
    print(f"\nFrame Statistics:")
    print(f"  Frames Sent: {metrics['data_frames_sent']}")
    print(f"  Frames Delivered: {metrics['data_frames_delivered']}")
    print(f"  Retransmissions: {metrics['retransmissions']}")
    print(f"  Frame Error Rate: {metrics['frame_error_rate']:.4f}")
    
    if metrics['rtt']['samples'] > 0:
        print(f"\nRTT Statistics:")
        print(f"  Mean: {metrics['rtt']['mean'] * 1000:.2f} ms")
        print(f"  Min: {metrics['rtt']['min'] * 1000:.2f} ms")
        print(f"  Max: {metrics['rtt']['max'] * 1000:.2f} ms")
    
    return results


def run_parameter_sweep(args):
    """Run full parameter sweep."""
    from simulation.runner import BatchRunner
    
    print("=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)
    
    # Determine parameter space
    if args.quick:
        window_sizes = [8, 16, 32]
        payload_sizes = [256, 512, 1024]
        runs = 3
        data_size = 100 * 1024  # 100 KB for quick test
    else:
        window_sizes = WINDOW_SIZES
        payload_sizes = PAYLOAD_SIZES
        runs = args.runs
        data_size = args.data_size
    
    runner = BatchRunner(
        window_sizes=window_sizes,
        payload_sizes=payload_sizes,
        runs_per_config=runs,
        data_size=data_size,
        output_file=args.output or RESULTS_CSV
    )
    
    print(f"\nConfiguration:")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Payload sizes: {payload_sizes}")
    print(f"  Runs per config: {runs}")
    print(f"  Total simulations: {runner.total_runs}")
    print(f"  Data size per run: {data_size / 1024:.1f} KB")
    print(f"  Output: {args.output or RESULTS_CSV}")
    
    print("\nStarting parameter sweep...")
    
    if args.parallel:
        results = runner.run_parallel(max_workers=args.workers)
    else:
        results = runner.run_sequential()
    
    # Save results
    runner.save_results()
    
    # Show optimal configuration
    optimal = runner.get_optimal_configuration()
    
    print("\n" + "=" * 60)
    print("OPTIMAL CONFIGURATION")
    print("=" * 60)
    print(f"  Window Size: {optimal['optimal_window_size']}")
    print(f"  Payload Size: {optimal['optimal_payload_size']} bytes")
    print(f"  Mean Goodput: {optimal['mean_goodput']:.2f} B/s")
    print(f"  Mean Efficiency: {optimal.get('mean_efficiency', 0) * 100:.2f}%")
    
    return results


def generate_visualizations(args):
    """Generate visualization plots."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    csv_file = args.csv or RESULTS_CSV
    
    if not os.path.exists(csv_file):
        print(f"Error: Results file not found: {csv_file}")
        print("Run a parameter sweep first: python main.py --sweep")
        return
    
    # Load results
    import csv
    results = []
    with open(csv_file, 'r') as f:
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
    
    print(f"Loaded {len(results)} results from {csv_file}")
    
    # Ensure output directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate heatmap
    print("\nGenerating heatmap...")
    from visualization.heatmap import GoodputHeatmap
    heatmap = GoodputHeatmap(results=results)
    heatmap_file = heatmap.plot(
        output_file=os.path.join(PLOTS_DIR, 'goodput_heatmap.png'),
        title="Goodput vs Window Size and Payload Size"
    )
    
    # Generate 3D surface plot
    print("Generating 3D surface plot...")
    from visualization.surface_plot import GoodputSurfacePlot
    surface = GoodputSurfacePlot(results=results)
    surface_file = surface.plot(
        output_file=os.path.join(PLOTS_DIR, 'goodput_surface.png'),
        title="Goodput Surface: f(Window Size, Payload Size)"
    )
    
    # Generate contour plot
    print("Generating contour plot...")
    contour_file = surface.plot_contour(
        output_file=os.path.join(PLOTS_DIR, 'goodput_contour.png')
    )
    
    # Generate multiple views
    print("Generating multi-view surface plot...")
    views_file = surface.plot_multiple_views(
        output_file=os.path.join(PLOTS_DIR, 'goodput_surface_views.png')
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATIONS GENERATED")
    print("=" * 60)
    print(f"  Heatmap: {heatmap_file}")
    print(f"  Surface: {surface_file}")
    print(f"  Contour: {contour_file}")
    print(f"  Multi-view: {views_file}")


def show_config(args):
    """Display current configuration."""
    print("=" * 60)
    print("SIMULATOR CONFIGURATION")
    print("=" * 60)
    
    # Run config module
    import config as cfg
    
    print(f"\nPhysical Layer Parameters:")
    print(f"  Bit Rate: {cfg.BIT_RATE / 1e6:.0f} Mbps")
    print(f"  Forward Delay: {cfg.FORWARD_PROPAGATION_DELAY * 1000:.0f} ms")
    print(f"  Reverse Delay: {cfg.REVERSE_PROPAGATION_DELAY * 1000:.0f} ms")
    print(f"  Processing Delay: {cfg.PROCESSING_DELAY * 1000:.0f} ms")
    
    print(f"\nGilbert-Elliot Channel:")
    print(f"  Good State BER: {cfg.GOOD_STATE_BER:.2e}")
    print(f"  Bad State BER: {cfg.BAD_STATE_BER:.2e}")
    print(f"  P(Good→Bad): {cfg.P_GOOD_TO_BAD}")
    print(f"  P(Bad→Good): {cfg.P_BAD_TO_GOOD}")
    
    avg_ber = cfg.calculate_average_ber()
    print(f"  Average BER: {avg_ber:.2e}")
    
    print(f"\nProtocol Headers:")
    print(f"  Transport Header: {cfg.TRANSPORT_HEADER_SIZE} bytes")
    print(f"  Link Header: {cfg.LINK_HEADER_SIZE} bytes")
    
    print(f"\nParameter Sweep:")
    print(f"  Window Sizes: {cfg.WINDOW_SIZES}")
    print(f"  Payload Sizes: {cfg.PAYLOAD_SIZES}")
    print(f"  Runs per config: {cfg.RUNS_PER_CONFIGURATION}")
    print(f"  Total simulations: {len(cfg.WINDOW_SIZES) * len(cfg.PAYLOAD_SIZES) * cfg.RUNS_PER_CONFIGURATION}")
    
    print(f"\nTheoretical RTT for each payload size:")
    for payload in cfg.PAYLOAD_SIZES:
        rtt = cfg.calculate_rtt_estimate(payload)
        print(f"  {payload} bytes: {rtt * 1000:.2f} ms")


def generate_test_file(args):
    """Generate test data file."""
    from src.layers.application_layer import TestDataGenerator
    
    filepath = TestDataGenerator.create_100mb_test_file()
    print(f"Test file created: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Selective Repeat ARQ Protocol Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single simulation:
    python main.py --single --window 16 --payload 512

  Quick parameter sweep (for testing):
    python main.py --sweep --quick

  Full parameter sweep (360 runs):
    python main.py --sweep --runs 10

  Parallel parameter sweep:
    python main.py --sweep --parallel --workers 4

  Generate visualizations:
    python main.py --visualize

  Show configuration:
    python main.py --config
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--single', action='store_true',
                     help='Run single simulation')
    mode.add_argument('--sweep', action='store_true',
                     help='Run parameter sweep')
    mode.add_argument('--visualize', action='store_true',
                     help='Generate visualizations')
    mode.add_argument('--config', action='store_true',
                     help='Show configuration')
    mode.add_argument('--generate-test-file', action='store_true',
                     help='Generate 100MB test file')
    
    # Single simulation options
    parser.add_argument('--window', '-w', type=int, default=8,
                       help='Window size (default: 8)')
    parser.add_argument('--payload', '-p', type=int, default=1024,
                       help='Payload size in bytes (default: 1024)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--adaptive-timeout', action='store_true',
                       help='Enable adaptive timeout')
    
    # Parameter sweep options
    parser.add_argument('--runs', '-r', type=int, 
                       default=RUNS_PER_CONFIGURATION,
                       help=f'Runs per configuration (default: {RUNS_PER_CONFIGURATION})')
    parser.add_argument('--parallel', action='store_true',
                       help='Run simulations in parallel')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced parameters')
    
    # Data options
    parser.add_argument('--data-size', type=int, default=1024*1024,
                       help='Data size in bytes (default: 1MB)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path')
    parser.add_argument('--csv', type=str,
                       help='CSV file for visualization')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Execute selected mode
    if args.single:
        run_single_simulation(args)
    elif args.sweep:
        run_parameter_sweep(args)
    elif args.visualize:
        generate_visualizations(args)
    elif args.config:
        show_config(args)
    elif args.generate_test_file:
        generate_test_file(args)


if __name__ == "__main__":
    main()
