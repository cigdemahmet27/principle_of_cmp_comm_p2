"""
Batch Runner for Parameter Sweep Simulations

This module implements the batch runner that executes all 360
simulations (6 window sizes × 6 payload sizes × 10 runs).
"""

import os
import csv
import time
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    WINDOW_SIZES, PAYLOAD_SIZES, RUNS_PER_CONFIGURATION,
    RNG_SEED_BASE, OUTPUT_DIR, RESULTS_CSV, FILE_SIZE
)
from simulation.simulator import Simulator, SimulatorConfig
from src.utils.logger import LogLevel

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class RunConfig:
    """Configuration for a single simulation run."""
    window_size: int
    payload_size: int
    run_id: int
    seed: int
    data_size: int


def run_single_simulation(run_config: RunConfig) -> Dict:
    """
    Run a single simulation with given configuration.
    
    This function is designed to be called in a separate process.
    
    Args:
        run_config: Configuration for this run
        
    Returns:
        Dictionary with results
    """
    try:
        config = SimulatorConfig(
            window_size=run_config.window_size,
            payload_size=run_config.payload_size,
            data_size=run_config.data_size,
            seed=run_config.seed,
            log_level=LogLevel.ERROR  # Minimal logging for batch runs
        )
        
        sim = Simulator(config)
        results = sim.run()
        
        # Extract key metrics
        metrics = results['metrics']
        
        return {
            'window_size': run_config.window_size,
            'payload_size': run_config.payload_size,
            'run_id': run_config.run_id,
            'seed': run_config.seed,
            'goodput': metrics['goodput'],
            'goodput_mbps': metrics['goodput_mbps'],
            'efficiency': metrics['efficiency'],
            'utilization': metrics['utilization'],
            'retransmissions': metrics['retransmissions'],
            'retransmission_rate': metrics['retransmission_rate'],
            'frame_error_rate': metrics['frame_error_rate'],
            'total_time': results['simulation_time'],
            'rtt_mean': metrics['rtt']['mean'] if metrics['rtt']['samples'] > 0 else 0,
            'rtt_min': metrics['rtt']['min'] if metrics['rtt']['samples'] > 0 else 0,
            'rtt_max': metrics['rtt']['max'] if metrics['rtt']['samples'] > 0 else 0,
            'buffer_full_events': metrics['buffer_full_events'],
            'backpressure_events': metrics['backpressure_events'],
            'data_valid': results['verification']['valid'],
            'complete': results['complete'],
            'error': None
        }
        
    except Exception as e:
        return {
            'window_size': run_config.window_size,
            'payload_size': run_config.payload_size,
            'run_id': run_config.run_id,
            'seed': run_config.seed,
            'goodput': 0,
            'error': str(e)
        }


class BatchRunner:
    """
    Batch Runner for parameter sweep simulations.
    
    Executes all (W, L) combinations with multiple runs each.
    
    Attributes:
        window_sizes: List of window sizes to test
        payload_sizes: List of payload sizes to test
        runs_per_config: Number of runs per configuration
        data_size: Size of data to transfer
    """
    
    def __init__(
        self,
        window_sizes: List[int] = None,
        payload_sizes: List[int] = None,
        runs_per_config: int = RUNS_PER_CONFIGURATION,
        data_size: int = FILE_SIZE,
        output_file: str = RESULTS_CSV,
        on_progress: Optional[Callable[[int, int, dict], None]] = None
    ):
        """
        Initialize batch runner.
        
        Args:
            window_sizes: List of window sizes (default from config)
            payload_sizes: List of payload sizes (default from config)
            runs_per_config: Number of runs per (W, L) pair
            data_size: Size of data to transfer
            output_file: Path to output CSV file
            on_progress: Callback for progress updates
        """
        self.window_sizes = window_sizes or WINDOW_SIZES
        self.payload_sizes = payload_sizes or PAYLOAD_SIZES
        self.runs_per_config = runs_per_config
        self.data_size = data_size
        self.output_file = output_file
        self.on_progress = on_progress
        
        # Results storage
        self.results: List[Dict] = []
        
        # Progress tracking
        self.total_runs = (len(self.window_sizes) * 
                         len(self.payload_sizes) * 
                         self.runs_per_config)
        self.completed_runs = 0
        self.start_time = 0.0
    
    def _generate_run_configs(self) -> List[RunConfig]:
        """Generate all run configurations."""
        configs = []
        
        for window_size in self.window_sizes:
            for payload_size in self.payload_sizes:
                for run_id in range(self.runs_per_config):
                    # Unique seed for each run
                    seed = (RNG_SEED_BASE + 
                           window_size * 1000 + 
                           payload_size + 
                           run_id * 10000)
                    
                    configs.append(RunConfig(
                        window_size=window_size,
                        payload_size=payload_size,
                        run_id=run_id,
                        seed=seed,
                        data_size=self.data_size
                    ))
        
        return configs
    
    def run_sequential(self) -> List[Dict]:
        """
        Run all simulations sequentially.
        
        Returns:
            List of result dictionaries
        """
        configs = self._generate_run_configs()
        self.results = []
        self.completed_runs = 0
        self.start_time = time.time()
        
        print(f"Running {self.total_runs} simulations sequentially...")
        
        iterator = tqdm(configs, desc="Simulations") if HAS_TQDM else configs
        
        for config in iterator:
            result = run_single_simulation(config)
            self.results.append(result)
            self.completed_runs += 1
            
            if self.on_progress:
                self.on_progress(self.completed_runs, self.total_runs, result)
            
            if not HAS_TQDM:
                elapsed = time.time() - self.start_time
                eta = (elapsed / self.completed_runs) * (self.total_runs - self.completed_runs)
                print(f"\rProgress: {self.completed_runs}/{self.total_runs} "
                      f"({self.completed_runs/self.total_runs*100:.1f}%) "
                      f"ETA: {eta:.0f}s", end="")
        
        if not HAS_TQDM:
            print()  # New line after progress
        
        total_time = time.time() - self.start_time
        print(f"Completed {self.total_runs} simulations in {total_time:.1f}s")
        
        return self.results
    
    def run_parallel(self, max_workers: Optional[int] = None) -> List[Dict]:
        """
        Run simulations in parallel using multiprocessing.
        
        Args:
            max_workers: Number of parallel workers (default: CPU count)
            
        Returns:
            List of result dictionaries
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        configs = self._generate_run_configs()
        self.results = []
        self.completed_runs = 0
        self.start_time = time.time()
        
        print(f"Running {self.total_runs} simulations with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_simulation, config): config 
                for config in configs
            }
            
            if HAS_TQDM:
                iterator = tqdm(as_completed(futures), total=len(futures), 
                               desc="Simulations")
            else:
                iterator = as_completed(futures)
            
            for future in iterator:
                result = future.result()
                self.results.append(result)
                self.completed_runs += 1
                
                if self.on_progress:
                    self.on_progress(self.completed_runs, self.total_runs, result)
                
                if not HAS_TQDM:
                    print(f"\rProgress: {self.completed_runs}/{self.total_runs} "
                          f"({self.completed_runs/self.total_runs*100:.1f}%)", end="")
        
        if not HAS_TQDM:
            print()
        
        total_time = time.time() - self.start_time
        print(f"Completed {self.total_runs} simulations in {total_time:.1f}s")
        
        return self.results
    
    def save_results(self, filepath: Optional[str] = None):
        """
        Save results to CSV file.
        
        Args:
            filepath: Output file path (default: self.output_file)
        """
        filepath = filepath or self.output_file
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not self.results:
            print("No results to save!")
            return
        
        # Get field names from first result
        fieldnames = list(self.results[0].keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Results saved to: {filepath}")
    
    def get_aggregated_results(self) -> Dict:
        """
        Get aggregated results by (W, L) pair.
        
        Returns:
            Dictionary with aggregated statistics
        """
        import statistics
        
        aggregated = {}
        
        for result in self.results:
            if result.get('error'):
                continue
            
            key = (result['window_size'], result['payload_size'])
            if key not in aggregated:
                aggregated[key] = {
                    'window_size': result['window_size'],
                    'payload_size': result['payload_size'],
                    'goodputs': [],
                    'retransmissions': [],
                    'rtt_means': [],
                    'efficiencies': []
                }
            
            aggregated[key]['goodputs'].append(result['goodput'])
            aggregated[key]['retransmissions'].append(result['retransmissions'])
            if result.get('rtt_mean', 0) > 0:
                aggregated[key]['rtt_means'].append(result['rtt_mean'])
            aggregated[key]['efficiencies'].append(result['efficiency'])
        
        # Calculate statistics
        for key, data in aggregated.items():
            goodputs = data['goodputs']
            if goodputs:
                data['goodput_mean'] = statistics.mean(goodputs)
                data['goodput_std'] = (statistics.stdev(goodputs) 
                                      if len(goodputs) > 1 else 0)
                data['goodput_min'] = min(goodputs)
                data['goodput_max'] = max(goodputs)
            
            if data['retransmissions']:
                data['retx_mean'] = statistics.mean(data['retransmissions'])
            
            if data['rtt_means']:
                data['rtt_mean_avg'] = statistics.mean(data['rtt_means'])
            
            if data['efficiencies']:
                data['efficiency_mean'] = statistics.mean(data['efficiencies'])
        
        return aggregated
    
    def get_optimal_configuration(self) -> Dict:
        """
        Find the optimal (W, L) configuration.
        
        Returns:
            Dictionary with optimal configuration info
        """
        aggregated = self.get_aggregated_results()
        
        if not aggregated:
            return {'error': 'No results available'}
        
        # Find max mean goodput
        best_key = max(aggregated.keys(), 
                      key=lambda k: aggregated[k].get('goodput_mean', 0))
        best_data = aggregated[best_key]
        
        return {
            'optimal_window_size': best_key[0],
            'optimal_payload_size': best_key[1],
            'mean_goodput': best_data.get('goodput_mean', 0),
            'goodput_std': best_data.get('goodput_std', 0),
            'mean_efficiency': best_data.get('efficiency_mean', 0),
            'mean_retransmissions': best_data.get('retx_mean', 0)
        }


if __name__ == "__main__":
    # Test batch runner with small parameter space
    print("=" * 60)
    print("BATCH RUNNER TEST")
    print("=" * 60)
    
    # Use small test configuration
    runner = BatchRunner(
        window_sizes=[4, 8],  # Just 2 values
        payload_sizes=[256, 512],  # Just 2 values
        runs_per_config=2,  # Just 2 runs
        data_size=1024 * 10,  # 10 KB for quick test
        output_file=os.path.join(OUTPUT_DIR, "test_results.csv")
    )
    
    print(f"\nTest configuration:")
    print(f"  Window sizes: {runner.window_sizes}")
    print(f"  Payload sizes: {runner.payload_sizes}")
    print(f"  Runs per config: {runner.runs_per_config}")
    print(f"  Total runs: {runner.total_runs}")
    print(f"  Data size: {runner.data_size} bytes")
    
    # Run sequentially for test
    print("\nRunning simulations...")
    results = runner.run_sequential()
    
    # Save results
    runner.save_results()
    
    # Show aggregated results
    print("\nAggregated results:")
    aggregated = runner.get_aggregated_results()
    for key, data in aggregated.items():
        print(f"  W={key[0]}, L={key[1]}: "
              f"Goodput={data.get('goodput_mean', 0):.2f} B/s, "
              f"Efficiency={data.get('efficiency_mean', 0)*100:.1f}%")
    
    # Find optimal
    optimal = runner.get_optimal_configuration()
    print(f"\nOptimal configuration:")
    print(f"  Window size: {optimal['optimal_window_size']}")
    print(f"  Payload size: {optimal['optimal_payload_size']}")
    print(f"  Mean Goodput: {optimal['mean_goodput']:.2f} B/s")
