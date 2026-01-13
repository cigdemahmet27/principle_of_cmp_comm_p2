# Selective Repeat ARQ Protocol Simulator

A cross-layer network simulator implementing Selective Repeat ARQ protocol over a Gilbert-Elliot burst error channel.

## Overview

This project implements a complete network simulation stack for evaluating the performance of Selective Repeat ARQ under realistic burst error conditions. The simulator supports:

- **Cross-layer integration**: Application, Transport, Link, and Physical layers
- **Gilbert-Elliot channel model**: Realistic burst error simulation
- **Selective Repeat ARQ**: Full implementation with per-frame timers
- **Parameter sweep**: Automated evaluation of 360 configurations
- **Visualization**: Heatmaps and 3D surface plots

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify configuration
python main.py --config
```

---

## Commands Reference

### Command 1: Show Configuration

```bash
python main.py --config
```

**What it does:** Displays all fixed simulation parameters without running any simulation.

**Example output:**
```
============================================================
SIMULATION CONFIGURATION
============================================================

Physical Layer:
  Bit Rate: 10.0 Mbps
  Forward Propagation Delay: 40.0 ms
  Reverse Propagation Delay: 10.0 ms
  Processing Delay: 2.0 ms

Gilbert-Elliot Channel:
  Good State BER: 1e-06
  Bad State BER: 0.005
  P(Good → Bad): 0.002
  P(Bad → Good): 0.05

Parameter Sweep:
  Window Sizes: [2, 4, 8, 16, 32, 64]
  Payload Sizes: [128, 256, 512, 1024, 2048, 4096]
  Runs per config: 10
  Total configurations: 360
```

---

### Command 2: Single Simulation

```bash
python main.py --single --window <W> --payload <L>
```

**What it does:** Runs ONE simulation with the specified window size (W) and payload size (L).

**Arguments:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--window` | Sender window size (how many frames in flight) | `--window 16` |
| `--payload` | Frame payload size in bytes | `--payload 512` |
| `--verbose` | Show detailed progress during simulation | `--verbose` |
| `--data-size` | Amount of data to transfer (default: 100 KB) | `--data-size 1048576` (1 MB) |

**Example:**
```bash
python main.py --single --window 16 --payload 512 --verbose
```

**Example output:**
```
============================================================
SINGLE SIMULATION
============================================================

Configuration:
  Window Size: 16
  Payload Size: 512 bytes
  Data Size: 100.0 KB

Running simulation...

============================================================
RESULTS
============================================================

Transfer:
  Complete: True
  Data Valid: True
  Simulation Time: 2.34 s

Performance Metrics:
  Goodput: 42735.89 B/s (0.34 Mbps)    ← MAIN METRIC: Useful data delivered per second
  Efficiency: 38.5%                     ← Useful bytes ÷ Total bytes transmitted
  Utilization: 8.2%                     ← Channel usage
  Retransmissions: 245                  ← Number of frames re-sent

RTT Statistics:
  Mean: 54.4 ms
  Min: 54.2 ms
  Max: 54.5 ms
```

**Output explanation:**
| Metric | What it means |
|--------|---------------|
| **Goodput** | Most important! Measures how fast useful data was delivered (excludes headers & retransmissions) |
| **Efficiency** | What fraction of transmitted data was useful (higher = less overhead/retransmissions) |
| **Utilization** | How much of the channel capacity was actually used |
| **Retransmissions** | How many frames had to be re-sent due to corruption |
| **RTT** | Round-trip time from sending a frame to receiving its ACK |

---

### Command 3: Quick Parameter Sweep

```bash
python main.py --sweep --quick
```

**What it does:** Runs a SMALL parameter sweep for testing (3×3×3 = 27 simulations instead of 360).

**Uses reduced parameter space:**
- Window Sizes: [8, 16, 32]
- Payload Sizes: [256, 512, 1024]
- Runs per config: 3

**Example output:**
```
============================================================
PARAMETER SWEEP
============================================================

Configuration:
  Window sizes: [8, 16, 32]
  Payload sizes: [256, 512, 1024]
  Runs per config: 3
  Total simulations: 27
  Data size per run: 100.0 KB
  Output: data/output/results.csv

Starting parameter sweep...
Running 27 simulations sequentially...
Simulations: 100%|██████████| 27/27 [00:37<00:00,  1.38s/it]
Completed 27 simulations in 37.4s
Results saved to: data/output/results.csv

============================================================
OPTIMAL CONFIGURATION
============================================================
  Window Size: 32                ← Best window size found
  Payload Size: 512 bytes        ← Best payload size found
  Mean Goodput: 46086.22 B/s     ← Highest average goodput achieved
  Mean Efficiency: 40.10%        ← Efficiency at optimal config
```

---

### Command 4: Full Parameter Sweep

```bash
python main.py --sweep --runs 10
```

**What it does:** Runs the FULL parameter sweep with all 360 configurations.

**Full parameter space:**
- Window Sizes: [2, 4, 8, 16, 32, 64] (6 values)
- Payload Sizes: [128, 256, 512, 1024, 2048, 4096] (6 values)
- Runs per config: 10
- **Total: 6 × 6 × 10 = 360 simulations**

**Arguments:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--runs` | Number of runs per (W, L) configuration | `--runs 10` |
| `--parallel` | Run simulations in parallel (faster) | `--parallel` |
| `--workers` | Number of parallel workers | `--workers 4` |
| `--output` | Custom output CSV file path | `--output myresults.csv` |

**Example (parallel execution):**
```bash
python main.py --sweep --runs 10 --parallel --workers 4
```

**Output saved to:** `data/output/results.csv`

---

### Command 5: Generate Visualizations

```bash
python main.py --visualize
```

**What it does:** Generates heatmaps and 3D plots from simulation results.

**Arguments:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--csv` | Specify custom CSV file to visualize | `--csv data/output/full_results.csv` |

**Example:**
```bash
python main.py --visualize --csv data/output/results.csv
```

**Example output:**
```
============================================================
GENERATING VISUALIZATIONS
============================================================
Loaded 360 results from data/output/results.csv

Generating heatmap...
Heatmap saved to: data/output/plots/goodput_heatmap.png
Generating 3D surface plot...
Surface plot saved to: data/output/plots/goodput_surface.png
Generating contour plot...
Generating multi-view surface plot...

============================================================
VISUALIZATIONS GENERATED
============================================================
  Heatmap: data/output/plots/goodput_heatmap.png
  Surface: data/output/plots/goodput_surface.png
  Contour: data/output/plots/goodput_contour.png
  Multi-view: data/output/plots/goodput_surface_views.png
```

---

### Command 6: Generate Test Data File

```bash
python main.py --generate-test-file
```

**What it does:** Creates a 100 MB test data file for simulation input.

**Output:** Creates `data/input/test_data_100mb.bin`

---

## Output Files Explained

### 1. Results CSV (`data/output/results.csv`)

Contains all simulation results. Each row is one simulation run.

**Columns:**
| Column | Description |
|--------|-------------|
| `window_size` | Window size (W) used |
| `payload_size` | Payload size (L) in bytes |
| `run_id` | Run number (0-9 for 10 runs) |
| `seed` | Random seed used |
| `goodput` | Goodput in bytes/second |
| `goodput_mbps` | Goodput in Mbps |
| `efficiency` | Efficiency ratio (0-1) |
| `utilization` | Channel utilization (0-1) |
| `retransmissions` | Number of retransmissions |
| `retransmission_rate` | Retransmissions ÷ Original frames |
| `frame_error_rate` | Corrupted frames ÷ Total frames |
| `total_time` | Simulation time in seconds |
| `rtt_mean` | Average round-trip time |
| `data_valid` | True if received data matches sent data |
| `complete` | True if transfer completed successfully |

---

### 2. Heatmap (`data/output/plots/goodput_heatmap.png`)

A 2D color-coded visualization showing Goodput for each (W, L) combination.

**How to read:**
- **X-axis**: Payload Size (L) in bytes
- **Y-axis**: Window Size (W) - larger at top, smaller at bottom
- **Color**: Goodput value (yellow = high, purple = low)
- **Red box**: Optimal configuration with highest Goodput
- **Values**: Mean Goodput in MB/s displayed in each cell

**Interpretation:**
- Look for the "hot zone" (yellow/green cells) - these are the best configurations
- Purple/dark cells indicate poor performance
- The optimal (W, L) pair is highlighted with a red box

---

### 3. Surface Plot (`data/output/plots/goodput_surface.png`)

A 3D visualization showing Goodput as a surface over the (W, L) parameter space.

**How to read:**
- **X-axis**: Window Size (W)
- **Y-axis**: Payload Size (L) in bytes
- **Z-axis (height)**: Goodput in MB/s
- **Red star**: Optimal point (highest peak)

**Interpretation:**
- Higher surface = better performance
- "Cliffs" show where performance drops sharply
- The peak shows the optimal configuration

---

### 4. Contour Plot (`data/output/plots/goodput_contour.png`)

A top-down view of the 3D surface with contour lines (like a topographic map).

**How to read:**
- Contour lines connect points of equal Goodput
- Closer lines = steeper performance change
- Center of innermost contour = optimal region

---

## Project Structure

```
pcom_p2/
├── config.py                 # Fixed baseline parameters
├── main.py                   # CLI entry point (run simulations here)
├── requirements.txt          # Python dependencies
│
├── src/                      # Source code
│   ├── layers/               # Network layer implementations
│   │   ├── application_layer.py  # File I/O, data chunking
│   │   ├── transport_layer.py    # Segmentation, flow control
│   │   ├── link_layer.py         # ARQ integration
│   │   └── physical_layer.py     # Transmission timing, delays
│   │
│   ├── arq/                  # Selective Repeat ARQ protocol
│   │   ├── frame.py          # Frame structure (header, CRC)
│   │   ├── sender.py         # SR-ARQ sender logic
│   │   ├── receiver.py       # SR-ARQ receiver logic
│   │   └── timer.py          # Per-frame timer management
│   │
│   ├── channel/              # Channel models
│   │   └── gilbert_elliot.py # Two-state burst error model
│   │
│   └── utils/                # Utilities
│       ├── metrics.py        # Goodput, efficiency calculation
│       ├── buffer.py         # Buffer management
│       └── logger.py         # Logging utilities
│
├── simulation/               # Simulation engine
│   ├── simulator.py          # Event-driven simulator core
│   ├── runner.py             # Batch runner for parameter sweeps
│   └── parameter_sweep.py    # Parameter space definition
│
├── visualization/            # Plotting modules
│   ├── heatmap.py           # 2D Goodput heatmaps
│   └── surface_plot.py      # 3D surface and contour plots
│
├── optimization/             # Phase 2: Protocol optimizations
│   ├── adaptive_timeout.py  # Dynamic RTO (Jacobson/Karels)
│   ├── ack_optimization.py  # ACK strategies (delayed, cumulative)
│   └── dynamic_window.py    # Congestion-based window adjustment
│
├── data/                     # Data files
│   ├── input/               # Input test files
│   └── output/              # Results CSV and plots
│       ├── results.csv      # Simulation results
│       └── plots/           # Generated visualizations
│
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

---

## Fixed Parameters

As specified in the assignment:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Bit Rate | 10 Mbps | Channel transmission speed |
| Forward Propagation Delay | 40 ms | Delay for data frames |
| Reverse Propagation Delay | 10 ms | Delay for ACK frames |
| Processing Delay | 2 ms | Delay at receiver |
| Good State BER | 1 × 10⁻⁶ | Bit error rate in "good" channel state |
| Bad State BER | 5 × 10⁻³ | Bit error rate in "bad" channel state |
| P(G → B) | 0.002 | Probability of transitioning good → bad |
| P(B → G) | 0.05 | Probability of transitioning bad → good |
| Transport Header | 8 bytes | Header overhead per segment |
| Link Header | 24 bytes | Header overhead per frame |
| Receiver Buffer | 256 KB | Application-side buffer size |

---

## Parameter Space (Sweep Variables)

| Parameter | Values | Count |
|-----------|--------|-------|
| Window Size (W) | 2, 4, 8, 16, 32, 64 | 6 |
| Payload Size (L) | 128, 256, 512, 1024, 2048, 4096 bytes | 6 |
| Runs per config | 10 | 10 |
| **Total simulations** | 6 × 6 × 10 | **360** |

---

## Understanding the Results

### What is Goodput?

**Goodput = Useful Data Delivered ÷ Total Time**

- Measured in **bytes/second** or **Mbps**
- Only counts APPLICATION data that was successfully delivered
- EXCLUDES: headers, retransmissions, corrupted frames

**Why it matters:** Goodput tells you the ACTUAL useful throughput, not just raw data transmitted.

### What is Efficiency?

**Efficiency = Useful Data Delivered ÷ Total Bytes Transmitted**

- Measured as a **percentage (0-100%)**
- Low efficiency = lots of overhead or retransmissions
- High efficiency = most transmitted data was useful

### Finding the Optimal Configuration

The "best" configuration is the (W, L) pair with the **highest mean Goodput** across all runs.

**Trade-offs:**
- **Window too small** → Channel sits idle waiting for ACKs → Low goodput
- **Window too large** → More frames corrupted during burst errors → More retransmissions
- **Payload too small** → High header overhead → Low efficiency
- **Payload too large** → Higher probability of frame corruption → More retransmissions

---

## Complete Workflow Example

```bash
# Step 1: Verify installation and configuration
python main.py --config

# Step 2: Run a quick test sweep (27 simulations, ~1 minute)
python main.py --sweep --quick

# Step 3: Generate visualizations from quick test
python main.py --visualize

# Step 4: View results in data/output/plots/

# Step 5: Run full parameter sweep (360 simulations, ~30-60 minutes)
python main.py --sweep --runs 10 --parallel --workers 4

# Step 6: Generate final visualizations
python main.py --visualize

# Step 7: Check the optimal configuration in output
# Look for "OPTIMAL CONFIGURATION" in console output
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_selective_repeat.py -v
```

---

## Phase 2: AI-Assisted Optimization

For Phase 2 of the assignment:

1. Export results CSV to AI assistant
2. Request trend identification and optimization suggestions
3. Implement recommended improvements
4. Compare before/after performance

Document all interactions in `ai_logs/` directory.

---

## License

Academic project - not for redistribution.
