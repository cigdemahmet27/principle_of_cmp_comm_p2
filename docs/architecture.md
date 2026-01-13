# System Architecture

## Overview

The simulator implements a cross-layer communication stack with four layers:

```
┌─────────────────────────────────────┐
│         Application Layer           │
│    (File I/O, Data Verification)    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│          Transport Layer            │
│  (Segmentation, Reassembly, Flow)   │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│            Link Layer               │
│     (Selective Repeat ARQ)          │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│          Physical Layer             │
│   (Gilbert-Elliot Channel Model)    │
└─────────────────────────────────────┘
```

## Layer Responsibilities

### Application Layer (`src/layers/application_layer.py`)

- Reads 100 MB input file
- Delivers data to transport layer in chunks
- Receives and verifies data at destination
- Calculates checksums for integrity verification

### Transport Layer (`src/layers/transport_layer.py`)

- **Sender Side**:
  - Segments application data
  - Adds transport header (8 bytes)
  - Sequence numbers for reassembly

- **Receiver Side**:
  - Reassembles segments in order
  - Manages 256 KB receive buffer
  - Provides backpressure to link layer when buffer full

### Link Layer (`src/layers/link_layer.py`)

Implements Selective Repeat ARQ:

- **Sender**:
  - Configurable send window size
  - Per-frame retransmission timers
  - Selective retransmission on timeout/NAK
  - Frame buffering for retransmission

- **Receiver**:
  - Matching receive window
  - Out-of-order frame buffering
  - Selective ACK generation
  - In-order delivery to transport

### Physical Layer (`src/layers/physical_layer.py`)

- Simulates channel transmission
- Gilbert-Elliot burst error model
- Asymmetric propagation delays
- CRC error detection

## Data Flow

### Sender Path

```
Application Data (100 MB)
         │
         ▼
    ┌─────────┐
    │Segment  │ → Transport segments (payload + 8B header)
    └────┬────┘
         ▼
    ┌─────────┐
    │Frame    │ → Link frames (segment + 24B header)
    └────┬────┘
         ▼
    ┌─────────┐
    │Transmit │ → Physical transmission with errors
    └────┬────┘
         ▼
      Channel
```

### Receiver Path

```
      Channel
         │
         ▼
    ┌─────────┐
    │Receive  │ ← CRC check, error detection
    └────┬────┘
         ▼
    ┌─────────┐
    │Buffer   │ ← Out-of-order buffering, ACK
    └────┬────┘
         ▼
    ┌─────────┐
    │Reassemble│ ← In-order delivery
    └────┬────┘
         ▼
    Application
```

## Gilbert-Elliot Channel Model

Two-state Markov chain:

```
     1 - P(G→B)        1 - P(B→G)
         ┌──┐              ┌──┐
         │  │              │  │
         ▼  │              ▼  │
    ┌─────────┐       ┌─────────┐
    │  GOOD   │──────▶│   BAD   │
    │ BER=10⁻⁶│◀──────│ BER=5×10⁻³│
    └─────────┘       └─────────┘
        P(G→B)=0.002   P(B→G)=0.05
```

**Steady-State Probabilities**:
- π(Good) ≈ 0.962
- π(Bad) ≈ 0.038

**Average BER** ≈ 1 × 10⁻⁴

## Timing Model

For a frame with payload L bytes:

```
Data Frame Transmission Time = (L + 24 + 8) × 8 / 10 Mbps

Total Forward Delay = Tx_data + Prop_forward + Processing
                    = Tx_data + 40ms + 2ms

Total Reverse Delay = Tx_ack + Prop_reverse + Processing
                    = Tx_ack + 10ms + 2ms

RTT ≈ Tx_data + 40ms + 2ms + Tx_ack + 10ms + 2ms
    = Tx_data + Tx_ack + 54ms
```

## Event-Driven Simulation

The simulator uses discrete-event simulation:

1. Event types: FRAME_ARRIVAL, TIMER_EXPIRY, DATA_READY
2. Events stored in priority queue (heap)
3. Simulation advances to next event time
4. Events processed in chronological order

```python
while not complete and time < max_time:
    event = heapq.heappop(event_queue)
    current_time = event.time
    process_event(event)
```
