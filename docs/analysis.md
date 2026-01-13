# Trade-off Analysis

## Parameter Trade-offs

This document analyzes the trade-offs in the (W, L) parameter space.

## Frame Size Trade-off

### Large Frame Size (L) - Pros
- Lower overhead ratio: `payload / (payload + headers)`
- Fewer frames needed for same data amount
- Better efficiency in good channel conditions

### Large Frame Size (L) - Cons
- Higher probability of frame corruption
- More data lost per corrupted frame
- Larger retransmission penalty

### Analysis

With Gilbert-Elliot burst errors:
- P(frame error) ≈ 1 - (1 - BER)^(frame_bits)
- Larger frames are more likely to encounter bad state during transmission
- Example at average BER ≈ 10⁻⁴:
  - 128 B frame: ~10% error rate
  - 4096 B frame: ~27% error rate

## Window Size Trade-off

### Large Window Size (W) - Pros
- Better pipeline utilization
- Higher throughput potential during good periods
- More efficient use of bandwidth-delay product

### Large Window Size (W) - Cons
- More frames in flight during burst errors
- Potentially more simultaneous retransmissions
- Larger buffer requirements

### Analysis

The bandwidth-delay product determines minimum window:
```
BDP = Bit_Rate × RTT = 10 Mbps × ~100 ms = 1 Mbit = 125 KB
```

For full utilization:
```
W_optimal ≈ BDP / frame_size
```

For L=1024: W_optimal ≈ 125 KB / 1056 B ≈ 118 frames

## Expected Optimal Region

Based on theoretical analysis:

1. **Window Size**: 16-32
   - Covers bandwidth-delay product
   - Not so large that burst losses are catastrophic

2. **Payload Size**: 512-1024 bytes
   - Good overhead ratio (~3-6% for headers)
   - Reasonable frame error probability

## Goodput Formula

```
Goodput = (Application Bytes Delivered) / (Total Time)

       = (Data Size) / (Transmission Time)

       = (Data Size) / (Frames × Frame_Time × (1 + Retransmission_Rate))
```

Where:
- Frame_Time = Transmission_Time + Propagation_Delay
- Retransmission_Rate depends on BER and frame size

## Sensitivity Analysis

### Sensitivity to BER
- Higher BER → smaller optimal frame size
- Higher BER → smaller optimal window (or adaptive window)

### Sensitivity to Delay
- Higher delay → need larger window for same utilization
- Asymmetric delays affect optimal buffer sizing

## Experimental Validation

After running 360 simulations, plot:
1. Goodput heatmap: `Goodput(W, L)`
2. Identify the peak region
3. Compare with theoretical predictions
4. Document any deviations and explanations

## Expected Results Template

| W | L | Goodput (MB/s) | Efficiency | Retx Rate |
|---|---|----------------|------------|-----------|
| 8 | 512 | ___ | ___ | ___ |
| 16 | 1024 | ___ | ___ | ___ |
| ... | ... | ... | ... | ... |

The optimal (W*, L*) should be highlighted along with:
- Achieved goodput
- Comparison to theoretical maximum
- Analysis of why this configuration is optimal
