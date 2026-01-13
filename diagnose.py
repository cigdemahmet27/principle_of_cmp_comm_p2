"""Diagnose why goodput values are low"""
from config import *

print('=' * 60)
print('DIAGNOSIS: WHY IS GOODPUT SO LOW?')
print('=' * 60)

print('\n=== CHANNEL PARAMETERS ===')
print(f'Bit Rate: {BIT_RATE / 1e6} Mbps')
print(f'Forward Delay: {FORWARD_PROPAGATION_DELAY * 1000} ms')
print(f'Reverse Delay: {REVERSE_PROPAGATION_DELAY * 1000} ms')
print(f'RTT: {(FORWARD_PROPAGATION_DELAY + REVERSE_PROPAGATION_DELAY + 2*PROCESSING_DELAY) * 1000} ms')

print('\n=== GILBERT-ELLIOT CHANNEL ===')
print(f'Good State BER: {GOOD_STATE_BER}')
print(f'Bad State BER: {BAD_STATE_BER}')
print(f'P(Good->Bad): {P_GOOD_TO_BAD}')
print(f'P(Bad->Good): {P_BAD_TO_GOOD}')

# Steady state probabilities
pi_good = P_BAD_TO_GOOD / (P_GOOD_TO_BAD + P_BAD_TO_GOOD)
pi_bad = P_GOOD_TO_BAD / (P_GOOD_TO_BAD + P_BAD_TO_GOOD)
print(f'\nSteady-state P(Good): {pi_good:.4f} ({pi_good*100:.2f}%)')
print(f'Steady-state P(Bad): {pi_bad:.4f} ({pi_bad*100:.2f}%)')

avg_ber = pi_good * GOOD_STATE_BER + pi_bad * BAD_STATE_BER
print(f'Average BER: {avg_ber:.6f} = {avg_ber:.2e}')

print('\n=== FRAME ERROR PROBABILITY ===')
print('This is the probability that a frame gets corrupted:')
for L in [128, 256, 512, 1024, 2048, 4096]:
    total_bits = (L + LINK_HEADER_SIZE) * 8  
    p_no_error_good = (1 - GOOD_STATE_BER) ** total_bits
    p_no_error_bad = (1 - BAD_STATE_BER) ** total_bits
    p_frame_ok = pi_good * p_no_error_good + pi_bad * p_no_error_bad
    p_frame_error = 1 - p_frame_ok
    print(f'  L={L:4d} bytes ({total_bits:5d} bits): P(error) = {p_frame_error*100:.1f}%')

print('\n=== BANDWIDTH-DELAY PRODUCT ===')
rtt = FORWARD_PROPAGATION_DELAY + REVERSE_PROPAGATION_DELAY + 2*PROCESSING_DELAY
bdp_bytes = BIT_RATE * rtt / 8
print(f'BDP = {bdp_bytes:.0f} bytes = {bdp_bytes/1024:.1f} KB')
print(f'To fully utilize channel, need W * L >= {bdp_bytes:.0f} bytes')

print('\n=== THEORETICAL MAX GOODPUT ===')
for W in [2, 4, 8, 16, 32, 64]:
    for L in [256, 512]:
        if W * L >= bdp_bytes:
            # Can keep pipe full
            efficiency = L / (L + LINK_HEADER_SIZE + TRANSPORT_HEADER_SIZE)
            max_goodput = BIT_RATE * efficiency / 8 / 1e6
            print(f'W={W:2d}, L={L:4d}: Max Goodput = {max_goodput:.2f} MB/s (if no errors)')

print('\n=== THE PROBLEMS ===')
print('1. In BAD state, BER = 0.005 means:')
print('   - For L=512: ~17% of frames get corrupted')
print('   - For L=4096: ~75% of frames get corrupted!')
print()
print('2. We spend ~3.8% of time in BAD state')
print('   - But during that time, almost ALL frames are corrupted')
print('   - This causes BURST of retransmissions')
print()
print('3. Small data size (50 KB) in test')
print('   - Not enough time to reach steady-state performance')
print()
print('4. Retransmission overhead:')
print('   - Each corrupted frame needs to be resent')
print('   - During burst errors, many frames in window get corrupted')
print('   - All need retransmission -> queuing delays')
