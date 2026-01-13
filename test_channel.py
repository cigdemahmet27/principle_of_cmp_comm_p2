"""Test channel behavior"""
from src.channel.gilbert_elliot import GilbertElliottChannel

channel = GilbertElliottChannel(seed=42)

print('Testing channel with 1000 frames of 512 bytes each...')

corrupted = 0
for i in range(1000):
    result, _ = channel.transmit_frame(512 * 8)  # 512 bytes = 4096 bits
    if result:
        corrupted += 1

stats = channel.get_statistics()
transitions = stats['state_transitions']
time_good = stats['time_in_good']
time_bad = stats['time_in_bad']
observed_ber = stats['observed_ber']
theoretical_ber = stats['theoretical_avg_ber']

print(f'Frames corrupted: {corrupted}/1000 = {corrupted/10:.1f}%')
print(f'State transitions: {transitions}')
print(f'Time in Good (bit-ticks): {time_good}')
print(f'Time in Bad (bit-ticks): {time_bad}')
print(f'Actual fraction in Bad: {time_bad/(time_good+time_bad)*100:.2f}%')
print(f'Expected fraction in Bad: 3.85%')
print()
print(f'Observed BER: {observed_ber:.6f}')
print(f'Theoretical BER: {theoretical_ber:.6f}')
print()
print('=== DIAGNOSIS ===')
print(f'Total bits: {time_good + time_bad}')
print(f'Transitions per frame: {transitions / 1000:.1f}')
print('EXPECTED: ~0.002 transitions per frame (once every 500 frames)')
print('ACTUAL: Channel transitions are happening WAY too often!')
