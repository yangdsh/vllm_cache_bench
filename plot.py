import matplotlib.pyplot as plt
import json

# Read data from file
dir = 'results/Qwen2.5-0.5B/exp_tay-1-hit-wonder.json'
dir = 'results/Qwen2.5-0.5B/exp_lmsys.json'
with open(dir, 'r') as file:
    data = json.load(file)

# Organize data by eviction algorithm
algorithm_data = {}
for config in data:
    if config['request_rate'] != 0.01:
        continue
    algorithm = config['eviction_algorithm']
    if 'algorithm' in config:
        algorithm = config['algorithm']
    if True or 'true' in algorithm or 'true' in algorithm or algorithm == 'lru':
        if '0.7-0.8' in algorithm or '0.7-0.6' in algorithm or '0.7-1' in algorithm:
            continue
        if '0.7-0.7' in algorithm:
            algorithm = algorithm[:-4]
        size = config['size']
        hit_ratio = float(config['hit_ratios'][-1])

        if algorithm not in algorithm_data:
            algorithm_data[algorithm] = {'sizes': [], 'hit_ratios': []}

        algorithm_data[algorithm]['sizes'].append(size)
        algorithm_data[algorithm]['hit_ratios'].append(hit_ratio)

# Plotting
plt.figure(figsize=(12, 6))
for algorithm, values in algorithm_data.items():
    sorted_pairs = sorted(zip(values['sizes'], values['hit_ratios']))
    sizes_sorted, hit_ratios_sorted = zip(*sorted_pairs)
    print(algorithm, hit_ratios_sorted)
    plt.plot(sizes_sorted, hit_ratios_sorted, marker='o', label=algorithm)

plt.ylim(bottom=0)
plt.xlabel('Cache Size')
plt.ylabel('Hit Ratio')
plt.title('Hit Ratios by Eviction Algorithm and Cache Size (Last Value)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save figure to file
plt.savefig('hit_ratios_line_plot.png', dpi=300)
plt.show()