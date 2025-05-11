import matplotlib.pyplot as plt
import json

def plot_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Organize data by eviction algorithm
    algorithm_data = {}
    for config in reversed(data):
        #if config['request_rate'] != 0.01:
        #    continue
        algorithm = config['eviction_algorithm']
        if 'algorithm' in config:
            algorithm = config['algorithm']
            if 'true' in algorithm:
                continue
        size = config['size']
        if algorithm not in algorithm_data or size not in algorithm_data[algorithm]['sizes']:
            hit_ratio = float(config['hit_ratios'][-1])

            if algorithm not in algorithm_data:
                algorithm_data[algorithm] = {'sizes': [], 'hit_ratios': [], 'miss_ratios': []}

            algorithm_data[algorithm]['sizes'].append(size)
            algorithm_data[algorithm]['hit_ratios'].append(hit_ratio)
            algorithm_data[algorithm]['miss_ratios'].append(1-hit_ratio)
    dataset_name = data[0]['dataset_name']

    # Plotting
    plt.figure(figsize=(3.5, 2.5))
    for algorithm, values in algorithm_data.items():
        sorted_pairs = sorted(zip(values['sizes'], values['hit_ratios']))
        sizes_sorted, hit_ratios_sorted = zip(*sorted_pairs)
        print(algorithm, hit_ratios_sorted)
        plt.plot(sizes_sorted, hit_ratios_sorted, marker='o', label=algorithm)

    plt.ylim(bottom=0)
    plt.xlabel('Cache Size (# of blocks)')
    plt.ylabel('Hit Ratio')
    # plt.title(dataset_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save figure to file
    print(f'fig/hit_ratios_{dataset_name}.png')
    plt.savefig(f'fig/hit_ratios_{dataset_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()

dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/result_Nconv=300_1'
file_paths = [f'{dir}/exp_chatbot001-1200.json',
f'{dir}/exp_sharegpt001-1200.json',
f'{dir}/exp_tay-600.json',
f'{dir}/exp_tay-300.json',
f'{dir}/exp_lmsys300.json',
f'{dir}/exp_chatbot300.json',
f'{dir}/exp_sharegpt300.json',
f'{dir}/exp_lmsys200.json',
f'{dir}/exp_chatbot200.json',
f'{dir}/exp_sharegpt200.json',
f'{dir}/exp_tay300.json',
]
for file_path in file_paths:
    plot_dataset(file_path)

# 200 does not have the conversaton end time adjustment by total output length
