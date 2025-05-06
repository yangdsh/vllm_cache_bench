import matplotlib.pyplot as plt
import json

def plot_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Organize data by eviction algorithm
    algorithm_data = {}
    for config in data:
        #if config['request_rate'] != 0.01:
        #    continue
        algorithm = config['eviction_algorithm']
        if 'algorithm' in config:
            algorithm = config['algorithm']
            #if 'lru' not in algorithm:
            #    continue
        if True or 'true' in algorithm or 'true' in algorithm or algorithm == 'lru':
            if '0.7-0.8' in algorithm or '0.7-0.6' in algorithm or '0.7-1' in algorithm:
                continue
            if '0.7-0.7' in algorithm:
                algorithm = algorithm[:-4]
            size = config['size']
            hit_ratio = float(config['hit_ratios'][-1])

            if algorithm not in algorithm_data:
                algorithm_data[algorithm] = {'sizes': [], 'hit_ratios': [], 'miss_ratios': []}

            algorithm_data[algorithm]['sizes'].append(size)
            algorithm_data[algorithm]['hit_ratios'].append(hit_ratio)
            algorithm_data[algorithm]['miss_ratios'].append(1-hit_ratio)
    dataset_name = data[0]['dataset_name']

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
    plt.title(dataset_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save figure to file
    plt.savefig(f'hit_ratios_{dataset_name}.png', dpi=300)
    plt.show()


file_paths = ['results/Qwen2.5-0.5B-Instruct/exp_tay0422.json',
'results/Qwen2.5-0.5B-Instruct/exp_sharegpt003-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_sharegpt001-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_lmsys0003-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_lmsys003-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_lmsys001-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_lmsys003-3-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_lmsys001-3-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_chatbot001-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_gpt001-1200.json',
'results/Qwen2.5-0.5B-Instruct/exp_gpt001-1200-100.json',
'results/Qwen2.5-0.5B-Instruct/exp_code0428.json',
'results/Qwen2.5-0.5B-Instruct/exp_math0422.json',
'results/Qwen2.5-0.5B-Instruct/exp_math0428.json',
'results/Qwen2.5-0.5B-Instruct/exp_science0428.json',
]
for file_path in file_paths:
    plot_dataset(file_path)
