import matplotlib.pyplot as plt
import json
import numpy as np

def extract_max_hit_ratios(file_paths):
    result = []  # List of tuples: (dataset, TrueLabel_y, LRU_y)
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)

        algorithm_data = {}
        for config in data:
            if 'algorithm' in config:
                algorithm = config['algorithm']
                if 'ml' in algorithm and 'true' not in algorithm:
                    continue
                x = config['size'] * 16 / 1000
                if algorithm not in algorithm_data:
                    algorithm_data[algorithm] = []
                if 'hit_ratios' in config:
                    y = float(config['hit_ratios'][-1])
                else:
                    continue
                algorithm_data[algorithm].append((x, y))

        dataset = data[0]['dataset_name']
        true_y = lru_y = None

        for algo_name, values in algorithm_data.items():
            if values:
                x_max, y_max = max(values, key=lambda t: t[0])
                if 'true' in algo_name.lower():
                    true_y = y_max
                elif 'lru' in algo_name.lower():
                    lru_y = y_max

        result.append((dataset, true_y, lru_y))
    return result

def plot_comparison_bar_chart(results):
    datasets = [r[0] for r in results]
    true_vals = [r[1] for r in results]
    lru_vals = [r[2] for r in results]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(4, 3))
    bars1 = ax.bar(x - width/2, true_vals, width, label='Oracle Labelled')
    bars2 = ax.bar(x + width/2, lru_vals, width, label='LRU')

    # Add y-values above bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Prefix Cache Hit Ratio')
    plt.ylim(top=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # Style tweaks
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig('fig/bar_comparison_true_vs_lru.png', dpi=300)
    plt.show()


# Update this with your actual file paths
file_paths = [
    'results/98a63908b41686889a6ade39c37616e54d49974d/result_Nconv=300_1/exp_chatbot200.json',
    'results/98a63908b41686889a6ade39c37616e54d49974d/result_Nconv=300_1/exp_sharegpt200.json',
    'results/98a63908b41686889a6ade39c37616e54d49974d/result_Nconv=300_1/exp_lmsys200.json',
]

results = extract_max_hit_ratios(file_paths)
plot_comparison_bar_chart(results)