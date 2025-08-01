import matplotlib.pyplot as plt
import json

name = {'ml': 'LPC', 'lru': 'LRU', 'ml-online': 'LPC (online)', 'ml-online-finetune': 'LPC (finetune)'}

def plot_dataset(file_path, y_label, x_label):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Organize data by eviction algorithm
    algorithm_data = {}
    for config in reversed(data):
        #if config['request_rate'] != 0.01:
        #    continue
        if config['time_limit'] != 1200:
            continue
        if 'prompt_tokens' not in config:
            continue
        if 'eviction_algorithm' in config:
            algorithm = config['eviction_algorithm']
            if 'true' in algorithm:
                continue
        x = config[x_label]
        # add extra memory overhead for ml
        if 'ml' in algorithm:
            x += 250
        # number of tokens in the cache
        x = x * 16 / 1000
        if algorithm not in algorithm_data or x not in algorithm_data[algorithm]['xs']:
            if y_label == 'hit_ratios':
                avg_hit_ratios = [float(r) for r in config[y_label]]
                prompt_tokens = [float(p) for p in config['prompt_tokens']]
                n = len(prompt_tokens)
                start_index = n // 2

                prompt_tokens = prompt_tokens[start_index:]
                avg_hit_ratios = avg_hit_ratios[start_index:]
                hit_tokens = []
                for i in range(len(avg_hit_ratios)):
                    hit_tokens.append(avg_hit_ratios[i] * prompt_tokens[i])

                total_hit_tokens_second_half = sum(hit_tokens)
                total_prompt_tokens_second_half = sum(prompt_tokens)

                if total_prompt_tokens_second_half == 0:
                    y = 0
                else:
                    y = total_hit_tokens_second_half / total_prompt_tokens_second_half
            else:
                y = float(config[y_label])

            if algorithm not in algorithm_data:
                algorithm_data[algorithm] = {'xs': [], y_label: [], 'miss_ratios': []}

            algorithm_data[algorithm]['xs'].append(x)
            algorithm_data[algorithm][y_label].append(y)
            algorithm_data[algorithm]['miss_ratios'].append(1-y)
    dataset_name = data[0]['dataset_name']

    # Plotting
    plt.figure(figsize=(3.5, 2.7))
    for algorithm in ['lru']:
        values = algorithm_data[algorithm]
        sorted_pairs = sorted(zip(values['xs'], values[y_label]))
        xs_sorted, ys_sorted = zip(*sorted_pairs)
        # print(algorithm, ys_sorted)
        if algorithm == 'lru':
            lru_hits = ys_sorted
        else:
            for i in range(len(ys_sorted)):
                print(ys_sorted[i] - lru_hits[i], (ys_sorted[i] - lru_hits[i]) / lru_hits[i])
        plt.plot(xs_sorted, ys_sorted, marker='o', label=name[algorithm])

    plt.ylim(bottom=0)
    # plt.ylim(top=0.5)
    plt.xlabel('Cache Size (×10³ tokens)')
    if y_label == 'p99_ttft':
        plt.ylabel('P99 TTFT (ms)')
    else:    
        plt.ylabel('Hit Ratio')
    # plt.title(dataset_name)
    # plt.grid(True, linestyle='--', alpha=0.7)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.tight_layout()

    # Save figure to file
    print(f'fig/{y_label}_{dataset_name}.png')
    plt.savefig(f'fig/{y_label}_{dataset_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.show()

dir = 'results/2df580c02b34307b00ccd91309e67ec5a89987a9'
#dir = 'results/a29cae3df5d16cc895083497dad6ba9530c7d84c'
#dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/result_Nconv=300_1'
file_paths = [
    f'{dir}/exp_lmsys_size-8b.json',
    f'{dir}/exp_chatbot_size-8b.json',
    f'{dir}/exp_sharegpt_size-8b.json',
]
file_paths = [
    f'{dir}/exp_sharegpt_size-lmcache.json',
    f'{dir}/exp_lmsys_size-lmcache.json',
    f'{dir}/exp_chatbot_size-lmcache.json',
]
#file_paths = [f'{dir}/exp_chatbot200.json',
#f'{dir}/exp_sharegpt200.json',
#f'{dir}/exp_lmsys200.json',
#]

for file_path in file_paths:
    plot_dataset(file_path, 'hit_ratios', 'size')

# 200 does not have the conversaton end time adjustment by total output length
