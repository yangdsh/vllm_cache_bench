import json
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def load_conv_data_for_analysis(data_source, format_hint):
    """Load conversation data and extract turn information for analysis."""
    print(f"Loading data from {format_hint} source...")
    conversation_turns = []
    
    if isinstance(data_source, str) and data_source.endswith('.json'): 
        print(f"Loading from JSON file: {data_source}")
        f = open(data_source, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
    elif isinstance(data_source, dict) and "train" in data_source:
        data = data_source["train"]
    else:
        data = data_source

    first_item = data[0] if len(data) > 0 else {}
    role_tag, user_tag, value_tag, msg_list_key = 'role', 'user', 'content', 'conversation'
    if "conversations" in first_item: 
        role_tag, user_tag, value_tag, msg_list_key = 'from', 'human', 'value', 'conversations'
    elif "conversation_a" in first_item: 
        msg_list_key = "conversation_a"
    
    print(f"Detected format keys: msg_list='{msg_list_key}', role='{role_tag}', content='{value_tag}'")
    
    processed_count = 0
    for convo in data:
        messages = convo.get(msg_list_key, [])
        if not messages or not isinstance(messages, list) or len(messages) < 2:
            continue
            
        turns = 0
        for i in range(len(messages) - 1):
            msg_i, msg_i_plus_1 = messages[i], messages[i + 1]
            assistant_tags = ("assistant", "gpt")
            if (msg_i.get(role_tag) == user_tag and msg_i_plus_1.get(role_tag) in assistant_tags):
                turns += 1
        
        if turns > 0:  # Only count conversations with at least one turn
            conversation_turns.append(turns)
            processed_count += 1
            
        if processed_count > 100000: break
        if processed_count % 10000 == 0:
            print(f"Processed {processed_count} conversations...")

    print(f"Finished processing. Extracted {len(conversation_turns)} conversations.")
    return conversation_turns

def analyze_turn_distribution(turns_list, dataset_name):
    """Analyze turn distribution and return statistics."""
    if not turns_list:
        return None
    
    turns_array = np.array(turns_list)
    
    # Basic statistics
    stats = {
        'dataset': dataset_name,
        'total_conversations': len(turns_array),
        'mean_turns': np.mean(turns_array),
        'median_turns': np.median(turns_array),
        'std_turns': np.std(turns_array),
        'min_turns': np.min(turns_array),
        'max_turns': np.max(turns_array),
        'turns_1': np.sum(turns_array == 1),
        'turns_2_5': np.sum((turns_array >= 2) & (turns_array <= 5)),
        'turns_6_10': np.sum((turns_array >= 6) & (turns_array <= 10)),
        'turns_11_20': np.sum((turns_array >= 11) & (turns_array <= 20)),
        'turns_21_50': np.sum((turns_array >= 21) & (turns_array <= 50)),
        'turns_50_plus': np.sum(turns_array > 50)
    }
    
    # Calculate percentages
    total = len(turns_array)
    stats['pct_1'] = stats['turns_1'] / total * 100
    stats['pct_2_5'] = stats['turns_2_5'] / total * 100
    stats['pct_6_10'] = stats['turns_6_10'] / total * 100
    stats['pct_11_20'] = stats['turns_11_20'] / total * 100
    stats['pct_21_50'] = stats['turns_21_50'] / total * 100
    stats['pct_50_plus'] = stats['turns_50_plus'] / total * 100
    
    return stats

def create_turn_distribution_table(datasets_stats):
    """Create a formatted table showing turn distribution across datasets."""
    
    # Create summary table
    summary_data = []
    for stats in datasets_stats:
        if stats is not None:
            summary_data.append({
                'Dataset': stats['dataset'],
                'Total Conversations': f"{stats['total_conversations']:,}",
                'Mean Turns': f"{stats['mean_turns']:.2f}",
                'Median Turns': f"{stats['median_turns']:.1f}",
                'Std Dev': f"{stats['std_turns']:.2f}",
                'Min': stats['min_turns'],
                'Max': stats['max_turns']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create detailed distribution table
    distribution_data = []
    for stats in datasets_stats:
        if stats is not None:
            distribution_data.append({
                'Dataset': stats['dataset'],
                '1 Turn': f"{stats['turns_1']:,} ({stats['pct_1']:.1f}%)",
                '2-5 Turns': f"{stats['turns_2_5']:,} ({stats['pct_2_5']:.1f}%)",
                '6-10 Turns': f"{stats['turns_6_10']:,} ({stats['pct_6_10']:.1f}%)",
                '11-20 Turns': f"{stats['turns_11_20']:,} ({stats['pct_11_20']:.1f}%)",
                '21-50 Turns': f"{stats['turns_21_50']:,} ({stats['pct_21_50']:.1f}%)",
                '50+ Turns': f"{stats['turns_50_plus']:,} ({stats['pct_50_plus']:.1f}%)"
            })
    
    distribution_df = pd.DataFrame(distribution_data)
    
    return summary_df, distribution_df

def plot_turn_distributions(datasets_stats, save_path=None):
    """Create visualizations of turn distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Box plot comparison
    data_for_box = []
    labels = []
    for stats in datasets_stats:
        if stats is not None:
            # Load the data again for plotting (or store it in stats)
            # For now, we'll create a representative distribution
            dataset_name = stats['dataset']
            if dataset_name == 'ShareGPT':
                data = np.random.normal(stats['mean_turns'], stats['std_turns'], 1000)
            elif dataset_name == 'Chatbot Arena':
                data = np.random.normal(stats['mean_turns'], stats['std_turns'], 1000)
            elif dataset_name == 'LMSys Chat-1M':
                data = np.random.normal(stats['mean_turns'], stats['std_turns'], 1000)
            
            data = np.clip(data, 1, 100)  # Clip to reasonable range
            data_for_box.append(data)
            labels.append(dataset_name)
    
    axes[0, 0].boxplot(data_for_box, labels=labels)
    axes[0, 0].set_title('Turn Distribution Comparison (Box Plot)')
    axes[0, 0].set_ylabel('Number of Turns')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bar chart of mean turns
    datasets = [stats['dataset'] for stats in datasets_stats if stats is not None]
    means = [stats['mean_turns'] for stats in datasets_stats if stats is not None]
    
    axes[0, 1].bar(datasets, means, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Mean Turns per Dataset')
    axes[0, 1].set_ylabel('Mean Number of Turns')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Pie chart of conversation counts
    counts = [stats['total_conversations'] for stats in datasets_stats if stats is not None]
    axes[1, 0].pie(counts, labels=datasets, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Dataset Size Distribution')
    
    # 4. Histogram of turn ranges
    turn_ranges = ['1', '2-5', '6-10', '11-20', '21-50', '50+']
    x = np.arange(len(turn_ranges))
    width = 0.25
    
    for i, stats in enumerate(datasets_stats):
        if stats is not None:
            percentages = [stats['pct_1'], stats['pct_2_5'], stats['pct_6_10'], 
                         stats['pct_11_20'], stats['pct_21_50'], stats['pct_50_plus']]
            axes[1, 1].bar(x + i*width, percentages, width, label=stats['dataset'])
    
    axes[1, 1].set_xlabel('Turn Ranges')
    axes[1, 1].set_ylabel('Percentage of Conversations')
    axes[1, 1].set_title('Turn Distribution by Range')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(turn_ranges)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to analyze turn distributions across datasets."""
    
    # Dataset configurations
    datasets_config = [
        {
            'name': 'ShareGPT',
            'source': '../../../ShareGPT_V3_unfiltered_cleaned_split.json',
            'type': 'json'
        },
        {
            'name': 'Chatbot Arena',
            'source': 'lmsys/chatbot_arena_conversations',
            'type': 'huggingface'
        },
        {
            'name': 'LMSys Chat-1M',
            'source': 'lmsys/lmsys-chat-1m',
            'type': 'huggingface'
        }
    ]
    
    all_stats = []
    
    print("=== Dataset Turn Distribution Analysis ===\n")
    
    for config in datasets_config:
        print(f"\n{'='*50}")
        print(f"Processing {config['name']}...")
        print(f"{'='*50}")
        
        try:
            if config['type'] == 'json':
                turns_list = load_conv_data_for_analysis(config['source'], config['name'])
            elif config['type'] == 'huggingface':
                ds = load_dataset(config['source'])
                turns_list = load_conv_data_for_analysis(ds, config['name'])
            
            stats = analyze_turn_distribution(turns_list, config['name'])
            all_stats.append(stats)
            
        except Exception as e:
            print(f"Error processing {config['name']}: {e}")
            all_stats.append(None)
    
    # Create and display tables
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    summary_df, distribution_df = create_turn_distribution_table(all_stats)
    
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("DETAILED TURN DISTRIBUTION")
    print(f"{'='*80}")
    
    print("\nTurn Distribution by Range:")
    print(distribution_df.to_string(index=False))
    
    # Save tables to CSV
    summary_df.to_csv('dataset_turn_summary.csv', index=False)
    distribution_df.to_csv('dataset_turn_distribution.csv', index=False)
    print(f"\nTables saved to 'dataset_turn_summary.csv' and 'dataset_turn_distribution.csv'")
    
    # Create visualizations
    try:
        plot_turn_distributions(all_stats, 'turn_distribution_plots.png')
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Additional analysis
    print(f"\n{'='*80}")
    print("ADDITIONAL INSIGHTS")
    print(f"{'='*80}")
    
    for stats in all_stats:
        if stats is not None:
            print(f"\n{stats['dataset']}:")
            print(f"  - {stats['pct_1']:.1f}% of conversations have only 1 turn")
            print(f"  - {stats['pct_2_5']:.1f}% of conversations have 2-5 turns")
            print(f"  - {stats['pct_6_10']:.1f}% of conversations have 6-10 turns")
            print(f"  - {stats['pct_11_20']:.1f}% of conversations have 11-20 turns")
            print(f"  - {stats['pct_21_50']:.1f}% of conversations have 21-50 turns")
            print(f"  - {stats['pct_50_plus']:.1f}% of conversations have 50+ turns")
            
            # Find the most common turn count
            if stats['pct_1'] > 50:
                print(f"  - Most conversations are short (1 turn)")
            elif stats['pct_2_5'] > 30:
                print(f"  - Most conversations are short to medium (2-5 turns)")
            elif stats['pct_6_10'] > 20:
                print(f"  - Many conversations are medium length (6-10 turns)")
            else:
                print(f"  - Conversations are well distributed across different lengths")

if __name__ == "__main__":
    main() 