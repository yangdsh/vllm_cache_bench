import json
from PIL.TiffTags import TagInfo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any

class ResultPlotter:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.results = json.load(f)
        self.df = self._process_results()

    def _process_results(self) -> pd.DataFrame:
        processed_data = []
        for res in self.results:
            flat_res = {}
            flat_res.update(res.get('configuration', {}))
            flat_res.update(res.get('benchmark_results', {}))
            flat_res.update(res.get('cache_statistics', {}))
            
            # Extract dataset name from the file path
            dataset_file = flat_res.get('dataset_file_path', '')
            if dataset_file:
                dataset_name = os.path.basename(dataset_file).split('.')[0]
                flat_res['dataset'] = dataset_name
            else:
                flat_res['dataset'] = 'unknown'

            processed_data.append(flat_res)
        return pd.DataFrame(processed_data)

    def plot_metrics(self, metrics: List[str], output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        datasets = self.df['dataset'].unique()
        for dataset in datasets:
            dataset_df = self.df[self.df['dataset'] == dataset].copy()
            
            # Renaming for clarity in plots
            dataset_df['cache_eviction_strategy'] = dataset_df['cache_eviction_strategy'].replace({
                'standard': 'baseline',
                'conversation_aware': 'conv-aware'
            })

            model_name = dataset_df['model_name'].iloc[0]

            # Create a 2x2 subplot for the four metrics
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()

            for idx, metric in enumerate(metrics):
                ax = axes[idx]
                sns.lineplot(data=dataset_df, x='cache_size_gb', y=metric, hue='cache_eviction_strategy', marker='o', ax=ax)
                ax.set_xlabel("Cache Size (GB)")
                ax.set_ylabel(metric)
                ax.set_title(metric)
                # ax.grid(True)
                if idx == 0:
                    ax.legend(title='Eviction Strategy')
                else:
                    ax.get_legend().remove()

            plt.tight_layout()
            plot_filename = f"{dataset}_{model_name}_all_metrics.png"
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined plot: {plot_filename}")


if __name__ == '__main__':
    # Adjust the file path and output directory as needed
    tag = 'qwen8b_2025-07-30'
    file_path = f'../experiment_logs/{tag}/summary.json'
    output_dir = tag
    
    plotter = ResultPlotter(file_path)
    
    # Define the metrics you want to plot
    metrics_to_plot = [
        'mean_ttft_ms', 
        'p99_ttft_ms',
        'lmcache_retrieved_rate',
        'lmcache_hit_rate'
    ]
    
    plotter.plot_metrics(metrics_to_plot, output_dir) 