import json
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

            for metric in metrics:
                plt.figure(figsize=(4, 4))
                
                # Use 'cache_eviction_strategy' for hue and 'cache_size_gb' for x-axis
                sns.lineplot(data=dataset_df, x='cache_size_gb', y=metric, hue='cache_eviction_strategy', marker='o')
                
                plt.xlabel("Cache Size (GB)")
                plt.ylabel(metric)
                # plt.grid(True)
                
                plot_filename = f"{dataset}_{model_name}_{metric}.png"
                plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
                plt.close()
                print(f"Saved plot: {plot_filename}")


if __name__ == '__main__':
    # Adjust the file path and output directory as needed
    file_path = '../experiment_logs/qwen8b_2025-07-23/summary.json'
    output_dir = 'plots'
    
    plotter = ResultPlotter(file_path)
    
    # Define the metrics you want to plot
    metrics_to_plot = [
        'mean_ttft_ms', 
        'p99_ttft_ms',
        'vllm_hit_rate',
        'lmcache_hit_rate'
    ]
    
    plotter.plot_metrics(metrics_to_plot, output_dir) 