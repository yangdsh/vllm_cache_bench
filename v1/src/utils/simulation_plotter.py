import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from typing import List, Dict, Any

class SimulationResultPlotter:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.results = json.load(f)
        self.df = self._process_results()

    def _process_results(self) -> pd.DataFrame:
        processed_data = []
        for res in self.results:
            flat_res = {}
            flat_res.update(res.get('configuration', {}))
            flat_res.update(res.get('basic_stats', {}))
            
            # Extract dataset name from the file path
            dataset_file = flat_res.get('dataset_file_path', '')
            if dataset_file:
                dataset_name = os.path.basename(dataset_file).split('.')[0]
                flat_res['dataset'] = dataset_name
            else:
                flat_res['dataset'] = 'unknown'

            # Handle LightGBM modes separately
            eviction_strategy = flat_res.get('cache_eviction_strategy', '')
            lightgbm_mode = flat_res.get('lightgbm_mode', '')

            if eviction_strategy == 'conversation_aware':
                flat_res['cache_eviction_strategy'] = f'(ours) rule-based'
            elif eviction_strategy == 'oracle':
                flat_res['cache_eviction_strategy'] = '(upperbound) oracle'
            
            if eviction_strategy == 'lightgbm' and lightgbm_mode:
                flat_res['cache_eviction_strategy'] = f'(ours) LightGBM-{lightgbm_mode}'

            processed_data.append(flat_res)
        return pd.DataFrame(processed_data)

    def plot_token_hit_ratio(self, output_dir: str):
        """Plot token hit ratio for different cache sizes and eviction strategies"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        datasets = self.df['dataset'].unique()
        for dataset in datasets:
            dataset_df = self.df[self.df['dataset'] == dataset].copy()
            
            # Convert cache_size_gb to numeric
            dataset_df['cache_size_gb'] = pd.to_numeric(dataset_df['cache_size_gb'], errors='coerce')

            model_name = dataset_df['model_name'].iloc[0] if 'model_name' in dataset_df.columns else 'unknown'

            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot token hit ratio for different eviction strategies
            sns.lineplot(
                data=dataset_df, 
                x='cache_size_gb', 
                y='token_hit_ratio', 
                hue='cache_eviction_strategy', 
                marker='o',
                linewidth=2,
                markersize=8
            )
            
            plt.xlabel("Cache Size (GB)", fontsize=12)
            plt.ylabel("Token Hit Ratio", fontsize=12)
            plt.title(f"Token Hit Ratio vs Cache Size - {dataset} ({model_name})", fontsize=14)
            plt.legend(title='Eviction Strategy', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            
            plt.tight_layout()
            plot_filename = f"{dataset}_{model_name}_token_hit_ratio.png"
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved token hit ratio plot: {plot_filename}")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate simulation plots from summary.json results')
    parser.add_argument('--summary-file', type=str, required=True, help='Path to summary.json file')
    parser.add_argument('--output-dir', type=str, default='outputs/plots/simulation', help='Output directory for plots')
    args = parser.parse_args()
    
    # Check if the summary.json file exists
    if not os.path.exists(args.summary_file):
        print(f"❌ Error: Summary file not found at {args.summary_file}")
        sys.exit(1)
    
    try:
        plotter = SimulationResultPlotter(args.summary_file)
        
        plotter.plot_token_hit_ratio(args.output_dir)

            
        print(f"✅ Successfully generated plots from: {args.summary_file}")
        print(f"📁 Plots saved to: {args.output_dir}")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        sys.exit(1)
