#!/usr/bin/env python3
"""
Log analyzer for experiment results.
Extracts metrics from server and client logs and saves structured results.
"""
import os
import re
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# To support running this script directly from the command line
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import ExperimentConfiguration
from config.config_loader import load_experiments_from_yaml
from environment.environment_provider import EnvironmentProvider


@dataclass
class ExperimentMetrics:
    """Structured experiment metrics extracted from logs"""
    experiment_id: str
    timestamp: str
    configuration: Dict[str, Any]
    benchmark_results: Dict[str, Any]
    cache_statistics: Dict[str, Any]
    conversation_features: Dict[str, Any]
    execution_success: bool
    execution_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class LogAnalyzer:
    """Analyzes experiment logs and extracts structured metrics"""
    
    def __init__(self):
        pass
    
    def extract_metrics_from_logs(self, 
                                 config: ExperimentConfiguration,
                                 client_log_path: str,
                                 server_log_path: str = "",
                                 execution_success: bool = True,
                                 execution_time: float = 0.0) -> ExperimentMetrics:
        """Extract comprehensive metrics from experiment logs"""
        
        experiment_id = config.get_experiment_identifier()
        timestamp = datetime.now().isoformat()
        
        # Extract configuration as dictionary
        config_dict = self._extract_configuration_dict(config)
        
        # Parse client logs for metrics
        benchmark_results = {}
        cache_statistics = {}
        conversation_features = {}
        
        if os.path.exists(client_log_path):
            benchmark_results, cache_statistics, conversation_features = \
                self._parse_client_log(client_log_path)
        
        return ExperimentMetrics(
            experiment_id=experiment_id,
            timestamp=timestamp,
            configuration=config_dict,
            benchmark_results=benchmark_results,
            cache_statistics=cache_statistics,
            conversation_features=conversation_features,
            execution_success=execution_success,
            execution_time_seconds=execution_time
        )
    
    def _extract_configuration_dict(self, config: ExperimentConfiguration) -> Dict[str, Any]:
        """Extract configuration as a clean dictionary"""
        return {
            'model_name': config.model_name,
            'cache_size_gb': config.cache_size_gb,
            'request_rate_per_second': config.request_rate_per_second,
            'dataset_type': config.dataset_type.value,
            'dataset_file_path': config.dataset_file_path,
            'max_prompt_count': config.max_prompt_count,
            'time_limit_seconds': config.time_limit_seconds,
            'gpu_memory_size_gb': config.gpu_memory_size_gb,
            'cache_eviction_strategy': config.cache_eviction_strategy.value,
            'enable_mock_decoding': config.enable_mock_decoding,
            'experiment_tag': config.experiment_tag
        }
    
    def _parse_client_log(self, client_log_path: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Parse client log file and extract metrics"""
        benchmark_results = {}
        cache_statistics = {}
        conversation_features = {}
        
        try:
            with open(client_log_path, 'r') as f:
                content = f.read()
                
                # Extract benchmark results section
                benchmark_results = self._extract_benchmark_results(content)
                
                # Extract client statistics
                cache_statistics, conversation_features = self._extract_client_statistics(content)
                
        except Exception as e:
            print(f"Error parsing client log {client_log_path}: {e}")
        
        return benchmark_results, cache_statistics, conversation_features
    
    def _extract_benchmark_results(self, log_content: str) -> Dict[str, Any]:
        """Extract benchmark results from log content"""
        benchmark_results = {}
        
        # Look for the last benchmark results section
        benchmark_section_pattern = r"============ Serving Benchmark Result ============(.*?)=================================================="
        benchmark_matches = re.findall(benchmark_section_pattern, log_content, re.DOTALL)
        
        if benchmark_matches:
            benchmark_section = benchmark_matches[-1]
            
            # Extract key metrics from benchmark section
            metrics_patterns = {
                'successful_requests': r'Successful requests:\s*(\d+)',
                'benchmark_duration': r'Benchmark duration \(s\):\s*([\d.]+)',
                'total_input_tokens': r'Total input tokens:\s*(\d+)',
                'total_generated_tokens': r'Total generated tokens:\s*(\d+)',
                'request_throughput': r'Request throughput \(req/s\):\s*([\d.]+)',
                'input_throughput': r'Input token throughput \(tok/s\):\s*([\d.]+)',
                'output_throughput': r'Output token throughput \(tok/s\):\s*([\d.]+)',
                'mean_ttft_ms': r'Mean TTFT \(ms\):\s*([\d.]+)',
                'median_ttft_ms': r'Median TTFT \(ms\):\s*([\d.]+)',
                'p99_ttft_ms': r'P99 TTFT \(ms\):\s*([\d.]+)',
                'mean_tpot_ms': r'Mean TPOT \(ms\):\s*([\d.]+)',
                'median_tpot_ms': r'Median TPOT \(ms\):\s*([\d.]+)',
                'p99_tpot_ms': r'P99 TPOT \(ms\):\s*([\d.]+)',
                'mean_itl_ms': r'Mean ITL \(ms\):\s*([\d.]+)',
                'median_itl_ms': r'Median ITL \(ms\):\s*([\d.]+)',
                'p99_itl_ms': r'P99 ITL \(ms\):\s*([\d.]+)'
            }
            
            for metric_name, pattern in metrics_patterns.items():
                match = re.search(pattern, benchmark_section)
                if match:
                    try:
                        value = float(match.group(1))
                        if metric_name in ['successful_requests', 'total_input_tokens', 'total_generated_tokens']:
                            benchmark_results[metric_name] = int(value)
                        else:
                            benchmark_results[metric_name] = value
                    except ValueError:
                        benchmark_results[metric_name] = match.group(1)
        
        return benchmark_results
    
    def _extract_client_statistics(self, log_content: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract client statistics from log content"""
        cache_statistics = {}
        conversation_features = {}
        
        # Look for the last client statistics section
        stats_section_pattern = r"CLIENT_STATISTICS_BEGIN(.*?)CLIENT_STATISTICS_END"
        stats_matches = re.findall(stats_section_pattern, log_content, re.DOTALL)
        
        if stats_matches:
            stats_section = stats_matches[-1]
            
            # Parse key-value pairs from statistics section
            for line in stats_section.split('\n'):
                line = line.strip()
                if ':' in line:
                    try:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Categorize statistics
                        if key.startswith('conversation_'):
                            conversation_features[key] = value
                        elif key.endswith('_rate'):
                            try:
                                cache_statistics[key] = float(value)
                            except ValueError:
                                cache_statistics[key] = value
                        else:
                            try:
                                cache_statistics[key] = int(value)
                            except ValueError:
                                try:
                                    cache_statistics[key] = float(value)
                                except ValueError:
                                    cache_statistics[key] = value
                    except ValueError:
                        continue
        
        return cache_statistics, conversation_features
    
    def save_metrics_to_json(self, metrics: ExperimentMetrics, log_directory: str) -> None:
        """Save metrics to summary.json file in append mode"""
        summary_file = os.path.join(log_directory, "summary.json")
        
        # Ensure log directory exists
        os.makedirs(log_directory, exist_ok=True)
        
        # Convert metrics to dictionary
        metrics_dict = metrics.to_dict()
        
        # Append to JSON file
        try:
            # Check if file exists and has content
            if os.path.exists(summary_file) and os.path.getsize(summary_file) > 0:
                # Read existing data
                with open(summary_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []
            
            # Append new metrics
            existing_data.append(metrics_dict)
            
            # Write updated data
            with open(summary_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            print(f"✅ Metrics saved to {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving metrics to {summary_file}: {e}")
    
    def print_metrics_summary(self, metrics: ExperimentMetrics) -> None:
        """Print a formatted summary of the metrics"""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT ANALYSIS: {metrics.experiment_id}")
        print(f"{'='*60}")
        
        print(f"📊 BENCHMARK RESULTS:")
        if metrics.benchmark_results:
            for key, value in metrics.benchmark_results.items():
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, float):
                    print(f"    {formatted_key}: {value:.2f}")
                else:
                    print(f"    {formatted_key}: {value}")
        else:
            print(f"    No benchmark results found")
        
        print(f"\n🔄 CACHE STATISTICS:")
        if metrics.cache_statistics:
            # Separate vLLM and LMCache stats
            vllm_stats = {k: v for k, v in metrics.cache_statistics.items() if k.startswith('vllm_')}
            lmcache_stats = {k: v for k, v in metrics.cache_statistics.items() if k.startswith('lmcache_')}
            
            if vllm_stats:
                print(f"    vLLM Local Prefix Cache:")
                for key, value in vllm_stats.items():
                    clean_key = key.replace('vllm_', '').replace('_', ' ').title()
                    if key.endswith('_rate'):
                        print(f"      {clean_key}: {float(value)*100:.1f}%")
                    else:
                        print(f"      {clean_key}: {value}")
            
            if lmcache_stats:
                print(f"    LMCache External Cache:")
                for key, value in lmcache_stats.items():
                    clean_key = key.replace('lmcache_', '').replace('_', ' ').title()
                    if key.endswith('_rate'):
                        print(f"      {clean_key}: {float(value)*100:.1f}%")
                    else:
                        print(f"      {clean_key}: {value}")
        else:
            print(f"    No cache statistics found")
        
        print(f"\n💬 CONVERSATION FEATURES:")
        if metrics.conversation_features:
            for key, value in metrics.conversation_features.items():
                formatted_key = key.replace('conversation_', '').replace('_', ' ').title()
                print(f"    {formatted_key}: {value}")
        else:
            print(f"    No conversation features found")
        
        print(f"\n⏱️  EXECUTION INFO:")
        print(f"    Success: {'✅' if metrics.execution_success else '❌'}")
        print(f"    Duration: {metrics.execution_time_seconds:.2f} seconds")
        print()


def main():
    """Main function to run log analysis from a YAML file."""
    parser = argparse.ArgumentParser(description='Run log analysis on experiment logs.')
    parser.add_argument('--yaml', type=str, required=True, help='Path to the experiment YAML file.')
    args = parser.parse_args()

    print(f"Running log analysis for experiments in {args.yaml}")

    try:
        environment_provider = EnvironmentProvider()
        experiments = load_experiments_from_yaml(args.yaml, environment_provider)
        analyzer = LogAnalyzer()

        for experiment in experiments:
            log_directory = experiment.get_log_directory()
            client_log_path = os.path.join(log_directory, f"client{experiment.get_experiment_identifier()}.log")
            server_log_path = os.path.join(log_directory, f"server{experiment.get_experiment_identifier()}.log")

            if not os.path.exists(client_log_path):
                print(f"Could not find client log for {experiment.get_experiment_identifier()} at {client_log_path}")
                continue

            metrics = analyzer.extract_metrics_from_logs(
                config=experiment,
                client_log_path=client_log_path,
                server_log_path=server_log_path,
                execution_success=True,  # Assume success for existing logs
                execution_time=0.0  # Not available from logs alone
            )
            analyzer.print_metrics_summary(metrics)
            analyzer.save_metrics_to_json(metrics, log_directory)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import argparse
    main()
