#!/usr/bin/env python3
"""
Main script to run batch experiments with improved architecture.
Demonstrates usage of the experiment runner with various configurations.
"""
import asyncio
import argparse

from config.config import create_default_experiment
from environment.environment_provider import EnvironmentProvider
from runner.resource_manager import ExperimentResourceManager
from runner.runner import create_experiment_runner
from config.config_loader import load_experiments_from_yaml


async def run_yaml_experiments(yaml_file: str, max_gpus: int = 8):
    """Run experiments loaded from YAML configuration"""
    print("="*80)
    print(f"RUNNING EXPERIMENTS FROM YAML: {yaml_file}")
    print("="*80)
    
    try:
        # Load experiments from YAML
        environment_provider = EnvironmentProvider()
        experiments = load_experiments_from_yaml(yaml_file, environment_provider)
        
        if not experiments:
            print("No experiments loaded from YAML file")
            return
        
        # Create experiment runner
        resource_manager = ExperimentResourceManager(total_gpus=max_gpus)
        runner = create_experiment_runner(environment_provider, resource_manager)
        
        # Run experiments
        results = await runner.run_all_experiments(experiments)
        
        # Print results summary
        print("\n" + "="*80)
        print("YAML EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        successful_count = sum(1 for r in results if r.success)
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {len(results) - successful_count}")
        
        if successful_count > 0:
            avg_duration = sum(r.execution_time_seconds for r in results if r.success) / successful_count
            print(f"Average execution time: {avg_duration:.1f}s")
        
    except Exception as e:
        print(f"Error running YAML experiments: {e}")
        raise


async def run_single_experiment_demo():
    """Demonstrate running a single experiment with detailed output"""
    print("="*80)
    print("SINGLE EXPERIMENT DEMONSTRATION")
    print("="*80)
    
    # Create a simple experiment
    experiment = create_default_experiment()
    experiment.model_name = "qwen-8b" # "llama-70b"
    print(f"Running experiment: {experiment.get_experiment_identifier()}")
    print(f"Model: {experiment.model_name}")
    print(f"Cache size: {experiment.cache_size_gb} GB")
    print(f"Request rate: {experiment.request_rate_per_second} req/s")
    
    # Create runner
    runner = create_experiment_runner()
    
    # Run single experiment
    results = await runner.run_all_experiments([experiment])
    
    if results:
        result = results[0]
        print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Duration: {result.execution_time_seconds:.1f}s")
        if result.server_log_path:
            print(f"Server log: {result.server_log_path}")
        if result.client_log_path:
            print(f"Client log: {result.client_log_path}")
        if result.error_message:
            print(f"Error: {result.error_message}")


async def main():
    """Main function with command line argument handling"""
    parser = argparse.ArgumentParser(description='Benchmark runner with clean architecture')
    parser.add_argument('--yaml', type=str, help='Path to YAML configuration file (for yaml mode)')
    parser.add_argument('--gpus', type=int, default=8, help='Number of available GPUs')
    
    args = parser.parse_args()
    
    if args.yaml:
        await run_yaml_experiments(args.yaml, args.gpus)
    else:
        await run_single_experiment_demo()
    
    print("\nBenchmark runner completed!")


if __name__ == "__main__":
    asyncio.run(main()) 