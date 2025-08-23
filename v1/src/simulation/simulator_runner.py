#!/usr/bin/env python3
"""
Simulator Runner for Cache Analysis

This script provides a high-level interface to run cache simulations
based on YAML configuration files and output results in JSON format.

Supported eviction policies:
- lru: Least Recently Used
- standard: Maps to LRU
- conversation_aware: Conversation-aware eviction
- lightgbm: Machine learning-based eviction using LightGBM
- oracle: Oracle policy for baseline comparison

For LightGBM policy, additional configuration options:
- lightgbm_mode: "ranking", "regression", or "classification" (default: "regression")
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config_loader import load_experiments_from_yaml, ConfigurationLoadingError
from environment.environment_provider import EnvironmentProvider
from simulation.cache_simulator import CacheSimulatorReplay, SimulatorConfig
from simulation.common import LogParser
from simulation.lightgbm_scorer import get_model_path_from_input_file, train_model_from_events

# Import the simulation plotter
try:
    from utils.simulation_plotter import SimulationResultPlotter
    PLOTTER_AVAILABLE = True
except ImportError:
    PLOTTER_AVAILABLE = False
    print("Warning: Simulation plotter not available. Plots will not be generated.")


@dataclass
class SimulatorResult:
    """Result of a single simulation run"""
    experiment_id: str
    success: bool
    execution_time_seconds: float
    cache_hit_rate: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    total_cache_size_gb: float
    eviction_policy: str
    input_file: str
    error_message: Optional[str] = None
    detailed_metrics: Optional[Dict[str, Any]] = None


@dataclass
class SimulationConfig:
    """Configuration for the simulator runner"""
    yaml_file: str
    output_dir: str = "/home/ubuntu/PrefixCacheInternProject/InferenceLab/outputs/simulation"
    max_events: Optional[int] = None
    train_max_events: Optional[int] = None
    model_dir: str = "models"
    force_retrain: bool = True
    verbose: bool = False
    no_plots: bool = False


class SimulatorRunner:
    """Runs cache simulations based on YAML configurations"""
    
    def __init__(self, config: SimulationConfig):
        """Initialize the simulator runner"""
        self.config = config
        self.environment_provider = EnvironmentProvider()
        self.results: List[SimulatorResult] = []
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_simulator_config(self, experiment_config: Any) -> SimulatorConfig:
        """Create simulator configuration from experiment configuration"""
        # Extract relevant parameters from experiment config
        cache_size_gb = getattr(experiment_config, 'cache_size_gb', 10.0)
        eviction_policy = getattr(experiment_config, 'cache_eviction_strategy', 'conversation_aware')
        
        # Map experiment eviction strategy to simulator policy
        policy_str = str(eviction_policy).lower()
        
        # Policy mapping from experiment config to simulator policy
        policy_mapping = {
            'lru': 'lru',
            'standard': 'lru',  # Map STANDARD to LRU for simulator
            'conversation_aware': 'conversation_aware',
            'lightgbm': 'lightgbm',  # LightGBM-based machine learning eviction
            'oracle': 'oracle'  # Oracle policy for baseline comparison
        }
        simulator_policy = policy_mapping.get(policy_str, policy_str)
        
        # Validate supported policies
        supported_policies = list(policy_mapping.values())
        if simulator_policy not in supported_policies:
            print(f"Warning: Unknown eviction policy '{simulator_policy}', using as-is")
            print(f"Supported policies: {supported_policies}")
        
        # Create simulator config
        simulator_config = SimulatorConfig(
            cache_size_gb=cache_size_gb,
            chunk_size=256,
            block_size=256,
            max_context_length=16384,
            eviction_policy=simulator_policy,
            decay_reference=120.0,
            bytes_per_token=144 * 1024,
            enable_detailed_analysis=True,
            chat_template_overhead=0
        )
        
        return simulator_config
    
    def _get_input_file_from_experiment(self, experiment_config: Any) -> str:
        """Extract input file path from experiment configuration"""
        if hasattr(experiment_config, 'dataset_file_path'):
            value = getattr(experiment_config, 'dataset_file_path')
            if value and os.path.exists(value):
                return value
        
        raise ValueError(f"No valid input file found")
    
    def _run_single_simulation(self, experiment_config: Any) -> SimulatorResult:
        """Run a single simulation for an experiment configuration"""
        start_time = time.time()
        experiment_id = getattr(experiment_config, 'get_experiment_identifier', lambda: 'unknown')()
        
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATION: {experiment_id}")
        print(f"{'='*60}")
        
        try:
            # Get input file
            input_file = self._get_input_file_from_experiment(experiment_config)
            print(f"Input file: {input_file}")
            
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            # Create simulator configuration
            simulator_config = self._create_simulator_config(experiment_config)
            print(f"Cache size: {simulator_config.cache_size_gb} GB")
            print(f"Eviction policy: {simulator_config.eviction_policy}")
            
            # Parse log file
            parser = LogParser(input_file)
            events = parser.parse_log_file(mode='server')
            
            if not events:
                raise ValueError("No events found in log file")
            
            # Limit events if specified
            if self.config.max_events:
                events = events[:self.config.max_events]
                print(f"Limited to first {len(events)} events")
                
            if simulator_config.eviction_policy == 'lightgbm':
                model_path = get_model_path_from_input_file(input_file, self.config.model_dir)
                needs_training = self.config.force_retrain or not os.path.exists(model_path)
                
                if needs_training:
                    print("Training LightGBM model...")
                    if self.config.train_max_events:
                        train_events = events[:self.config.train_max_events] 
                    else:
                        train_events = events
                    
                    # Get lightgbm configuration from experiment config if available
                    lightgbm_mode = getattr(experiment_config, 'lightgbm_mode', 'regression')
                    print(f"Using LightGBM mode: {lightgbm_mode}")
                    
                    try:
                        trained_scorer = train_model_from_events(train_events, mode=lightgbm_mode)
                        trained_scorer.save_model(model_path)
                        print(f"Model saved to {model_path}")
                    except Exception as e:
                        print(f"Error training LightGBM model: {e}")
                        raise
                
                simulator_config.model_path = model_path
            
            # Initialize and run simulator
            simulator = CacheSimulatorReplay(simulator_config)
            
            # Use same events for training and simulation if LightGBM
            simulation_events = events
            if simulator_config.eviction_policy == 'lightgbm' and self.config.train_max_events:
                simulation_events = events[:self.config.train_max_events]
            
            simulator.replay_log_events(simulation_events)
            
            # Extract results
            stats = simulator.simulator.get_detailed_statistics()
            execution_time = time.time() - start_time
            
            # Extract basic stats from the detailed statistics
            basic_stats = stats.get('basic_stats', {})
            
            result = SimulatorResult(
                experiment_id=experiment_id,
                success=True,
                execution_time_seconds=execution_time,
                cache_hit_rate=basic_stats.get('hit_ratio', 0.0),
                total_requests=basic_stats.get('total_requests', 0),
                cache_hits=basic_stats.get('cache_hits', 0),
                cache_misses=basic_stats.get('cache_misses', 0),
                total_cache_size_gb=simulator_config.cache_size_gb,
                eviction_policy=simulator_config.eviction_policy,
                input_file=input_file,
                detailed_metrics=stats
            )
            
            print(f"✅ Simulation completed successfully")
            print(f"   Hit rate: {result.cache_hit_rate:.2%}")
            print(f"   Total requests: {result.total_requests}")
            print(f"   Execution time: {execution_time:.1f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Simulation failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return SimulatorResult(
                experiment_id=experiment_id,
                success=False,
                execution_time_seconds=execution_time,
                cache_hit_rate=0.0,
                total_requests=0,
                cache_hits=0,
                cache_misses=0,
                total_cache_size_gb=0.0,
                eviction_policy='unknown',
                input_file='',
                error_message=error_msg
            )
    
    def run_simulations(self) -> List[SimulatorResult]:
        """Run simulations for all experiments in the YAML file"""
        print(f"\n{'='*80}")
        print(f"SIMULATOR RUNNER")
        print(f"YAML file: {self.config.yaml_file}")
        print(f"Output directory: {self.config.output_dir}")
        print(f"{'='*80}")
        
        try:
            # Load experiments from YAML
            experiments = load_experiments_from_yaml(self.config.yaml_file, self.environment_provider)
            
            if not experiments:
                print("No experiments found in YAML file")
                return []
            
            print(f"Found {len(experiments)} experiments to simulate")
            
            # Show experiment details
            for i, experiment in enumerate(experiments, 1):
                experiment_id = getattr(experiment, 'get_experiment_identifier', lambda: f'experiment_{i}')()
                cache_size = getattr(experiment, 'cache_size_gb', 'unknown')
                strategy = getattr(experiment, 'cache_eviction_strategy', 'unknown')
                print(f"  {i}. {experiment_id} (cache: {cache_size}GB, strategy: {strategy})")
            
            # Run simulations
            for i, experiment in enumerate(experiments, 1):
                print(f"\nProgress: {i}/{len(experiments)}")
                result = self._run_single_simulation(experiment)
                self.results.append(result)
            
            # Save results
            self._save_results()
            
            return self.results
            
        except ConfigurationLoadingError as e:
            print(f"Configuration error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    def _save_results(self):
        """Save simulation results to a single JSON file in summary format"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save results in summary.json format
        name = self.config.yaml_file.split("/")[-1].split(".")[0]
        summary_file = os.path.join(self.config.output_dir, f"summary_{name}.json")
        
        # Convert results to summary format
        summary_data = []
        for result in self.results:
            if result.success:
                # Extract configuration from experiment
                experiment_config = self._extract_experiment_config(result.experiment_id)
                
                # Extract basic stats for summary
                basic_stats = result.detailed_metrics.get('basic_stats', {})
                
                summary_entry = {
                    "experiment_id": result.experiment_id,
                    "timestamp": datetime.now().isoformat(),
                    "configuration": experiment_config,
                    "basic_stats": basic_stats,
                    "execution_success": result.success,
                    "execution_time_seconds": result.execution_time_seconds
                }
                
                if result.error_message:
                    summary_entry["error_message"] = result.error_message
                
                summary_data.append(summary_entry)
        
        # Write to summary file
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"\nSimulation results saved to: {summary_file}")
        
        # Also save a detailed results file with all data
        detailed_file = os.path.join(self.config.output_dir, "detailed_results.json")
        detailed_data = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.results],
            'summary': self._generate_summary()
        }
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        print(f"Detailed results saved to: {detailed_file}")
        
        self._generate_plots(summary_file)
    
    def _generate_plots(self, summary_file: str):
        """Generate plots from the simulation results"""
        try:
            # Create plots directory
            plots_dir = "outputs/plots/simulation"
            os.makedirs(plots_dir, exist_ok=True)
            
            print(f"\nGenerating plots from: {summary_file}")
            
            # Create plotter and generate plots
            plotter = SimulationResultPlotter(summary_file)
            plotter.plot_token_hit_ratio(plots_dir)
            
            print(f"✅ Plots generated successfully in: {plots_dir}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to generate plots: {e}")
            print("Simulation results saved, but plots could not be generated.")
    
    def _extract_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """Extract configuration from experiment ID and YAML data"""
        try:
            experiments = load_experiments_from_yaml(self.config.yaml_file, self.environment_provider)
            
            # Find matching experiment
            for experiment in experiments:
                if hasattr(experiment, 'get_experiment_identifier'):
                    if experiment.get_experiment_identifier() == experiment_id:
                        config_dict = {}
                        for attr in dir(experiment):
                            if not attr.startswith('_') and not callable(getattr(experiment, attr)):
                                value = getattr(experiment, attr)
                                if value is not None:
                                    config_dict[attr] = str(value) if hasattr(value, '__str__') else value
                        return config_dict
            
            # If not found, return basic config
            return {
                "experiment_id": experiment_id,
                "cache_eviction_strategy": "unknown",
                "cache_size_gb": 0.0
            }
            
        except Exception as e:
            return {
                "experiment_id": experiment_id,
                "error": f"Failed to extract configuration: {e}",
                "cache_eviction_strategy": "unknown",
                "cache_size_gb": 0.0
            }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        
        summary = {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(self.results) - len(successful_results),
            'success_rate': len(successful_results) / len(self.results) if self.results else 0.0
        }
        
        return summary
    
    def print_summary(self):
        """Print summary of simulation results"""
        if not self.results:
            print("No results to summarize")
            return
        
        summary = self._generate_summary()
        
        print(f"\n{'='*80}")
        print(f"SIMULATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful_experiments']}")
        print(f"Failed: {summary['failed_experiments']}")
        print(f"Success rate: {summary['success_rate']:.1%}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Cache Simulator Runner')
    parser.add_argument('--yaml', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--output-dir', type=str, 
        default='/home/ubuntu/PrefixCacheInternProject/InferenceLab/outputs/simulation', help='Output directory for results')
    parser.add_argument('--max-events', type=int, help='Maximum number of events to process')
    parser.add_argument('--train-max-events', type=int, help='Maximum events for LightGBM training')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory for LightGBM models')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SimulationConfig(
        yaml_file=args.yaml,
        output_dir=args.output_dir,
        max_events=args.max_events,
        train_max_events=args.train_max_events,
        model_dir=args.model_dir,
        force_retrain=True,
        verbose=args.verbose,
    )
    
    # Run simulations
    runner = SimulatorRunner(config)
    results = runner.run_simulations()
    
    # Print summary
    runner.print_summary()
    
    print(f"\nSimulation runner completed!")


if __name__ == "__main__":
    main()
