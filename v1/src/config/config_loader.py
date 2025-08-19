#!/usr/bin/env python3
"""
Configuration loader for experiment configurations.
Loads configurations from YAML files and validates them.
"""
import yaml
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from config.config import (
    ExperimentConfiguration, 
    ExperimentConfigurationBuilder, 
    CacheEvictionStrategy, 
    DatasetType
)
from environment.environment_provider import EnvironmentProvider
from config.model_registry import ModelRegistry


class ConfigurationLoadingError(Exception):
    """Exception raised when configuration loading fails"""
    pass


class YAMLExperimentConfigLoader:
    """YAML configuration loader with clean error handling"""
    
    def __init__(self, yaml_file_path: str, environment_provider: Optional[EnvironmentProvider] = None):
        """Initialize the configuration loader"""
        self.yaml_file_path = yaml_file_path
        self.environment_provider = environment_provider or EnvironmentProvider()
        self.yaml_config_data = self._load_and_validate_yaml()
    
    def _load_and_validate_yaml(self) -> Dict[str, Any]:
        """Load and validate YAML configuration file"""
        if not os.path.exists(self.yaml_file_path):
            raise ConfigurationLoadingError(f"YAML file not found: {self.yaml_file_path}")
        
        try:
            with open(self.yaml_file_path, 'r') as file:
                # Handle multi-document YAML files
                yaml_documents = list(yaml.safe_load_all(file))
                
                if len(yaml_documents) == 0:
                    raise ConfigurationLoadingError("YAML file is empty")
                elif len(yaml_documents) == 1:
                    config_data = yaml_documents[0]
                else:
                    print(f"Multi-document YAML detected ({len(yaml_documents)} documents), using first document")
                    config_data = yaml_documents[0]
                    
        except yaml.YAMLError as e:
            raise ConfigurationLoadingError(f"Error parsing YAML file: {e}")
        
        if not isinstance(config_data, dict):
            raise ConfigurationLoadingError("YAML root must be a dictionary")
        
        if 'experiments' not in config_data:
            raise ConfigurationLoadingError("YAML file must contain 'experiments' section")
        
        return config_data
    
    def _create_experiment_log_directory(self) -> str:
        """Create experiment-specific log directory"""
        experiment_tag = self.yaml_config_data.get('tag', 'experiment')
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_base_directory = self.yaml_config_data.get('log_base_dir', 'logs')
        
        log_directory_name = f"{experiment_tag}_{current_date}"
        full_log_path = os.path.join(log_base_directory, log_directory_name)
        
        os.makedirs(full_log_path, exist_ok=True)
        return full_log_path
    
    def _resolve_model_name(self) -> str:
        """Resolve model name from YAML, defaulting to llama-8b if not specified"""
        yaml_model_name = self.yaml_config_data.get('model_name', 'llama-8b')
        
        # Validate model exists in registry
        try:
            ModelRegistry.get_model_spec(yaml_model_name)
            return yaml_model_name
        except ValueError:
            available_models = ModelRegistry.list_available_models()
            raise ConfigurationLoadingError(
                f"Unknown model '{yaml_model_name}' specified in YAML. "
                f"Available models: {available_models}"
            )
    
    def _parse_dataset_configurations(self, 
                                    dataset_config: Dict[str, Any], 
                                    global_defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse dataset configuration and expand into individual experiment configs"""
        individual_configs = []
        
        dataset_path = dataset_config.get('path')
        if not dataset_path:
            raise ConfigurationLoadingError("Dataset configuration must include 'path'")
        
        # Get parameter ranges
        cache_sizes = dataset_config.get('cache_sizes', [64.0])
        request_rates = dataset_config.get('request_rates', [1.0])
        
        # Handle cache eviction strategies
        eviction_strategies = global_defaults.get('use_conversation_evictions', [False])
        if not isinstance(eviction_strategies, list):
            eviction_strategies = [eviction_strategies]
        
        # Get conversation eviction config (can be dataset-specific or global)
        conversation_eviction_config = dataset_config.get('conversation_eviction_config', 
            global_defaults.get('conversation_eviction_config', None))
        
        # Generate all combinations
        for cache_size in cache_sizes:
            for request_rate in request_rates:
                for use_conversation_eviction in eviction_strategies:
                    config_params = {
                        'dataset_file_path': dataset_path,
                        'cache_size_gb': float(cache_size),
                        'request_rate_per_second': float(request_rate),
                        'cache_eviction_strategy': (
                            CacheEvictionStrategy.CONVERSATION_AWARE 
                            if use_conversation_eviction 
                            else CacheEvictionStrategy.STANDARD
                        )
                    }
                    
                    # Add conversation eviction config if using conversation-aware strategy
                    if use_conversation_eviction and conversation_eviction_config is not None:
                        config_params['conversation_eviction_config'] = conversation_eviction_config
                    
                    # Add other dataset-specific parameters
                    for key, value in dataset_config.items():
                        if key not in ['path', 'cache_sizes', 'request_rates', 
                                       'use_conversation_eviction', 'conversation_eviction_config']:
                            # Map old parameter names to new ones
                            if key == 'num_prompts':
                                config_params['max_prompt_count'] = value
                            elif key == 'time_limit':
                                config_params['time_limit_seconds'] = value
                            elif key == 'mock_decoding':
                                config_params['enable_mock_decoding'] = value
                            else:
                                config_params[key] = value
                    
                    individual_configs.append(config_params)
        
        return individual_configs
    
    def load_experiment_configurations(self) -> List[ExperimentConfiguration]:
        """Load and create experiment configurations from YAML"""
        model_name = self._resolve_model_name()
        log_directory = self._create_experiment_log_directory()
        experiment_tag = self.yaml_config_data.get('tag', 'experiment')
        
        print(f"Loading experiments from: {self.yaml_file_path}")
        print(f"Model: {model_name}")
        print(f"Log directory: {log_directory}")
        print(f"Experiment tag: {experiment_tag}")
        
        experiment_configurations = []
        global_defaults = self.yaml_config_data.get('defaults', {})
        
        # Handle both list and single dictionary formats
        experiments_data = self.yaml_config_data['experiments']
        if isinstance(experiments_data, dict):
            experiments_list = [experiments_data]
        elif isinstance(experiments_data, list):
            experiments_list = experiments_data
        else:
            raise ConfigurationLoadingError("'experiments' must be a dictionary or list of dictionaries")
        
        for experiment_data in experiments_list:
            experiment_defaults = {**global_defaults, **experiment_data.get('defaults', {})}
            datasets = experiment_data.get('datasets', [])
            
            if not datasets:
                raise ConfigurationLoadingError("Each experiment must have at least one dataset")
            
            for dataset_config in datasets:
                individual_config_dicts = self._parse_dataset_configurations(dataset_config, experiment_defaults)
                
                for config_dict in individual_config_dicts:
                    try:
                        # Create experiment configuration using builder
                        builder = ExperimentConfigurationBuilder()
                        
                        # Set model
                        builder.with_model(model_name)
                        
                        # Set core parameters
                        builder.with_cache_size(config_dict.get('cache_size_gb', 64.0))
                        builder.with_request_rate(config_dict.get('request_rate_per_second', 1.0))
                        
                        # Set dataset
                        dataset_path = config_dict['dataset_file_path']
                        dataset_type = DatasetType.CONVERSATIONAL_CSV  # Default for now
                        builder.with_dataset(dataset_path, dataset_type)
                        
                        # Set cache eviction strategy
                        eviction_strategy = config_dict.get('cache_eviction_strategy', CacheEvictionStrategy.STANDARD)
                        builder.with_cache_eviction_strategy(eviction_strategy)
                        
                        # Set conversation eviction config if specified
                        if config_dict.get('conversation_eviction_config') is not None:
                            builder.with_conversation_eviction_config(config_dict['conversation_eviction_config'])
                        
                        # Set limits
                        max_prompts = config_dict.get('max_prompt_count', experiment_defaults.get('num_prompts', 30000))
                        time_limit = config_dict.get('time_limit_seconds', experiment_defaults.get('time_limit', 1200))
                        builder.with_limits(max_prompts, time_limit)
                        
                        # Set logging
                        builder.with_logging(log_directory, experiment_tag)
                        
                        # Set mock decoding if specified
                        if config_dict.get('enable_mock_decoding', experiment_defaults.get('mock_decoding', False)):
                            builder.enable_mock_decoding()
                        
                        # Build the configuration
                        experiment_config = builder.build()
                        experiment_configurations.append(experiment_config)
                        
                    except Exception as e:
                        print(f"Warning: Failed to create configuration from {config_dict}: {e}")
                        continue
        
        print(f"Successfully loaded {len(experiment_configurations)} experiment configurations")
        return experiment_configurations


def load_experiments_from_yaml(yaml_file_path: str, 
                             environment_provider: Optional[EnvironmentProvider] = None) -> List[ExperimentConfiguration]:
    """Convenience function to load experiment configurations from YAML"""
    loader = YAMLExperimentConfigLoader(yaml_file_path, environment_provider)
    configs = loader.load_experiment_configurations()
    return configs 