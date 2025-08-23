#!/usr/bin/env python3
"""
Experiment configuration with clean architecture and readability.
Decoupled from environment-specific details.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
from .model_registry import ModelRegistry, ModelSpec


class DatasetType(Enum):
    """Supported dataset types"""
    CONVERSATIONAL_CSV = "conversational_csv"





@dataclass(frozen=False)
class ExperimentConfiguration:
    """Immutable experiment configuration with clear naming"""
    
    # Core experiment parameters
    model_name: str = "qwen-8b"
    cache_size_gb: float = 32.0
    request_rate_per_second: float = 1.0
    
    # Dataset configuration
    dataset_type: DatasetType = DatasetType.CONVERSATIONAL_CSV
    dataset_file_path: str = 'data/test_data.csv'
    max_prompt_count: int = 30000
    time_limit_seconds: int = 600
    
    # Infrastructure configuration  
    gpu_memory_size_gb: float = 40.0
    server_port: int = 8000
    
    # Advanced options
    cache_eviction_strategy: str = "lru"
    conversation_eviction_config: Optional[Dict[str, Any]] = None
    enable_mock_decoding: bool = False
    experiment_tag: Optional[str] = None
    
    # LightGBM-specific configuration
    lightgbm_mode: str = "regression"  # Options: "ranking", "regression", "classification"
    
    # Logging configuration
    log_directory: Optional[str] = "outputs/logs"
    
    def get_model_spec(self) -> ModelSpec:
        """Get the model specification for this experiment"""
        return ModelRegistry.get_model_spec(self.model_name)
    
    def get_total_gpu_requirement(self) -> int:
        """Get total number of GPUs required for this experiment"""
        model_spec = self.get_model_spec()
        return model_spec.total_gpu_requirement
    
    def get_log_directory(self) -> str:
        """Get log directory, defaulting to 'logs' if not specified"""
        return self.log_directory

    def get_experiment_identifier(self) -> str:
        """Generate a unique identifier for this experiment configuration"""
        eviction_suffix = f"_{self.cache_eviction_strategy}"
        
        # Add lightgbm_mode to identifier if using lightgbm strategy
        if self.cache_eviction_strategy == "lightgbm" and hasattr(self, 'lightgbm_mode'):
            eviction_suffix = f"_{self.cache_eviction_strategy}_{self.lightgbm_mode}"
        
        return (f"_{self.cache_size_gb}gb_{self.request_rate_per_second}rps{eviction_suffix}"
                f"_{self.dataset_file_path.split('/')[-1].split('.')[0][-10:]}")


class ExperimentConfigurationBuilder:
    """Builder pattern for creating experiment configurations"""
    
    def __init__(self):
        self._config_params: Dict[str, Any] = {}
    
    def with_model(self, model_name: str) -> 'ExperimentConfigurationBuilder':
        """Set the model for the experiment"""
        self._config_params['model_name'] = model_name
        return self
    
    def with_cache_size(self, size_gb: float) -> 'ExperimentConfigurationBuilder':
        """Set cache size in GB"""
        self._config_params['cache_size_gb'] = size_gb
        return self
    
    def with_request_rate(self, rate_per_second: float) -> 'ExperimentConfigurationBuilder':
        """Set request rate per second"""
        self._config_params['request_rate_per_second'] = rate_per_second
        return self
    
    def with_dataset(self, dataset_path: str, dataset_type: DatasetType = DatasetType.CONVERSATIONAL_CSV,
                     ) -> 'ExperimentConfigurationBuilder':
        """Set dataset configuration"""
        self._config_params['dataset_file_path'] = dataset_path
        self._config_params['dataset_type'] = dataset_type
        return self
    
    def with_cache_eviction_strategy(self, strategy: str) -> 'ExperimentConfigurationBuilder':
        """Set cache eviction strategy"""
        # Store the strategy as a string
        self._config_params['cache_eviction_strategy'] = strategy.lower()
        return self
    
    def with_conversation_eviction_config(self, config: Optional[Dict[str, Any]]) -> 'ExperimentConfigurationBuilder':
        """Set conversation eviction configuration"""
        self._config_params['conversation_eviction_config'] = config
        return self
    
    def with_lightgbm_mode(self, mode: str) -> 'ExperimentConfigurationBuilder':
        """Set LightGBM mode (ranking, regression, or classification)"""
        self._config_params['lightgbm_mode'] = mode.lower()
        return self
    
    def with_limits(self, max_prompts: int, time_limit: int) -> 'ExperimentConfigurationBuilder':
        """Set experiment limits"""
        self._config_params['max_prompt_count'] = max_prompts
        self._config_params['time_limit_seconds'] = time_limit
        return self
    
    def with_logging(self, log_directory: str, experiment_tag: str = None) -> 'ExperimentConfigurationBuilder':
        """Set logging configuration"""
        self._config_params['log_directory'] = log_directory
        if experiment_tag:
            self._config_params['experiment_tag'] = experiment_tag
        return self
    
    def enable_mock_decoding(self) -> 'ExperimentConfigurationBuilder':
        """Enable mock decoding for testing"""
        self._config_params['enable_mock_decoding'] = True
        return self
    
    def build(self) -> ExperimentConfiguration:
        """Build the final experiment configuration"""        
        return ExperimentConfiguration(**self._config_params)


@dataclass(frozen=True)
class RuntimeExperimentContext:
    """Runtime context for a running experiment"""
    configuration: ExperimentConfiguration
    assigned_gpu_devices: str  # e.g., "0,1,2,3"
    assigned_server_port: int
    
    @property
    def gpu_device_list(self) -> list[str]:
        """Get list of assigned GPU devices"""
        return self.assigned_gpu_devices.split(',')
    
    @property 
    def gpu_count(self) -> int:
        """Get number of assigned GPUs"""
        return len(self.gpu_device_list)


def create_default_experiment() -> ExperimentConfiguration:
    """Create a default experiment configuration for testing"""
    return ExperimentConfigurationBuilder().build() 