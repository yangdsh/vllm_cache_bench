#!/usr/bin/env python3
"""
Environment provider with simplified architecture.
Uses only user-based environment detection.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from config.model_registry import Environment, ModelRegistry


@dataclass(frozen=True)
class EnvironmentSettings:
    """Immutable environment-specific settings"""
    environment_type: Environment
    shell_command_prefix: str
    shell_command_suffix: str = ""
    server_ready_pattern: str = r"Capturing CUDA graph shapes: 100%"


class EnvironmentProvider:
    """Provides environment-specific configurations and settings"""
    
    # Environment-specific configurations
    _ENVIRONMENT_SETTINGS: Dict[Environment, EnvironmentSettings] = {
        Environment.DELLA: EnvironmentSettings(
            environment_type=Environment.DELLA,
            shell_command_prefix=(
                "export HF_HOME=/scratch/gpfs/dy5/.cache/huggingface/ && "
                "source /usr/licensed/anaconda3/2024.6/etc/profile.d/conda.sh && "
                "conda activate vllm-cuda121 && "
            ),
            server_ready_pattern=r"Capturing CUDA graph shapes: 100%"
        ),
        
        Environment.FAT2: EnvironmentSettings(
            environment_type=Environment.FAT2,
            shell_command_prefix="",
            server_ready_pattern=r"Capturing CUDA graph shapes: 100%"
        ),
        
        Environment.LMCACHE: EnvironmentSettings(
            environment_type=Environment.LMCACHE,
            shell_command_prefix=(
                "LMCACHE_CHUNK_SIZE=256 "
                "LMCACHE_LOCAL_CPU=True "
                "LMCACHE_SAVE_DECODE_CACHE=true "
            ),
            server_ready_pattern=r"Starting vLLM API server"
        )
    }
    
    def __init__(self, environment: Optional[Environment] = None):
        """Initialize environment provider"""
        if environment is None:
            environment = self._detect_current_environment()
        
        self.current_environment = environment
        self.settings = self._ENVIRONMENT_SETTINGS[environment]
    
    def _detect_current_environment(self) -> Environment:
        """Detect the current environment using username"""
        home_path = os.path.expanduser("~")
        
        if "dy5" in home_path:
            return Environment.DELLA
        elif "dongshengy" in home_path:
            return Environment.FAT2
        elif "ubuntu" in home_path:
            return Environment.LMCACHE
        
        # Default fallback
        return Environment.LMCACHE
    
    def get_environment_settings(self) -> EnvironmentSettings:
        """Get current environment settings"""
        return self.settings
    
    def get_model_path_for_environment(self, model_name: str) -> str:
        """Get environment-specific model path"""
        return ModelRegistry.get_model_path(model_name, self.current_environment)
    
    def is_environment_type(self, environment_type: Environment) -> bool:
        """Check if current environment matches the given type"""
        return self.current_environment == environment_type
    
    def get_available_environments(self) -> List[Environment]:
        """Get list of available environment types"""
        return list(self._ENVIRONMENT_SETTINGS.keys())


# Global environment provider instance
current_environment_provider = EnvironmentProvider() 