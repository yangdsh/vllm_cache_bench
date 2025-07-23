#!/usr/bin/env python3
"""
Centralized model registry for managing model specifications.
Decouples model management from environment configuration.
"""
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class Environment(Enum):
    """Supported environment types"""
    DELLA = "della"
    FAT2 = "fat2" 
    LMCACHE = "lmcache"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable model specification"""
    name: str
    huggingface_path: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_length: int = 16384
    
    @property
    def total_gpu_requirement(self) -> int:
        """Calculate total GPUs needed for this model"""
        return self.tensor_parallel_size * self.pipeline_parallel_size


class ModelRegistry:
    """Centralized registry for model specifications"""
    
    _models: Dict[str, ModelSpec] = {}
    _environment_model_paths: Dict[Environment, Dict[str, str]] = {}
    
    @classmethod
    def register_model(cls, model_spec: ModelSpec) -> None:
        """Register a model specification"""
        cls._models[model_spec.name] = model_spec
    
    @classmethod
    def register_environment_path(cls, 
                                 environment: Environment, 
                                 model_name: str, 
                                 filesystem_path: str) -> None:
        """Register environment-specific filesystem path for a model"""
        if environment not in cls._environment_model_paths:
            cls._environment_model_paths[environment] = {}
        cls._environment_model_paths[environment][model_name] = filesystem_path
    
    @classmethod
    def get_model_spec(cls, model_name: str) -> ModelSpec:
        """Get model specification by name"""
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available_models}")
        return cls._models[model_name]
    
    @classmethod
    def get_model_path(cls, model_name: str, environment: Environment) -> str:
        """Get model path for specific environment, fallback to HuggingFace path"""
        # Check for environment-specific path first
        env_paths = cls._environment_model_paths.get(environment, {})
        if model_name in env_paths:
            return env_paths[model_name]
        
        # Fallback to HuggingFace path
        model_spec = cls.get_model_spec(model_name)
        return model_spec.huggingface_path
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of available model names"""
        return list(cls._models.keys())
    
    @classmethod
    def initialize_default_models(cls) -> None:
        """Initialize registry with default model specifications"""
        # Small models
        cls.register_model(ModelSpec(
            name="qwen-0.5b",
            huggingface_path="Qwen/Qwen2.5-0.5B-Instruct",
            max_model_length=8192
        ))
        
        # Medium models  
        cls.register_model(ModelSpec(
            name="qwen-8b",
            huggingface_path="Qwen/Qwen3-8B-FP8",
            max_model_length=16384
        ))
        
        cls.register_model(ModelSpec(
            name="llama-8b",
            huggingface_path="meta-llama/Llama-3.1-8B-Instruct",
            max_model_length=16384,
            tensor_parallel_size=1,
        ))

        cls.register_model(ModelSpec(
            name="llama-70b",
            huggingface_path="meta-llama/Llama-3.3-70B-Instruct",
            max_model_length=16384,
            pipeline_parallel_size=4,
        ))
        
        # Large models requiring multiple GPUs
        cls.register_model(ModelSpec(
            name="qwen-32b",
            huggingface_path="Qwen/Qwen3-32B-FP8",
            tensor_parallel_size=2,
            max_model_length=16384
        ))
        
        # Environment-specific paths
        cls.register_environment_path(
            Environment.DELLA, 
            "qwen-8b",
            "/scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/a29cae3df5d16cc895083497dad6ba9530c7d84c"
        )
        
        cls.register_environment_path(
            Environment.DELLA,
            "qwen-32b", 
            "/scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/98a63908b41686889a6ade39c37616e54d49974d"
        )


# Initialize default models on import
ModelRegistry.initialize_default_models() 