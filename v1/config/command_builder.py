#!/usr/bin/env python3
"""
Command builder with clean architecture.
Handles vLLM server command construction with proper separation of concerns.
"""
import json
from .config import ExperimentConfiguration, CacheEvictionStrategy, RuntimeExperimentContext
from .model_registry import ModelSpec, Environment


class ServerCommandBuilder:
    """Builder for vLLM server commands with environment-specific configuration"""
    
    def __init__(self, environment_provider):
        self.environment_provider = environment_provider
        self.environment_settings = environment_provider.get_environment_settings()
    
    def build_server_command(self, 
                           experiment_context: RuntimeExperimentContext) -> str:
        """Build complete server command including environment setup"""
        config = experiment_context.configuration
        model_spec = config.get_model_spec()
        model_path = self.environment_provider.get_model_path_for_environment(config.model_name)
        
        # Build base vLLM arguments
        vllm_arguments = self._build_vllm_arguments(config, model_spec, experiment_context)
        
        # Build environment-specific prefixes and suffixes
        environment_prefix = self._build_environment_prefix(config)
        cuda_device_assignment = f"CUDA_VISIBLE_DEVICES={experiment_context.assigned_gpu_devices}"
        
        # Construct final command
        base_command = (
            "VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO LMCACHE_LOG_LEVEL=INFO PYTHONUNBUFFERED=1 "
            f"vllm serve {model_path} {vllm_arguments}"
        )
        
        full_command = (
            f"{environment_prefix} "
            f"{cuda_device_assignment} "
            f"{base_command} "
            f"{self.environment_settings.shell_command_suffix}"
        ).strip()
        
        return full_command
    
    def _build_vllm_arguments(self, 
                            config: ExperimentConfiguration, 
                            model_spec: ModelSpec,
                            experiment_context: RuntimeExperimentContext) -> str:
        """Build vLLM-specific command arguments"""
        base_args = [
            "--dtype half",
            f"--quantization {model_spec.quantization}" if model_spec.quantization else "",
            "--gpu_memory_utilization 0.8",
            "--disable-log-requests",
            "--uvicorn-log-level warning",
            "--max_num_seqs 128",
            "--num-scheduler-steps 1",
            f"--max-model-len {model_spec.max_model_length}",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            f"--port {experiment_context.assigned_server_port}"
        ]
        
        # Add parallelism arguments for multi-GPU models
        if model_spec.tensor_parallel_size > 1:
            base_args.append(f"--tensor-parallel-size {model_spec.tensor_parallel_size}")
        if model_spec.pipeline_parallel_size > 1:
            base_args.append(f"--pipeline-parallel-size {model_spec.pipeline_parallel_size}")
        
        # Environment-specific configurations
        if self.environment_provider.is_environment_type(Environment.LMCACHE):
            base_args.extend(self._build_lmcache_arguments(config))
        else:
            base_args.extend(self._build_standard_cache_arguments(config))
        
        return " ".join(base_args)
    
    def _build_lmcache_arguments(self, config: ExperimentConfiguration) -> list[str]:
        """Build LMCache-specific arguments"""
        kv_connector_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        args = [
            # todo: p/d disaggregation
            "--max-num-batched-tokens 16384", # this disables chunked prefill
            f"--kv-transfer-config '{json.dumps(kv_connector_config)}'"
        ]
        return args
    
    def _build_standard_cache_arguments(self, config: ExperimentConfiguration) -> list[str]:
        """Build standard vLLM cache arguments"""
        gpu_utilization = config.cache_size_gb / config.gpu_memory_size_gb
        eviction_algorithm_config = {"enable_online_learning": 1}
        
        args = [
            "--eviction_algorithm ml",
            "--max-num-batched-tokens 2048",
            f"--gpu-memory-utilization {gpu_utilization}",
            "--num-gpu-blocks-override 4000",
            f"--eviction_algorithm_config '{json.dumps(eviction_algorithm_config)}'"
        ]
        return args
    
    def _build_environment_prefix(self, config: ExperimentConfiguration) -> str:
        """Build environment-specific prefix including cache configuration"""
        base_prefix = self.environment_settings.shell_command_prefix
        
        if self.environment_provider.is_environment_type(Environment.LMCACHE):
            # Divide cache size by total GPU requirement for multi-GPU setups
            total_gpus = config.get_total_gpu_requirement()
            local_cpu_cache_size = config.cache_size_gb / total_gpus
            cache_prefix = f"LMCACHE_MAX_LOCAL_CPU_SIZE={local_cpu_cache_size}"
            
            # Add conversation eviction configuration
            if config.cache_eviction_strategy == CacheEvictionStrategy.CONVERSATION_AWARE:
                extra_config = {"use_conversation_eviction": True}
                
                # Add conversation eviction config if specified
                if config.conversation_eviction_config is not None:
                    extra_config["conversation_eviction_config"] = config.conversation_eviction_config
                
                cache_prefix += f" LMCACHE_EXTRA_CONFIG='{json.dumps(extra_config)}'"
            
            return f"{base_prefix} {cache_prefix}".strip()
        
        return base_prefix


class ClientCommandBuilder:
    """Builder for benchmark client commands"""
    
    def __init__(self, environment_provider):
        self.environment_provider = environment_provider
    
    def build_client_command(self, experiment_context: RuntimeExperimentContext) -> str:
        """Build complete client command for benchmarking"""
        config = experiment_context.configuration
        client_arguments = self._build_client_arguments(config, experiment_context)
        
        base_command = (
            "LMCACHE_LOG_LEVEL=DEBUG PYTHONUNBUFFERED=1 python client/benchmark_serving.py "
            f"{client_arguments}"
        )
        
        cuda_assignment = f"CUDA_VISIBLE_DEVICES={experiment_context.assigned_gpu_devices}"
        return f"{cuda_assignment} {base_command}"
    
    def _build_client_arguments(self, 
                              config: ExperimentConfiguration,
                              experiment_context: RuntimeExperimentContext) -> str:
        """Build client-specific arguments"""
        model_path = self.environment_provider.get_model_path_for_environment(config.model_name)
        
        # Build argument dictionary
        client_args = {
            'result-dir': config.get_log_directory(),
            'model': model_path,
            'endpoint': '/v1/chat/completions',
            'dataset-name': config.dataset_type.value,
            'host': 'localhost',
            'port': experiment_context.assigned_server_port,
            'result-filename': f'vllm{config.get_experiment_identifier()}.log',
            'num-prompts': config.max_prompt_count,
            'use-oracle': 0,
            'request-rate': config.request_rate_per_second,
            'session-rate': -1,
            'max-active-conversations': -1,
            'checkpoint': 'None',
            'dataset-path': config.dataset_file_path,
            'time-limit': config.time_limit_seconds,
            'save-result': None,
        }
        
        # Add mock decoding if enabled
        if config.enable_mock_decoding:
            client_args['mock-decoding'] = None
        
        # Convert to command line arguments
        arg_strings = []
        for arg_name, arg_value in client_args.items():
            if arg_value is None:
                arg_strings.append(f"--{arg_name}")
            else:
                arg_strings.append(f"--{arg_name} {arg_value}")
        
        return " ".join(arg_strings)


class CommandFactory:
    """Factory for creating command builders"""
    
    def __init__(self, environment_provider):
        self.environment_provider = environment_provider
    
    def create_server_command_builder(self) -> ServerCommandBuilder:
        """Create server command builder"""
        return ServerCommandBuilder(self.environment_provider)
    
    def create_client_command_builder(self) -> ClientCommandBuilder:
        """Create client command builder"""
        return ClientCommandBuilder(self.environment_provider)


# Convenience functions for common usage
def build_server_command(experiment_context: RuntimeExperimentContext,
                        environment_provider) -> str:
    """Convenience function to build server command"""
    builder = ServerCommandBuilder(environment_provider)
    return builder.build_server_command(experiment_context)


def build_client_command(experiment_context: RuntimeExperimentContext,
                        environment_provider) -> str:
    """Convenience function to build client command"""
    builder = ClientCommandBuilder(environment_provider)
    return builder.build_client_command(experiment_context) 