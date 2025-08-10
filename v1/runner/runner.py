#!/usr/bin/env python3
"""
Experiment runner with clean architecture.
Orchestrates experiment execution and result collection.
"""
import asyncio
import time
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from config.config import ExperimentConfiguration, RuntimeExperimentContext
from environment.environment_provider import EnvironmentProvider
from config.command_builder import CommandFactory
from .resource_manager import ExperimentResourceManager
from .log_analyzer import LogAnalyzer
from util.common import kill_server


@dataclass
class ExperimentResult:
    """Result of a single experiment execution"""
    experiment_identifier: str
    success: bool
    execution_time_seconds: float
    error_message: Optional[str] = None
    server_log_path: Optional[str] = None
    client_log_path: Optional[str] = None


class ExperimentExecutor:
    """Executes individual experiments with proper resource management"""
    
    def __init__(self,
                 environment_provider: EnvironmentProvider,
                 command_factory: CommandFactory,
                 resource_manager: ExperimentResourceManager):
        """Initialize experiment executor with dependencies"""
        self.environment_provider = environment_provider
        self.command_factory = command_factory
        self.resource_manager = resource_manager
        self.log_analyzer = LogAnalyzer()
        self.environment_settings = environment_provider.get_environment_settings()
    
    async def execute_experiment(self, config: ExperimentConfiguration) -> ExperimentResult:
        """Execute a single experiment with proper resource allocation"""
        start_time = time.time()
        experiment_id = config.get_experiment_identifier()
        
        print(f"\n{'='*60}")
        print(f"EXECUTING EXPERIMENT: {experiment_id}")
        print(f"{'='*60}")
        
        # Allocate resources
        try:
            resource_allocation = self.resource_manager.allocate_resources_for_experiment(config)
            runtime_context = resource_allocation.create_runtime_context()
        except Exception as e:
            execution_time = time.time() - start_time
            return ExperimentResult(
                experiment_identifier=experiment_id,
                success=False,
                execution_time_seconds=execution_time,
                error_message=f"Resource allocation failed: {e}"
            )
        
        try:
            # Execute the experiment
            result = await self._run_experiment_with_context(runtime_context)
            result.execution_time_seconds = time.time() - start_time
            
            # If client execution was successful, analyze logs and save metrics
            if result.success and result.client_log_path:
                try:
                    await self._analyze_and_save_metrics(runtime_context, result)
                except Exception as e:
                    print(f"⚠️  Warning: Log analysis failed for {experiment_id}: {e}")
            
            return result
            
        finally:
            # Always deallocate resources
            self.resource_manager.deallocate_resources_for_experiment(resource_allocation)
    
    async def _analyze_and_save_metrics(self, 
                                      context: RuntimeExperimentContext, 
                                      result: ExperimentResult) -> None:
        """Analyze logs and save metrics to JSON"""
        config = context.configuration
        
        print(f"🔍 Analyzing logs for experiment: {result.experiment_identifier}")
        
        # Extract metrics from logs
        metrics = self.log_analyzer.extract_metrics_from_logs(
            config=config,
            client_log_path=result.client_log_path or "",
            server_log_path=result.server_log_path or "",
            execution_success=result.success,
            execution_time=result.execution_time_seconds
        )
        
        # Save metrics to JSON file
        log_directory = config.get_log_directory()
        self.log_analyzer.save_metrics_to_json(metrics, log_directory)
        
        # Print summary to console
        self.log_analyzer.print_metrics_summary(metrics)
    
    async def _run_experiment_with_context(self, context: RuntimeExperimentContext) -> ExperimentResult:
        """Run experiment with allocated resources"""
        config = context.configuration
        experiment_id = config.get_experiment_identifier()
        
        # Build commands
        server_builder = self.command_factory.create_server_command_builder()
        client_builder = self.command_factory.create_client_command_builder()
        
        server_command = server_builder.build_server_command(context)
        client_command = client_builder.build_client_command(context)
        
        # Setup logging
        log_directory = config.get_log_directory()
        os.makedirs(log_directory, exist_ok=True)
        
        server_log_path = f"{log_directory}/server{config.get_experiment_identifier()}.log"
        client_log_path = f"{log_directory}/client{config.get_experiment_identifier()}.log"
        
        print(f"GPU Assignment: {context.assigned_gpu_devices}")
        print(f"Server Port: {context.assigned_server_port}")
        print(f"Server Log: {server_log_path}")
        print(f"Client Log: {client_log_path}")
        
        # Start server process
        server_process = await self._start_server_process(server_command, server_log_path)
        if server_process is None:
            return ExperimentResult(
                experiment_identifier=experiment_id,
                success=False,
                execution_time_seconds=0,
                error_message="Failed to start server process",
                server_log_path=server_log_path
            )
        
        try:
            # Wait for server to be ready
            server_ready = await self._wait_for_server_ready(server_log_path)
            if not server_ready:
                return ExperimentResult(
                    experiment_identifier=experiment_id,
                    success=False,
                    execution_time_seconds=0,
                    error_message="Server failed to become ready",
                    server_log_path=server_log_path
                )
            
            # Execute client
            client_success = await self._run_client_process(client_command, client_log_path)
            
            return ExperimentResult(
                experiment_identifier=experiment_id,
                success=client_success,
                execution_time_seconds=0,  # Will be set by caller
                server_log_path=server_log_path,
                client_log_path=client_log_path
            )
            
        finally:
            # Clean up server process
            await self._cleanup_server_process(server_process)
    
    async def _start_server_process(self, server_command: str, log_path: str):
        """Start server process with logging"""
        try:
            server_log_file = open(log_path, "w")
            server_process = await asyncio.create_subprocess_shell(
                server_command,
                stdout=server_log_file,
                stderr=server_log_file
            )
            server_log_file.write(f"Server command: {server_command}\n")
            server_log_file.flush()
            # Store log file reference for cleanup
            server_process._log_file = server_log_file
            return server_process
        except Exception as e:
            print(f"Failed to start server: {e}")
            return None
    
    async def _wait_for_server_ready(self, log_path: str, timeout_seconds: int = 1200) -> bool:
        """Wait for server to become ready"""
        pattern = self.environment_settings.server_ready_pattern
        
        for i in range(timeout_seconds):
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                        if re.search(pattern, log_content):
                            print(f"Server ready after {i} seconds")
                            return True
                except IOError:
                    # File might be locked, continue waiting
                    pass
            
            await asyncio.sleep(1)
        
        print(f"Server not ready after {timeout_seconds} seconds")
        return False
    
    async def _run_client_process(self, client_command: str, log_path: str) -> bool:
        """Run client process and return success status"""
        try:
            client_log_file = open(log_path, "w")
            client_log_file.write(f"Command: {client_command}\n")
            client_log_file.flush()
            
            client_process = await asyncio.create_subprocess_shell(
                client_command,
                stdout=client_log_file,
                stderr=client_log_file
            )
            
            await client_process.wait()
            client_log_file.close()
            
            return client_process.returncode == 0
            
        except Exception as e:
            print(f"Client process failed: {e}")
            return False
    
    async def _cleanup_server_process(self, server_process):
        """Clean up server process and close log files"""
        try:
            server_process.terminate()
            await server_process.wait()
            if hasattr(server_process, '_log_file'):
                server_process._log_file.close()
        except Exception as e:
            print(f"Error during server cleanup: {e}")


class BatchExperimentRunner:
    """Runs batches of experiments with optimal resource utilization"""
    
    def __init__(self,
                 environment_provider: EnvironmentProvider,
                 resource_manager: ExperimentResourceManager):
        """Initialize batch experiment runner"""
        self.environment_provider = environment_provider
        self.resource_manager = resource_manager
        self.command_factory = CommandFactory(environment_provider)
        self.executor = ExperimentExecutor(
            environment_provider, 
            self.command_factory, 
            resource_manager
        )
    
    def calculate_experiment_batches(self, configs: List[ExperimentConfiguration]) -> List[List[ExperimentConfiguration]]:
        """Calculate optimal batches of experiments based on resource constraints"""
        batches = []
        current_batch = []
        current_gpu_usage = 0
        
        total_gpus = self.resource_manager.gpu_manager.total_available_gpus
        
        for config in configs:
            required_gpus = config.get_total_gpu_requirement()
            
            if current_gpu_usage + required_gpus > total_gpus:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [config]
                current_gpu_usage = required_gpus
            else:
                current_batch.append(config)
                current_gpu_usage += required_gpus
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def run_experiment_batch(self, batch_configs: List[ExperimentConfiguration]) -> List[ExperimentResult]:
        """Run a batch of experiments concurrently"""
        print(f"\nRunning batch of {len(batch_configs)} experiments concurrently")
        
        # Start all experiments concurrently
        tasks = []
        for config in batch_configs:
            task = asyncio.create_task(self.executor.execute_experiment(config))
            tasks.append(task)
        
        # Wait for all experiments to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                config = batch_configs[i]
                final_results.append(ExperimentResult(
                    experiment_identifier=config.get_experiment_identifier(),
                    success=False,
                    execution_time_seconds=0,
                    error_message=f"Exception during execution: {result}"
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def run_all_experiments(self, configs: List[ExperimentConfiguration]) -> List[ExperimentResult]:
        """Run all experiments in optimally sized batches"""
        if not configs:
            return []
        
        print(f"\nRunning {len(configs)} experiments")
        print(f"Available GPUs: {self.resource_manager.gpu_manager.total_available_gpus}")
        
        # Calculate batches
        batches = self.calculate_experiment_batches(configs)
        print(f"Calculated {len(batches)} batches")
        
        for i, batch in enumerate(batches, 1):
            total_gpu_usage = sum(config.get_total_gpu_requirement() for config in batch)
            print(f"  Batch {i}: {len(batch)} experiments, {total_gpu_usage} GPUs")
        
        # Run batches sequentially
        all_results = []
        for batch_id, batch_configs in enumerate(batches, 1):
            # Kill any existing server processes
            kill_server('')

            print(f"\n{'='*80}")
            print(f"EXECUTING BATCH {batch_id}/{len(batches)}")
            print(f"{'='*80}")
            
            batch_results = await self.run_experiment_batch(batch_configs)
            all_results.extend(batch_results)
        
        kill_server('')
        return all_results


def create_experiment_runner(environment_provider: Optional[EnvironmentProvider] = None,
                           resource_manager: Optional[ExperimentResourceManager] = None) -> BatchExperimentRunner:
    """Factory function to create experiment runner with default dependencies"""
    if environment_provider is None:
        from environment.environment_provider import current_environment_provider
        environment_provider = current_environment_provider
    
    if resource_manager is None:
        from .resource_manager import default_resource_manager
        resource_manager = default_resource_manager
    
    return BatchExperimentRunner(environment_provider, resource_manager) 