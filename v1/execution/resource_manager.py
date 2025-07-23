#!/usr/bin/env python3
"""
Resource management for experiments using clean architecture.
Handles GPU allocation and ensures resource cleanup.
"""
from dataclasses import dataclass
from typing import List, Set, Optional, Dict
from config.config import ExperimentConfiguration, RuntimeExperimentContext


class ResourceAllocationError(Exception):
    """Exception raised when resources cannot be allocated"""
    pass


@dataclass(frozen=True)
class GpuAllocation:
    """Immutable GPU allocation information"""
    allocated_gpu_ids: List[int]
    gpu_count: int
    
    @property
    def gpu_device_string(self) -> str:
        """Get comma-separated GPU device string for CUDA_VISIBLE_DEVICES"""
        return ','.join(map(str, self.allocated_gpu_ids))
    
    def __post_init__(self):
        # Ensure consistency
        if len(self.allocated_gpu_ids) != self.gpu_count:
            raise ValueError("GPU count must match length of allocated GPU IDs")


@dataclass(frozen=True)
class PortAllocation:
    """Immutable port allocation information"""
    allocated_port: int


@dataclass
class ResourceAllocation:
    """Complete resource allocation for an experiment"""
    gpu_allocation: GpuAllocation
    port_allocation: PortAllocation
    experiment_config: ExperimentConfiguration
    
    def create_runtime_context(self) -> RuntimeExperimentContext:
        """Create runtime experiment context from this allocation"""
        return RuntimeExperimentContext(
            configuration=self.experiment_config,
            assigned_gpu_devices=self.gpu_allocation.gpu_device_string,
            assigned_server_port=self.port_allocation.allocated_port
        )


class GpuResourceManager:
    """Manages GPU resource allocation and tracking"""
    
    def __init__(self, total_available_gpus: int = 8):
        """Initialize GPU resource manager"""
        if total_available_gpus <= 0:
            raise ValueError("Total available GPUs must be positive")
        
        self.total_available_gpus = total_available_gpus
        self.allocated_gpu_ids: Set[int] = set()
        self.allocation_history: List[GpuAllocation] = []
    
    def get_available_gpu_count(self) -> int:
        """Get number of currently available GPUs"""
        return self.total_available_gpus - len(self.allocated_gpu_ids)
    
    def get_available_gpu_ids(self) -> List[int]:
        """Get list of currently available GPU IDs"""
        all_gpu_ids = set(range(self.total_available_gpus))
        available_ids = all_gpu_ids - self.allocated_gpu_ids
        return sorted(list(available_ids))
    
    def can_allocate_gpus(self, required_gpu_count: int) -> bool:
        """Check if the requested number of GPUs can be allocated"""
        return self.get_available_gpu_count() >= required_gpu_count
    
    def allocate_gpus(self, required_gpu_count: int) -> GpuAllocation:
        """Allocate the requested number of GPUs"""
        if not self.can_allocate_gpus(required_gpu_count):
            raise ResourceAllocationError(
                f"Cannot allocate {required_gpu_count} GPUs. "
                f"Available: {self.get_available_gpu_count()}, "
                f"Total: {self.total_available_gpus}"
            )
        
        available_gpu_ids = self.get_available_gpu_ids()
        allocated_ids = available_gpu_ids[:required_gpu_count]
        
        # Update allocation tracking
        self.allocated_gpu_ids.update(allocated_ids)
        
        allocation = GpuAllocation(
            allocated_gpu_ids=allocated_ids,
            gpu_count=required_gpu_count
        )
        
        self.allocation_history.append(allocation)
        return allocation
    
    def deallocate_gpus(self, allocation: GpuAllocation) -> None:
        """Deallocate the specified GPUs"""
        gpu_ids_to_release = set(allocation.allocated_gpu_ids)
        
        # Verify these GPUs were actually allocated
        if not gpu_ids_to_release.issubset(self.allocated_gpu_ids):
            unallocated_ids = gpu_ids_to_release - self.allocated_gpu_ids
            raise ValueError(f"Attempting to deallocate unallocated GPUs: {unallocated_ids}")
        
        self.allocated_gpu_ids -= gpu_ids_to_release
    
    def reset_allocations(self) -> None:
        """Reset all GPU allocations (for cleanup/testing)"""
        self.allocated_gpu_ids.clear()
        self.allocation_history.clear()
    
    def get_allocation_summary(self) -> Dict[str, int]:
        """Get summary of current GPU allocation state"""
        return {
            'total_gpus': self.total_available_gpus,
            'allocated_gpus': len(self.allocated_gpu_ids),
            'available_gpus': self.get_available_gpu_count(),
            'total_allocations_made': len(self.allocation_history)
        }


class PortResourceManager:
    """Manages port allocation for server processes"""
    
    def __init__(self, base_port: int = 8000, max_concurrent_ports: int = 20):
        """Initialize port resource manager"""
        self.base_port = base_port
        self.max_concurrent_ports = max_concurrent_ports
        self.allocated_ports: Set[int] = set()
        self.next_available_port = base_port
    
    def allocate_port(self) -> PortAllocation:
        """Allocate next available port"""
        if len(self.allocated_ports) >= self.max_concurrent_ports:
            raise ResourceAllocationError(
                f"Maximum concurrent ports ({self.max_concurrent_ports}) exceeded"
            )
        
        # Find next available port
        while self.next_available_port in self.allocated_ports:
            self.next_available_port += 1
        
        allocated_port = self.next_available_port
        self.allocated_ports.add(allocated_port)
        self.next_available_port += 1
        
        return PortAllocation(allocated_port=allocated_port)
    
    def deallocate_port(self, allocation: PortAllocation) -> None:
        """Deallocate the specified port"""
        if allocation.allocated_port not in self.allocated_ports:
            raise ValueError(f"Port {allocation.allocated_port} was not allocated")
        
        self.allocated_ports.remove(allocation.allocated_port)
    
    def reset_allocations(self) -> None:
        """Reset all port allocations"""
        self.allocated_ports.clear()
        self.next_available_port = self.base_port
    
    def get_available_port_count(self) -> int:
        """Get number of available ports"""
        return self.max_concurrent_ports - len(self.allocated_ports)


class ExperimentResourceManager:
    """Combined resource manager for complete experiment resource allocation"""
    
    def __init__(self, 
                 total_gpus: int = 8, 
                 base_port: int = 8000,
                 max_concurrent_experiments: int = 20):
        """Initialize experiment resource manager"""
        self.gpu_manager = GpuResourceManager(total_gpus)
        self.port_manager = PortResourceManager(base_port, max_concurrent_experiments)
        self.active_allocations: Dict[str, ResourceAllocation] = {}
    
    def can_allocate_resources_for_experiment(self, config: ExperimentConfiguration) -> bool:
        """Check if resources can be allocated for the experiment"""
        required_gpus = config.get_total_gpu_requirement()
        return (self.gpu_manager.can_allocate_gpus(required_gpus) and 
                self.port_manager.get_available_port_count() > 0)
    
    def allocate_resources_for_experiment(self, config: ExperimentConfiguration) -> ResourceAllocation:
        """Allocate all required resources for an experiment"""
        experiment_id = config.get_experiment_identifier()
        
        if experiment_id in self.active_allocations:
            raise ResourceAllocationError(f"Resources already allocated for experiment: {experiment_id}")
        
        # Allocate GPUs
        required_gpus = config.get_total_gpu_requirement()
        gpu_allocation = self.gpu_manager.allocate_gpus(required_gpus)
        
        try:
            # Allocate port
            port_allocation = self.port_manager.allocate_port()
            
            # Create combined allocation
            full_allocation = ResourceAllocation(
                gpu_allocation=gpu_allocation,
                port_allocation=port_allocation,
                experiment_config=config
            )
            
            self.active_allocations[experiment_id] = full_allocation
            return full_allocation
            
        except Exception:
            # Rollback GPU allocation if port allocation fails
            self.gpu_manager.deallocate_gpus(gpu_allocation)
            raise
    
    def deallocate_resources_for_experiment(self, allocation: ResourceAllocation) -> None:
        """Deallocate all resources for an experiment"""
        experiment_id = allocation.experiment_config.get_experiment_identifier()
        
        if experiment_id not in self.active_allocations:
            raise ValueError(f"No active allocation found for experiment: {experiment_id}")
        
        # Deallocate resources
        self.gpu_manager.deallocate_gpus(allocation.gpu_allocation)
        self.port_manager.deallocate_port(allocation.port_allocation)
        
        # Remove from active allocations
        del self.active_allocations[experiment_id]
    
    def get_resource_summary(self) -> Dict[str, any]:
        """Get comprehensive resource usage summary"""
        return {
            'gpu_summary': self.gpu_manager.get_allocation_summary(),
            'active_experiments': len(self.active_allocations),
            'available_ports': self.port_manager.get_available_port_count(),
            'allocated_ports': len(self.port_manager.allocated_ports)
        }
    
    def reset_all_allocations(self) -> None:
        """Reset all resource allocations (for cleanup/testing)"""
        self.gpu_manager.reset_allocations()
        self.port_manager.reset_allocations()
        self.active_allocations.clear()


# Default resource manager instance
default_resource_manager = ExperimentResourceManager() 