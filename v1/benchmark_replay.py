#!/usr/bin/env python3
#cd ~/PrefixCacheInternProject/vllm_cache_bench/v1/client
"""
Log Replay LMCache Integration - Parses client logs and generates LMCache operations.

This script reads client log files, extracts conversation events (Send/Done), 
and generates corresponding LMCache lookup, retrieve, and store operations.
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoConfig

# First Party - LMCache imports
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig  
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Local imports
from client.conversation_manager import ConversationManager
# Import cache statistics functionality
from client.print_statistics import print_lmcache_statistics

@dataclass
class LogEvent:
    """Represents a parsed log event"""
    event_type: str  # "send" or "done"
    request_id: int
    conversation_id: int
    turn_id: Optional[int] = None
    input_len: Optional[int] = None
    output_len: Optional[int] = None
    waiting_time: Optional[float] = None
    generated_len: Optional[int] = None  # For done events
    timestamp: float = 0.0

class LogParser:
    """Parses client log files to extract conversation events"""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.events: List[LogEvent] = []
    
    def parse_log_file(self) -> List[LogEvent]:
        """Parse the log file and extract Send/Done events"""
        logger.info(f"Parsing log file: {self.log_file_path}")
        
        send_pattern = re.compile(
            r'Send conv_id: (\d+), turn_id: (\d+), waiting_time: ([\d.]+), '
            r'input_len: (\d+), output_len: (\d+)'
        )
        
        done_pattern = re.compile(
            r'Done conv_id: (\d+), output_len: (\d+), turn_id: (\d+)'
        )
        
        timestamp_pattern = re.compile(r'\[([\d-]+) ([\d:,]+)\]')
        
        events = []
        
        with open(self.log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Extract timestamp
                timestamp_match = timestamp_pattern.search(line)
                timestamp = time.time() if not timestamp_match else 0.0
                
                # Try to match Send event
                send_match = send_pattern.search(line)
                if send_match:
                    event = LogEvent(
                        event_type="send",
                        request_id=len(events),
                        conversation_id=int(send_match.group(1)),
                        turn_id=int(send_match.group(2)),
                        waiting_time=float(send_match.group(3)),
                        input_len=int(send_match.group(4)),
                        output_len=int(send_match.group(5)),
                        timestamp=timestamp
                    )
                    events.append(event)
                    continue
                
                # Try to match Done event
                done_match = done_pattern.search(line)
                if done_match:
                    event = LogEvent(
                        event_type="done",
                        request_id=len(events),
                        conversation_id=int(done_match.group(1)),
                        generated_len=int(done_match.group(2)),
                        timestamp=timestamp
                    )
                    events.append(event)
                    continue
        
        logger.info(f"Parsed {len(events)} events from log file")
        self.events = events
        return events

class LMCacheLogReplay:
    """Main class for replaying logs and generating LMCache operations"""
    
    def __init__(self, model_name: str, 
                 chunk_size: int = 256, device: str = None,
                 max_context_length: int = 16384, page_size: int = 16,
                 cache_size: float = 10.0):
        self.model_name = model_name
        self.cache_size = cache_size
        self.chunk_size = chunk_size
        # Force GPU usage - always use CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU usage is required")
        self.device = 'cuda'
        self.max_context_length = max_context_length
        self.page_size = page_size
        
        # LMCache configuration - Load model config dynamically
        self.config = self._load_model_config()
        self.dtype = torch.bfloat16
        self.hidden_dim = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        # Use correct number of key-value heads for GQA (Grouped Query Attention)
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_attention_heads)
            
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        # Correct KV shape format: (num_layers, 2, page_size, num_kv_heads, head_dim)
        self.kv_shape = (self.num_layers, 2, self.page_size, self.num_key_value_heads, self.head_dim)
        
        # Initialize tokenizer and LMCache engine
        self.tokenizer = self._load_tokenizer()
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(self.tokenizer)
        
        # Track conversations whose send events were aborted
        self.aborted_conversations: set[int] = set()
        
        # Statistics tracking
        self.aborted_requests = 0
        self.processed_requests = 0
        self.length_violations = []  # Store details of aborted requests

        # Log expected cache size per token for debugging
        self._log_cache_size_info()
        
        # Initialize LMCache engine
        self._initialize_lmcache()
        
        logger.info(f"LMCacheLogReplay initialized with device: {self.device}")
    
    def _log_cache_size_info(self):
        """Log expected KV cache size per token for debugging"""
        # Calculate expected cache size per token
        bytes_per_element = 2  # Using bfloat16 which is 2 bytes
        
        # KV cache size per layer per token: 2 (K+V) * num_kv_heads * head_dim * bytes_per_element
        cache_size_per_layer_per_token = 2 * self.num_key_value_heads * self.head_dim * bytes_per_element
        
        # Total cache size per token across all layers
        total_cache_size_per_token = cache_size_per_layer_per_token * self.num_layers
        
        logger.info("="*60)
        logger.info("KV CACHE SIZE ANALYSIS")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Data type: {self.dtype} ({bytes_per_element} bytes per element)")
        logger.info(f"Architecture:")
        logger.info(f"  - Layers: {self.num_layers}")
        logger.info(f"  - Key-Value heads: {self.num_key_value_heads}")
        logger.info(f"  - Head dimension: {self.head_dim}")
        logger.info(f"Expected KV cache size per token:")
        logger.info(f"  - Per layer: {cache_size_per_layer_per_token:,} bytes ({cache_size_per_layer_per_token/1024:.2f} KB)")
        logger.info(f"  - Total (all layers): {total_cache_size_per_token:,} bytes ({total_cache_size_per_token/1024:.2f} KB)")
        logger.info("="*60)
    
    def _load_model_config(self):
        """Load the model configuration from Hugging Face Hub"""
        try:
            config = AutoConfig.from_pretrained(self.model_name)
            return config
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            raise

    def _get_kv_cache_for_tokens(self, num_tokens: int) -> List[torch.Tensor]:
        """
        Create fresh KV cache tensors for LMCache operations.
        
        Args:
            num_tokens: Number of tokens to generate KV cache for
            
        Returns:
            List of fresh KV cache tensors, one per layer
        """
        # Get the correct tensor shape for this number of tokens
        tensor_shape = self._get_tensor_shape_for_tokens(num_tokens)
        
        # Create fresh tensors for each layer
        fresh_kv_cache = []
        
        for i in range(self.num_layers):
            # Create fresh tensor on GPU with the correct shape
            kv = torch.zeros(tensor_shape, dtype=self.dtype, device=self.device)
            fresh_kv_cache.append(kv)
        
        # Debug: Log tensor info for the first call
        if not hasattr(self, '_first_tensor_creation'):
            self._first_tensor_creation = True
            total_elements = tensor_shape.numel()
            bytes_per_element = 2  # Using bfloat16 which is 2 bytes
            total_bytes_per_layer = total_elements * bytes_per_element
            total_bytes_all_layers = total_bytes_per_layer * self.num_layers
            
            logger.info(f"KV Cache Tensor Creation Debug:")
            logger.info(f"  - Tensor shape per layer: {tensor_shape}")
            logger.info(f"  - Metadata kv_shape: {self.kv_shape}")
            logger.info(f"  - Elements per layer: {total_elements:,}")
            logger.info(f"  - Bytes per layer: {total_bytes_per_layer:,} ({total_bytes_per_layer/1024:.2f} KB)")
            logger.info(f"  - Total bytes all layers: {total_bytes_all_layers:,} ({total_bytes_all_layers/(1024*1024):.2f} MB)")
            logger.info(f"  - Effective tokens cached: {num_tokens}")
        
        return fresh_kv_cache
    
    def _get_tensor_shape_for_tokens(self, num_tokens: int) -> torch.Size:
        """
        Get the correct tensor shape for KV cache based on number of tokens.
        
        Args:
            num_tokens: Number of tokens to generate KV cache for
            
        Returns:
            Correct tensor shape for the KV cache
        """
        # Calculate number of pages needed for this many tokens
        pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        return torch.Size([2, pages_needed, self.page_size, self.num_key_value_heads, self.head_dim])
    
    def _generate_slot_mapping(self, num_tokens: int) -> torch.Tensor:
        """Generate slot mapping for KV cache with chunk-aware local indices"""
        
        # Generate slot mapping using LOCAL indices within each chunk
        # LMCache internally chunks sequences with chunk_size, so we need
        # to ensure that within each chunk, slot indices start from 0
        slot_mapping = []
        
        for token_idx in range(num_tokens):
            # Calculate the local position within the chunk
            local_token_idx = token_idx % self.chunk_size
            
            # Map to local slot indices within the chunk
            page_idx = local_token_idx // self.page_size
            slot_in_page = local_token_idx % self.page_size
            local_physical_slot = page_idx * self.page_size + slot_in_page
            
            slot_mapping.append(local_physical_slot)
        
        return torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
    
    def _load_tokenizer(self):
        """Load the tokenizer for the model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded tokenizer for {self.model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Fallback to a simple tokenizer
            return None
    
    def _initialize_lmcache(self):
        """Initialize LMCache engine and components"""
        # Create GPU connector WITH GPU buffer for efficient storage
        # This enables reusable GPU buffer and reduces storage overhead
        self.connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size=self.head_dim * self.num_key_value_heads,  # Correct: head_dim * num_kv_heads
            num_layers=self.num_layers,
            use_gpu=True,  # Enable GPU buffer for efficient storage
            chunk_size=self.chunk_size,
            dtype=self.dtype,
            device=self.device,
            page_size=self.page_size
        )
        
        # Log GPU buffer info if created
        if hasattr(self.connector, 'gpu_buffer') and self.connector.gpu_buffer is not None:
            logger.info(f"GPU buffer created: {self.connector.gpu_buffer.shape}")
            logger.info(f"GPU buffer device: {self.connector.gpu_buffer.device}")
            logger.info(f"GPU buffer dtype: {self.connector.gpu_buffer.dtype}")
        else:
            logger.info("No GPU buffer created")

        self.metadata = LMCacheEngineMetadata(
            self.model_name,  # model_name
            1,                # world_size  
            0,                # worker_id
            "vllm",           # fmt
            self.dtype,       # dtype
            self.kv_shape     # kv_shape
        )
        
        # Create config
        self.config = LMCacheEngineConfig.from_defaults(
            chunk_size=self.chunk_size,
            local_cpu=True,              # Enable CPU caching
            max_local_cpu_size=self.cache_size,
            local_disk=None,             # No disk cache
            max_local_disk_size=0,       # No disk cache
            remote_url=None              # No remote cache
        )
        
        # Create engine
        logger.info("Creating LMCacheEngine...")
        self.engine = LMCacheEngineBuilder.get_or_create(
            "log_replay",
            self.config,
            self.metadata,
            self.connector
        )
        
        logger.info("LMCache engine initialized successfully")
    
    def handle_send_event(self, event: LogEvent):
        """Handle Send event - perform lookup and retrieve operations"""
        logger.info(f"Processing Send event: conv_id={event.conversation_id}, "
                   f"turn_id={event.turn_id}, input_len={event.input_len}")
        
        # The 'input_len' from the log event represents the length of the current turn's prompt.
        prompt_len = event.input_len
        
        # Generate prompt text using the calculated prompt length for this turn
        prompt_text = self.conversation_manager.generate_prompt(
            prompt_len, event.conversation_id
        )
        
        # Update conversation history using conversation manager
        self.conversation_manager.add_user_message(event.conversation_id, prompt_text)
        messages = self.conversation_manager.get_all_messages(event.conversation_id)

        # Generate tokens using the calculated prompt length
        tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)

        # Check if total length exceeds limit
        total_length = len(tokens) + event.output_len
        if total_length > self.max_context_length:
            logger.warning(f"⚠️  ABORTING REQUEST: Total length ({total_length}) exceeds {self.max_context_length} tokens. "
                          f"input_len={event.input_len}, output_len={event.output_len}")
            self.aborted_requests += 1
            self.length_violations.append({
                "conversation_id": event.conversation_id,
                "turn_id": event.turn_id,
                "input_len": event.input_len,
                "output_len": event.output_len,
                "total_len": total_length
            })
            self.aborted_conversations.add(event.conversation_id)
            return
        
        # Perform lookup
        start_time = time.perf_counter()
        cached_tokens = self.engine.lookup(tokens, request_id=event.request_id, pin=True)
        lookup_time = time.perf_counter() - start_time
        
        hit_rate = cached_tokens / len(tokens) if len(tokens) > 0 else 0
        logger.info(f"  Lookup: {cached_tokens}/{len(tokens)} tokens cached "
                   f"({hit_rate:.1%} hit rate, {lookup_time:.4f}s)")
        
        # Perform retrieve if we have cached data
        if cached_tokens > 0:
            slot_mapping = self._generate_slot_mapping(len(tokens))
            
            # Create fresh KV cache tensors and update the connector
            kv_caches = self._get_kv_cache_for_tokens(len(tokens))
            self.connector.kvcaches = kv_caches  # Force update the connector's KV caches
            
            start_time = time.perf_counter()
            ret_mask = self.engine.retrieve(
                tokens,
                kvcaches=kv_caches,
                slot_mapping=slot_mapping,
                request_id=event.request_id
            )
            retrieve_time = time.perf_counter() - start_time
            retrieved_tokens = torch.sum(ret_mask).item()
            
            logger.info(f"  ✓ Retrieved {retrieved_tokens} tokens in {retrieve_time:.4f}s")
        else:
            logger.info(f"  ✗ No tokens to retrieve")
        
        self.engine.lookup_unpin([event.request_id])
        
        # Track successful processing
        self.processed_requests += 1
    
    def handle_done_event(self, event: LogEvent):
        """Handle Done event - perform store operation"""
        
        # Skip if the conversation was aborted
        if event.conversation_id in self.aborted_conversations:
            logger.warning(f"Skipping Done event for aborted conversation {event.conversation_id}")
            return

        exisiting_tokens = self.tokenizer.apply_chat_template(
            self.conversation_manager.get_all_messages(event.conversation_id),
            add_generation_prompt=True
        )
        generated_text = self.conversation_manager.generate_prompt(
            event.generated_len, event.conversation_id
        )
        self.conversation_manager.add_gpt_message(event.conversation_id, generated_text)
        
        # Generate tokens for the entire conversation (prompt + response)
        tokens = exisiting_tokens + self.tokenizer.encode(generated_text)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        
        # Perform store operation
        slot_mapping = self._generate_slot_mapping(len(tokens))
        logger.info(f"Processing Done event: conv_id={event.conversation_id}, "
            f"generated_len={event.generated_len}, all_tokens={len(tokens)}")
        
        # Create fresh KV cache tensors and update the connector
        kv_caches = self._get_kv_cache_for_tokens(len(tokens))
        self.connector.kvcaches = kv_caches  # Force update the connector's KV caches
        
        start_time = time.perf_counter()
        self.engine.store(
            tokens=tokens,
            kvcaches=kv_caches,
            slot_mapping=slot_mapping,
            request_id=event.request_id
        )
        store_time = time.perf_counter() - start_time
        
        logger.info(f"  ✓ Stored {len(tokens)} tokens in {store_time:.4f}s")
    
    def replay_log_events(self, events: List[LogEvent]):
        """Replay all log events and generate LMCache operations"""
        logger.info(f"Starting replay of {len(events)} events")
        
        for i, event in enumerate(events):
            logger.info(f"\n--- Event {i+1}/{len(events)} ---")
            
            try:
                if event.event_type == "send":
                    self.handle_send_event(event)
                elif event.event_type == "done":
                    self.handle_done_event(event)
                else:
                    logger.warning(f"Unknown event type: {event.event_type}")
            
            except Exception as e:
                logger.error(f"Error processing event {i+1}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        logger.info(f"\nCompleted replay of {len(events)} events")
    
    def print_statistics(self, api_url: str = None):
        """Print final statistics using conversation manager"""
        logger.info("\n" + "="*60)
        logger.info("LOG REPLAY STATISTICS")
        logger.info("="*60)
        
        stats = self.conversation_manager.get_conversation_stats()
        
        # Basic conversation statistics
        logger.info(f"Total conversations processed: {stats['total_conversations']}")
        logger.info(f"Total messages processed: {stats['total_messages']}")
        logger.info(f"Average messages per conversation: {stats['avg_messages_per_conversation']:.1f}")
        
        # Request processing statistics
        total_send_events = self.processed_requests + self.aborted_requests
        logger.info(f"\nRequest Processing:")
        logger.info(f"  Successfully processed: {self.processed_requests}")
        logger.info(f"  Aborted (length limit): {self.aborted_requests}")
        logger.info(f"  Total Send events: {total_send_events}")
        if total_send_events > 0:
            abort_rate = (self.aborted_requests / total_send_events) * 100
            logger.info(f"  Abort rate: {abort_rate:.1f}%")
        
        # Turn distribution
        if stats['turn_distribution']:
            logger.info("\nTurn distribution:")
            for turns, count in sorted(stats['turn_distribution'].items()):
                logger.info(f"  {turns} turns: {count} conversations")
        
        # Print cache statistics if API URL is provided
        if api_url:
            logger.info("\nCache Statistics:")
            try:
                print_lmcache_statistics(api_url)
            except Exception as e:
                logger.warning(f"Could not fetch cache statistics: {e}")
        else:
            logger.info("\nCache Statistics: Not available (no API URL provided)")
        
        logger.info("="*60)
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        try:
            LMCacheEngineBuilder.destroy("log_replay")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Replay client logs and generate LMCache operations')
    parser.add_argument('--log-file', default='logs/client_32.0gb_1.0rps_test_data.log',
                       help='Path to the client log file')
    parser.add_argument('--cache-size', type=float, default=64,
                       help='LMCache cache size in GB')
    parser.add_argument('--model', default='Qwen/Qwen3-8B-FP8', 
                       help='Model name for tokenizer and metadata')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process')
    parser.add_argument('--chunk-size', type=int, default=256,
                       help='LMCache chunk size (default: 256)')
    parser.add_argument('--max-context-length', type=int, default=16384,
                       help='Maximum expected context length in tokens (default: 16384)')
    parser.add_argument('--page-size', type=int, default=16,
                       help='KV cache page size in tokens (default: 16)')
    parser.add_argument('--api-url', type=str, default="http://localhost:9000/v1/chat/completions",
                       help='API URL for fetching cache statistics (optional)')
    
    args = parser.parse_args()
    
    print("LMCache Log Replay Tool")
    print("="*60)
    print(f"Log file: {args.log_file}")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max context length: {args.max_context_length}")
    print(f"page size: {args.page_size}")
    if args.api_url:
        print(f"API URL: {args.api_url}")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available, but GPU usage is required!")
        return
    else:
        print("INFO: Using GPU (CUDA) for all operations")
    
    # Parse log file
    parser = LogParser(args.log_file)
    events = parser.parse_log_file()
    
    if not events:
        print("No events found in log file!")
        return
    
    # Limit events if specified
    if args.max_events:
        events = events[:args.max_events]
        print(f"Limited to first {len(events)} events")
    
    # Initialize replay system
    replay = LMCacheLogReplay(
        model_name=args.model,
        chunk_size=args.chunk_size,
        max_context_length=args.max_context_length,
        page_size=args.page_size,
        device=None,
        cache_size=args.cache_size
    )
    
    try:
        # Replay events
        replay.replay_log_events(events)
        
        # Print final statistics
        replay.print_statistics(api_url=args.api_url)
        
    except Exception as e:
        logger.error(f"Replay failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        replay.cleanup()
    
    print("\nLMCache Log Replay Completed!")

if __name__ == "__main__":
    main() 