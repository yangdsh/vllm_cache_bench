#!/usr/bin/env python3
"""
Parses client or server logs and generates LMCache operations.

This script reads log files, extracts events, and generates LMCache lookup, retrieve, and store.
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer
from collections import defaultdict

# First Party - LMCache imports
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig  
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Remove any existing handlers
logger.handlers.clear()

# Add handler with no formatting
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.propagate = False

# Local imports
from utils.conversation_manager import ConversationManager
from common import get_bytes_per_token, get_chat_template_overhead, LogParser, Event
from utils.print_statistics import print_lmcache_statistics

class LMCacheLogReplay:
    """Unified class for replaying client or server logs and generating LMCache operations"""
    def __init__(self, model_name: str, chunk_size: int = 256, device: str = None,
                 max_context_length: int = 16384, page_size: int = 16,
                 cache_size: float = 64.0, eviction_policy: str = "lru", mode: str = "server",
                 enable_timing_sync: bool = False):
        self.model_name = model_name
        self.cache_size = cache_size
        self.chunk_size = chunk_size
        self.mode = mode
        self.enable_timing_sync = enable_timing_sync
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU usage is required")
        self.device = 'cuda'
        self.max_context_length = max_context_length
        self.page_size = page_size
        # "conversation_aware" would require timing sync, which is disabled by default
        supported_algorithms = ["lru"] #, "conversation_aware"
        if eviction_policy not in supported_algorithms:
            logger.warning(
                f"Eviction policy '{eviction_policy}' is not supported. "
                f"Supported: {supported_algorithms}. Using 'lru' instead."
                "conversation_aware requires timing sync, which is disabled by default"
            )
            eviction_policy = "lru"
        self.eviction_policy = eviction_policy
        self.config, self.bytes_per_token = get_bytes_per_token(self.model_name)
        
        # Calculate chat template overhead
        self.chat_template_overhead = get_chat_template_overhead(self.model_name)
        logger.info(f"Chat template overhead {self.model_name}: {self.chat_template_overhead}")
        
        self.dtype = torch.bfloat16
        self.hidden_dim = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.config, 'num_key_value_heads', self.num_attention_heads)
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.kv_shape = (
            self.num_layers, 2, self.page_size, self.num_key_value_heads, self.head_dim)
        self.tokenizer = self._load_tokenizer()
        self.conversation_manager = ConversationManager(self.tokenizer)
        self.aborted_conversations: set[int] = set()
        self.aborted_requests = 0
        self.processed_requests = 0
        self.length_violations = []
        self.total_lookup_time = 0.0
        self.total_retrieve_time = 0.0
        self.total_store_time = 0.0
        self.total_running_time = 0.0
        self.lookup_count = 0
        self.retrieve_count = 0
        self.store_count = 0
        self.total_hit_tokens = 0
        self.total_tokens = 0
        self.requests_with_hits = 0
        
        # Add turn-based hit statistics tracking
        self.conversation_turns: Dict[int, int] = defaultdict(int)
        self.hits_by_turns: Dict[int, int] = defaultdict(int)
        self.requests_by_turns: Dict[int, int] = defaultdict(int)
        self.lookup_tokens_by_turns: Dict[int, int] = defaultdict(int)
        self.hit_tokens_by_turns: Dict[int, int] = defaultdict(int)
        
        self._initialize_lmcache()
        logger.info(f"device: {self.device}, mode: {self.mode}, sync: {self.enable_timing_sync}")

    def _load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded tokenizer for {self.model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return None

    def _initialize_lmcache(self):
        self.connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size=self.head_dim * self.num_key_value_heads,
            num_layers=self.num_layers,
            use_gpu=True,
            chunk_size=self.chunk_size,
            dtype=self.dtype,
            device=self.device,
            page_size=self.page_size
        )
        if hasattr(self.connector, 'gpu_buffer') and self.connector.gpu_buffer is not None:
            logger.info(f"GPU buffer created: {self.connector.gpu_buffer.shape}")
            logger.info(f"GPU buffer device: {self.connector.gpu_buffer.device}")
            logger.info(f"GPU buffer dtype: {self.connector.gpu_buffer.dtype}")
        else:
            logger.info("No GPU buffer created")
        self.metadata = LMCacheEngineMetadata(
            self.model_name, 1, 0, "vllm", self.dtype, self.kv_shape
        )
        extra_config = {}
        if self.eviction_policy == "lru":
            logger.info("Using LRU eviction policy")
        elif self.eviction_policy == "conversation_aware":
            extra_config["use_conversation_eviction"] = True
            logger.info("Using conversation-aware eviction policy")
        logger.info(f"Extra config: {extra_config}")
        self.config = LMCacheEngineConfig.from_defaults(
            chunk_size=self.chunk_size,
            local_cpu=True,
            max_local_cpu_size=self.cache_size,
            local_disk=None,
            max_local_disk_size=0,
            remote_url=None,
            extra_config=extra_config
        )
        logger.info("Creating LMCacheEngine...")
        self.engine = LMCacheEngineBuilder.get_or_create(
            f"log_replay_{self.mode}",
            self.config,
            self.metadata,
            self.connector
        )
        logger.info("LMCache engine initialized successfully")

    def _get_kv_cache_for_tokens(self, num_tokens: int) -> List[torch.Tensor]:
        tensor_shape = self._get_tensor_shape_for_tokens(num_tokens)
        fresh_kv_cache = []
        for i in range(self.num_layers):
            kv = torch.zeros(tensor_shape, dtype=self.dtype, device=self.device)
            fresh_kv_cache.append(kv)
        if not hasattr(self, '_first_tensor_creation'):
            self._first_tensor_creation = True
            total_elements = tensor_shape.numel()
            bytes_per_element = 2
            total_bytes_per_layer = total_elements * bytes_per_element
            total_bytes_all_layers = total_bytes_per_layer * self.num_layers
            logger.info(f"KV Cache Tensor Creation Debug:")
            logger.info(f"  - Tensor shape per layer: {tensor_shape}")
            logger.info(f"  - Metadata kv_shape: {self.kv_shape}")
            logger.info(f"  - Elements per layer: {total_elements:,}")
            logger.info(f"  - Bytes per layer: {total_bytes_per_layer/1024:.2f} KB")
            logger.info(f"  - Total bytes all layers: {total_bytes_all_layers/(1024*1024):.2f} MB")
            logger.info(f"  - Effective tokens cached: {num_tokens}")
        return fresh_kv_cache

    def _get_tensor_shape_for_tokens(self, num_tokens: int) -> torch.Size:
        pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        return torch.Size([2, pages_needed, self.page_size, self.num_key_value_heads, self.head_dim])

    def _generate_slot_mapping(self, num_tokens: int) -> torch.Tensor:
        slot_mapping = []
        for token_idx in range(num_tokens):
            local_token_idx = token_idx % self.chunk_size
            page_idx = local_token_idx // self.page_size
            slot_in_page = local_token_idx % self.page_size
            local_physical_slot = page_idx * self.page_size + slot_in_page
            slot_mapping.append(local_physical_slot)
        return torch.tensor(slot_mapping, dtype=torch.long, device=self.device)

    def handle_send_event(self, event: Event):
        logger.info(f"Processing Send event: conv_id={event.conversation_id}, "
                    f"turn_number={event.turn_number}, input_tokens={event.input_tokens}")
        
        # Update conversation turn state
        self.conversation_turns[event.conversation_id] = max(
            self.conversation_turns[event.conversation_id], event.turn_number
        )
        
        # Track requests by turn count
        turn_count = self.conversation_turns[event.conversation_id]
        self.requests_by_turns[turn_count] += 1
        
        if self.mode == 'client':
            prompt_len = event.input_tokens
        else:
            event.input_tokens -= self.chat_template_overhead
            context_len = self.conversation_manager.get_token_count(event.conversation_id)
            prompt_len = event.input_tokens - context_len
        prompt_text = self.conversation_manager.generate_prompt(prompt_len, event.conversation_id)
        self.conversation_manager.add_user_message(event.conversation_id, prompt_text, prompt_len)
        messages = self.conversation_manager.get_all_messages(event.conversation_id)
        tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        total_length = len(tokens) + event.generated_tokens
        if total_length > self.max_context_length:
            logger.warning(f"⚠️  ABORTING REQUEST: Total length is {total_length}.")
            self.aborted_requests += 1
            self.length_violations.append({
                "conversation_id": event.conversation_id,
                "turn_number": event.turn_number,
                "input_tokens": event.input_tokens,
                "generated_tokens": event.generated_tokens,
                "total_len": total_length
            })
            self.aborted_conversations.add(event.conversation_id)
            return
        
        # Track lookup tokens by turn count
        self.lookup_tokens_by_turns[turn_count] += len(tokens)
        
        start_time = time.perf_counter()
        cached_tokens = self.engine.lookup(tokens, lookup_id=event.request_id, pin=True)
        lookup_time = time.perf_counter() - start_time
        self.total_lookup_time += lookup_time
        self.lookup_count += 1
        hit_rate = cached_tokens / len(tokens) if len(tokens) > 0 else 0
        logger.info(f"  Lookup: {cached_tokens}/{len(tokens)} tokens cached"
                    f"({hit_rate:.1%} hit rate, {lookup_time:.4f}s)")
        self.total_tokens += event.input_tokens
        
        # Track hits by turn count
        if cached_tokens > 0:
            self.hits_by_turns[turn_count] += 1
            self.hit_tokens_by_turns[turn_count] += cached_tokens
        
        if self.mode == 'server':
            if cached_tokens != event.hit_tokens:
                logger.warning(f"hit tokens mismatch: {cached_tokens} != {event.hit_tokens}")
            if event.hit_tokens > 0:
                self.total_hit_tokens += event.hit_tokens
                self.requests_with_hits += 1
        if cached_tokens > 0:
            slot_mapping = self._generate_slot_mapping(len(tokens))
            kv_caches = self._get_kv_cache_for_tokens(len(tokens))
            self.connector.kvcaches = kv_caches
            start_time = time.perf_counter()
            ret_mask = self.engine.retrieve(tokens, **{
                'kvcaches': kv_caches,
                'slot_mapping': slot_mapping,
                'request_id': event.request_id
            })
            retrieve_time = time.perf_counter() - start_time
            self.total_retrieve_time += retrieve_time
            self.retrieve_count += 1
            retrieved_tokens = torch.sum(ret_mask).item()
            logger.info(f"  ✓ Retrieved {retrieved_tokens} tokens in {retrieve_time:.4f}s")
        else:
            logger.info(f"  ✗ No tokens to retrieve")
        self.engine.lookup_unpin([event.request_id])
        self.processed_requests += 1

        slot_mapping = self._generate_slot_mapping(len(tokens))
        kv_caches = self._get_kv_cache_for_tokens(len(tokens))
        self.connector.kvcaches = kv_caches
        start_time = time.perf_counter()
        self.engine.store(tokens, **{
            'kvcaches': kv_caches,
            'slot_mapping': slot_mapping,
            'request_id': event.request_id
        })
        store_time = time.perf_counter() - start_time
        self.total_store_time += store_time
        # self.store_count += 1
        logger.info(f"  ✓ Stored {len(tokens)} tokens in {store_time:.4f}s")

    def handle_done_event(self, event: Event):
        if event.conversation_id in self.aborted_conversations:
            logger.warning(f"Skipping Done event for aborted conversation {event.conversation_id}")
            return
        generated_text = self.conversation_manager.generate_prompt(
            event.generated_tokens, event.conversation_id
        )
        self.conversation_manager.add_gpt_message(
            event.conversation_id, generated_text, event.generated_tokens)
        messages = self.conversation_manager.get_all_messages(event.conversation_id)
        tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        token_text = self.tokenizer.decode(tokens)
        if '</think>' in token_text:
            token_text = token_text.replace("<think>\n\n</think>\n\n", "")
        tokens = self.tokenizer.encode(token_text)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)

        token_len = len(tokens)
        logger.info(f"Processing Done event: conv_id={event.conversation_id}, "
            f"generated_tokens={event.generated_tokens}, all_tokens={len(tokens)}")
        aligned_token_len = (
            token_len // self.chunk_size * self.chunk_size
        )
        aligned_tokens = tokens[:aligned_token_len]
        slot_mapping = self._generate_slot_mapping(len(aligned_tokens))
        kv_caches = self._get_kv_cache_for_tokens(len(aligned_tokens))
        self.connector.kvcaches = kv_caches
        start_time = time.perf_counter()
        self.engine.store(aligned_tokens, **{
            'kvcaches': kv_caches,
            'slot_mapping': slot_mapping,
            'request_id': event.request_id
        })
        store_time = time.perf_counter() - start_time
        self.total_store_time += store_time
        self.store_count += 1
        logger.info(f"  ✓ Stored {len(aligned_tokens)} tokens in {store_time:.4f}s")

    def replay_log_events(self, events: List[Event]):
        if self.enable_timing_sync:
            logger.info(f"Starting replay of {len(events)} events with original timing")
        else:
            logger.info(f"Starting replay of {len(events)} events (timing synchronization disabled)")
        
        start_time = time.perf_counter()
        
        # Sort events by timestamp to ensure proper order
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Calculate original log duration for timing comparison
        if self.enable_timing_sync:
            valid_timestamps = [e.timestamp for e in sorted_events if e.timestamp > 0]
            if len(valid_timestamps) >= 2:
                self._original_duration = max(valid_timestamps) - min(valid_timestamps)
                logger.info(f"Original log duration: {self._original_duration:.2f}s")

            # Find the first valid timestamp to establish baseline
            first_timestamp = None
            for event in sorted_events:
                if event.timestamp > 0:
                    first_timestamp = event.timestamp
                    break
            
            if first_timestamp is None:
                logger.warning("No valid timestamps found, proceeding without timing synchronization")
                first_timestamp = time.time()
            
            last_event_time = first_timestamp
            replay_start_time = time.time()
        
        for i, event in enumerate(sorted_events):
            # Calculate when this event should occur in replay time
            if self.enable_timing_sync and event.timestamp > 0:
                # Calculate the time difference from the first event
                time_since_first = event.timestamp - first_timestamp
                expected_replay_time = replay_start_time + time_since_first
                
                # Calculate how long to wait
                current_time = time.time()
                wait_time = expected_replay_time - current_time
                
                # Format original timestamp for display
                original_time_str = time.strftime('%H:%M:%S', time.localtime(event.timestamp))
                
                logger.info(f"\n--- Event {i+1}/{len(sorted_events)} (time: {original_time_str} ---")
                time.sleep(wait_time)
            else:
                logger.info(f"\n--- Event {i+1}/{len(sorted_events)} ---")
            
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
                
            if self.enable_timing_sync:
                last_event_time = event.timestamp if event.timestamp > 0 else last_event_time
        
        self.total_running_time = time.perf_counter() - start_time
        logger.info(f"Completed replay of {len(sorted_events)} events in {self.total_running_time:.2f}s")

    def print_statistics(self, api_url: str = None, verbose: bool = False):
        logger.info("\n" + "="*60)
        logger.info("REPLAY STATISTICS")
        logger.info("="*60)
        stats = self.conversation_manager.get_conversation_stats()
        logger.info(f"Total conversations processed: {stats['total_conversations']}")
        logger.info(f"Total messages processed: {stats['total_messages']}")
        logger.info(f"Average messages per conversation: {stats['avg_messages_per_conversation']:.1f}")
        total_send_events = self.processed_requests + self.aborted_requests
        logger.info(f"\nRequest Processing:")
        logger.info(f"  Successfully processed: {self.processed_requests}")
        logger.info(f"  Aborted (length limit): {self.aborted_requests}")
        logger.info(f"  Total Send events: {total_send_events}")
        if total_send_events > 0:
            abort_rate = (self.aborted_requests / total_send_events) * 100
            logger.info(f"  Abort rate: {abort_rate:.1f}%")
        
        # Add timing comparison statistics
        if self.enable_timing_sync and hasattr(self, '_original_duration'):
            logger.info(f"\nTiming Comparison:")
            logger.info(f"  Original log duration: {self._original_duration:.2f}s")
            logger.info(f"  Replay duration: {self.total_running_time:.2f}s")
            if self._original_duration > 0:
                timing_ratio = self.total_running_time / self._original_duration
                logger.info(f"  Timing ratio (replay/original): {timing_ratio:.2f}x")
        
        logger.info(f"\nTiming Statistics:")
        logger.info(f"  Total running time: {self.total_running_time:.4f}s")
        logger.info(f"   lookup time: {self.total_lookup_time:.4f}s ({self.lookup_count})")
        logger.info(f"   retrieve time: {self.total_retrieve_time:.4f}s ({self.retrieve_count})")
        logger.info(f"   store time: {self.total_store_time:.4f}s ({self.store_count})")
        if self.lookup_count > 0:
            avg_lookup_time = self.total_lookup_time / self.lookup_count
            logger.info(f"  Average lookup time: {avg_lookup_time:.4f}s")
        if self.retrieve_count > 0:
            avg_retrieve_time = self.total_retrieve_time / self.retrieve_count
            logger.info(f"  Average retrieve time: {avg_retrieve_time:.4f}s")
        if self.store_count > 0:
            avg_store_time = self.total_store_time / self.store_count
            logger.info(f"  Average store time: {avg_store_time:.4f}s")
        
        # Replace turn_distribution with turn hit statistics
        logger.info(f"\nTurn Hit Statistics:")
        logger.info(f"  Hit Ratios by Turn Count:")
        for turn_count in sorted(self.requests_by_turns.keys()):
            hits = self.hits_by_turns.get(turn_count, 0)
            requests = self.requests_by_turns.get(turn_count, 0)
            hit_ratio = hits / requests if requests > 0 else 0.0
            logger.info(f"    {turn_count} turns: {hit_ratio:.2%} ({hits}/{requests} requests)")
        
        logger.info(f"  Token Hit Ratios by Turn Count:")
        for turn_count in sorted(self.lookup_tokens_by_turns.keys()):
            lookup_tokens = self.lookup_tokens_by_turns.get(turn_count, 0)
            hit_tokens = self.hit_tokens_by_turns.get(turn_count, 0)
            token_hit_ratio = hit_tokens / lookup_tokens if lookup_tokens > 0 else 0.0
            logger.info(f"    {turn_count} turns: {token_hit_ratio:.2%} ({hit_tokens}/{lookup_tokens})")
        
        if self.mode == 'server':
            if self.requests_with_hits > 0:
                logger.info(f"\nLog Statistics:")
                logger.info(f"  Total hit tokens: {self.total_hit_tokens}")
                logger.info(f"  Requests with hits: {self.requests_with_hits}")
                avg_hit_tokens = self.total_hit_tokens / self.total_tokens if self.total_tokens else 0
                logger.info(f"  Hit rate: {avg_hit_tokens:.2%}")
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
        logger.info("Cleaning up...")
        try:
            LMCacheEngineBuilder.destroy(f"log_replay_{self.mode}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description='Replay client or server logs')
    parser.add_argument('--mode', type=str, choices=['client', 'server'], default='server')
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--cache-size', type=float, default=64)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--max-context-length', type=int, default=16384)
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--api-url', type=str, default="http://localhost:9000/v1/chat/completions")
    parser.add_argument('--eviction-policy', type=str, default="lru")
    parser.add_argument('--timing-sync', action='store_true')
    args = parser.parse_args()

    if 'qwen-8b' in args.input_file:
        model_name = 'Qwen/Qwen3-8B-FP8'
    elif 'llama-8b' in args.input_file:
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    else:
        raise ValueError(f"Unknown model: {args.input_file}")

    print("LMCache Log Replay Tool")
    print("="*60)
    print(f"Log file: {args.input_file}")
    print(f"Mode: {args.mode}")
    print(f"Model: {model_name}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max context length: {args.max_context_length}")
    print(f"page size: {args.page_size}")
    print(f"eviction policy: {args.eviction_policy}")
    print(f"Timing synchronization: {'Disabled' if not args.timing_sync else 'Enabled'}")
    if args.api_url:
        print(f"API URL: {args.api_url}")
    print("="*60)
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available, but GPU usage is required!")
        return
    else:
        print("INFO: Using GPU (CUDA) for all operations")
    parser_obj = LogParser(args.input_file)
    events = parser_obj.parse_log_file(mode=args.mode)
    if not events:
        print("No events found in log file!")
        return
    if args.max_events:
        events = events[:args.max_events]
        print(f"Limited to first {len(events)} events")
    replay = LMCacheLogReplay(
        model_name=model_name,
        chunk_size=args.chunk_size,
        max_context_length=args.max_context_length,
        page_size=args.page_size,
        device=None,
        cache_size=args.cache_size,
        eviction_policy=args.eviction_policy,
        mode=args.mode,
        enable_timing_sync=args.timing_sync
    )
    try:
        replay.replay_log_events(events)
        replay.print_statistics(api_url=args.api_url, verbose=True)
    except Exception as e:
        logger.error(f"Replay failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        replay.cleanup()
    print("\nLMCache Log Replay Completed!")

if __name__ == "__main__":
    main() 