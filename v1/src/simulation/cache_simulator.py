#!/usr/bin/env python3
"""
Cache Simulator for LMCache Eviction Analysis

This script provides comprehensive analysis of cache behavior including:
- Detailed reuse interval analysis by conversation turn count
- Conversation pattern analysis (length, frequency, patterns)
- Eviction policy effectiveness comparison
- Cache efficiency metrics by conversation characteristics
"""

import argparse
import time
import sys
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import math

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


# When executed as a standalone script
from simulation.common import LogParser, Event, get_bytes_per_token, get_chat_template_overhead
from simulation.conversation_tracker import ConversationTracker, ConversationFeature
from simulation.rule_scorer import RuleScorer, RuleScorerConfig

from simulation.lightgbm_scorer import (
    LightGBMScorer, 
    LightGBMScorerConfig, 
    get_model_path_from_input_file,
    train_model_from_events
)

from simulation.oracle_scorer import (
    create_oracle_scorer_from_events
)

@dataclass
class CacheEntry:
    """Represents a cache entry in the simulator"""
    conversation_id: int
    turn_number: int
    token_count: int  # Total tokens stored
    size: int
    access_timestamp: float
    feature: Optional[ConversationFeature] = None
    score: float = 0.0
    hit_count: int = 0
    creation_time: float = 0.0
    eviction_time: Optional[float] = None
    is_last_prefill: bool = False


@dataclass
class SimulatorConfig:
    """Configuration for the cache simulator"""
    cache_size_gb: float = 10.0
    chunk_size: int = 256
    block_size: int = 256  # Block size for cache entries
    max_context_length: int = 16384
    page_size: int = 16
    bytes_per_token: int = 144 * 1024  # Estimated bytes per token for KV cache
    eviction_policy: str = "conversation_aware"  # "lru", "conversation_aware", "lightgbm", "oracle"
    decay_reference: float = 120.0
    scoring_config: Optional[RuleScorerConfig] = None
    lightgbm_config: Optional[LightGBMScorerConfig] = None
    score_update_interval: float = 10.0
    enable_detailed_analysis: bool = True
    chat_template_overhead: int = 0  # Tokens added by chat template
    model_path: Optional[str] = None  # Path to trained LightGBM model

class LRUEvictionPolicy:
    """Simple LRU eviction policy for comparison"""
    
    def __init__(self):
        self.access_order: deque = deque()
        self.key_to_position: Dict[str, int] = {}
        self.total_size = 0
    
    def add(self, key: str, entry: CacheEntry):
        """Add a key to the LRU policy"""
        if key in self.key_to_position:
            raise ValueError(f"Key {key} already exists in LRU policy")
        self.total_size += entry.size
        self.access_order.append(key)
        self.key_to_position[key] = len(self.access_order) - 1
    
    def get_evict_candidate(self) -> Optional[str]:
        """Get the least recently used key"""
        if self.access_order:
            return self.access_order[0]
        return None
    
    def remove(self, key: str, entry: CacheEntry):
        """Remove a key from the policy"""
        if key in self.key_to_position:
            self.access_order.remove(key)
            del self.key_to_position[key]
            self.total_size -= entry.size

class ConversationAwareEvictionPolicy:
    """Conversation-aware eviction policy using scoring"""
    
    def __init__(self, scoring_config: Optional[RuleScorerConfig] = None,
                 score_update_interval: float = 10.0,
                 scorer=None,
                 decay_reference: float = 120.0):
        # Allow custom scorer or create default RuleScorer
        self.scorer = scorer if scorer is not None else RuleScorer(scoring_config)
        self.score_update_interval = score_update_interval
        self.current_time = 0
        self._last_score_update = 0
        self.total_size = 0
        self.start_time = None  # Track the start time for relative timestamps
        self.decay_reference = decay_reference
        
        # Key data storage: key -> entry data
        self.key_data: Dict[str, Dict] = {}
        
        # Sorted list of (score, key) tuples for efficient eviction
        self.sorted_scores: List[Tuple[float, str]] = []
    
    def _score_decay(self, access_timestamp: float, score: float) -> float:
        """Score decay function"""
        
        relative_access_time = access_timestamp - self.start_time
        interval_seconds = self.current_time - relative_access_time
        
        # Prevent math range error by clamping the exponent
        exponent = -interval_seconds / self.decay_reference
        
        return score + exponent
    
    def _calculate_score(self, entry: CacheEntry) -> float:
        """Calculate score for a cache entry. Can be overridden by subclasses."""
        # Default implementation for RuleScorer
        self.scorer.set_current_time(self.current_time)
        # score can be negative
        score = self.scorer.calculate_score(entry.feature)
        
        # Apply decay based on relative timestamps
        return self._score_decay(entry.access_timestamp, score)
    
    def add(self, key: str, entry: CacheEntry):
        """Add a key with calculated score"""
        # Set start time if not set
        if self.start_time is None:
            self.start_time = entry.access_timestamp
        
        self.current_time = entry.access_timestamp - self.start_time
        score = self._calculate_score(entry)
        
        self.key_data[key] = {
            "entry": entry,
            "score": score,
            "last_score_update": self.current_time
        }
        self.total_size += entry.size
        # Insert into sorted list
        self._insert_sorted_score(score, key)
    
    def get_evict_candidate(self) -> Optional[str]:
        """Get the key with lowest score"""
        if self.current_time - self._last_score_update > self.score_update_interval:
            self._update_all_scores()
            self._last_score_update = self.current_time
        
        if self.sorted_scores:
            return self.sorted_scores[0][1]
        return None
    
    def remove(self, key: str, entry: CacheEntry):
        """Remove a key from the policy"""
        if key in self.key_data:
            score = self.key_data[key]["score"]
            self._remove_sorted_score(score, key)
            del self.key_data[key]
            self.total_size -= entry.size

    def _insert_sorted_score(self, score: float, key: str):
        """Insert a (score, key) tuple into the sorted list"""
        import bisect
        insert_pos = bisect.bisect_left(self.sorted_scores, (score, key))
        self.sorted_scores.insert(insert_pos, (score, key))
    
    def _remove_sorted_score(self, score: float, key: str):
        """Remove a (score, key) tuple from the sorted list"""
        import bisect
        try:
            index = bisect.bisect_left(self.sorted_scores, (score, key))
            if (index < len(self.sorted_scores) and 
                self.sorted_scores[index] == (score, key)):
                self.sorted_scores.pop(index)
        except (ValueError, IndexError):
            pass

    def _update_all_scores(self):
        """Update all scores based on current time."""
        current_time = self.current_time
        
        # Create a new sorted list
        new_sorted_scores = []
        
        # Recalculate all scores and build new sorted list
        for key, data in self.key_data.items():
            entry = data["entry"]
            
            # Calculate updated score using the flexible scoring method
            new_score = self._calculate_score(entry)
            
            # Update stored score
            data["score"] = new_score
            data["last_score_update"] = current_time
            
            # Insert into new sorted list
            import bisect
            insert_pos = bisect.bisect_left(new_sorted_scores, (new_score, key))
            new_sorted_scores.insert(insert_pos, (new_score, key))
        
        # Replace the old sorted list with the new one
        self.sorted_scores = new_sorted_scores
        self._last_score_update = current_time

class LightGBMEvictionPolicy(ConversationAwareEvictionPolicy):
    """LightGBM-based eviction policy using machine learning scoring"""
    
    def __init__(self, lightgbm_config: Optional[LightGBMScorerConfig] = None,
                 model_path: Optional[str] = None,
                 score_update_interval: float = 10.0):
        # Check if LightGBM is available
        if LightGBMScorer is None:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        # Create LightGBM scorer
        scorer = LightGBMScorer(lightgbm_config)
        if model_path and os.path.exists(model_path):
            scorer.load_model(model_path)
        
        # Initialize parent class with the LightGBM scorer
        super().__init__(scoring_config=None, score_update_interval=score_update_interval, scorer=scorer)


class OracleEvictionPolicy(ConversationAwareEvictionPolicy):
    """Oracle-based eviction policy using perfect future knowledge"""
    
    def __init__(self, events: List[Event]):
        # Create Oracle scorer and initialize with events
        scorer = create_oracle_scorer_from_events(events)
        
        # Initialize parent class with the Oracle scorer
        super().__init__(scorer=scorer)
    
    def _score_decay(self, access_timestamp: float, score: float) -> float:
        """Score decay function"""
        return score


class CacheSimulator:
    """Cache simulator with detailed analysis capabilities"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.cache_size_bytes = config.cache_size_gb * 1024 * 1024 * 1024 * 0.95
        self.current_cache_size = 0
        
        # Cache storage: key -> CacheEntry
        self.cache: Dict[str, CacheEntry] = {}
        
        # Initialize eviction policy
        if config.eviction_policy == "lru":
            self.eviction_policy = LRUEvictionPolicy()
        elif config.eviction_policy == "lightgbm":
            self.eviction_policy = LightGBMEvictionPolicy(
                config.lightgbm_config, config.model_path, config.score_update_interval)
        elif config.eviction_policy == "oracle":
            # Oracle policy needs events for initialization, so we'll defer this
            # The eviction policy will be set later when events are available
            self.eviction_policy = None
        else:  # conversation_aware
            self.eviction_policy = ConversationAwareEvictionPolicy(
                config.scoring_config, 
                config.score_update_interval, 
                decay_reference=config.decay_reference
            )
        
        # Basic statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.skipped_cache_entries = 0  # Track entries not cached due to optimization
        
        # Conversation tracking
        self.conversation_tracker = ConversationTracker()

        # Hit ratio by turn count
        self.hits_by_turns: Dict[int, int] = defaultdict(int)
        self.requests_by_turns: Dict[int, int] = defaultdict(int)
        
        # Token-level tracking by turn count
        self.lookup_tokens_by_turns: Dict[int, int] = defaultdict(int)
        self.hit_tokens_by_turns: Dict[int, int] = defaultdict(int)
        
        # Turn analysis metrics  
        self.turn_analysis: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"hit_ratio": 0.0, "avg_reuse_interval": 0.0, 
            "eviction_rate": 0.0, "token_hit_ratio": 0.0,
            "avg_query_length": 0.0, "avg_response_length": 0.0}
        )
        # Token-level statistics
        self.lookup_tokens = 0
        self.hit_tokens = 0
        logger.info(f"Cache simulator initialized with {config.cache_size_gb}GB cache")
    
    def initialize_oracle_policy(self, events: List[Event]):
        """Initialize oracle eviction policy with events (called after events are loaded)"""
        if self.config.eviction_policy == "oracle" and self.eviction_policy is None:
            self.eviction_policy = OracleEvictionPolicy(events)
    
    def _generate_cache_key(self, conversation_id: int, turn_number: int) -> str:
        """Generate a cache key for a conversation turn"""
        return f"conv_{conversation_id}_turn_{turn_number}"
    
    def _get_aligned_token_count(self, token_count: int) -> int:
        """Get the number of tokens that form complete blocks"""
        block_size = self.config.block_size
        complete_blocks = token_count // block_size
        return complete_blocks * block_size
    
    def _calculate_entry_size(self, token_count: int) -> int:
        """Calculate the size of a cache entry in bytes, only counting complete blocks"""
        return token_count * self.config.bytes_per_token
    
    def _evict_if_needed(self, required_size: int):
        """Evict entries if cache is full"""
        while self.current_cache_size + required_size > self.cache_size_bytes:
            evict_key = self.eviction_policy.get_evict_candidate()
            if not evict_key:
                logger.warning("No eviction candidate found but cache is full!")
                break
            
            # Remove from cache
            entry = self.cache[evict_key]
            entry.eviction_time = time.time()
            
            del self.cache[evict_key]
            self.eviction_policy.remove(evict_key, entry)
            self.current_cache_size -= entry.size
            self.evictions += 1
            
            logger.debug(f"Evicted {evict_key}, size: {entry.size}, "
                         f"remaining: {self.current_cache_size}")
    
    def lookup(self, conversation_id: int, turn_number: int, token_count: int, 
               current_time: float, query_len: int = 0) -> Tuple[bool, int]:
        """
        Simulate a cache lookup operation.
        
        Returns:
            Tuple of (is_hit, hit_token_count)
        """
        self.total_requests += 1
        self.lookup_tokens += token_count
        
        self.requests_by_turns[turn_number] += 1
        self.lookup_tokens_by_turns[turn_number] += token_count
        
        # Generate cache key
        lookup_turn_number = max(1, turn_number - 1)
        cache_key = self._generate_cache_key(conversation_id, lookup_turn_number)
        
        # Check for cache hit
        hit = False
        hit_tokens = 0
        
        if cache_key in self.cache:
            # Use complete blocks count for hit token calculation
            # the last prefill has different key and is not counted as a hit
            entry = self.cache[cache_key]
            hit_token_count = entry.token_count

            self.cache_hits += 1
            self.hit_tokens += hit_token_count
            self.hits_by_turns[turn_number] += 1
            self.hit_tokens_by_turns[turn_number] += hit_token_count
            hit = True
            hit_tokens = hit_token_count
        else:
            # Cache miss or hit tokens is 0
            self.cache_misses += 1
        
        return hit, hit_tokens
    
    def store(self, conversation_id: int, turn_number: int, token_count: int, 
              current_time: float, feature: ConversationFeature, is_last_prefill: bool):
        """Simulate storing an entry in the cache"""
        if token_count == 0:
            return
        
        # Optimization: Check if entry should be cached at all
        # For conversation-aware policy, use the scorer to decide
        if self.config.eviction_policy == "conversation_aware":
            if not self.eviction_policy.scorer.cache_admission(feature):
                self.skipped_cache_entries += 1
                logger.debug(f"Skipping cache for conv_{conversation_id}_turn_{turn_number} "
                           f"- unlikely to be reused (skipped: {self.skipped_cache_entries})")
                return
        
        # Calculate entry size based on complete blocks (for cache management)
        entry_size_bytes = self._calculate_entry_size(token_count)
        
        # evict the previous cache key if it exists
        if not is_last_prefill:
            lookup_turn_number = max(1, turn_number - 1)
            prev_cache_key = self._generate_cache_key(conversation_id, lookup_turn_number)
            if prev_cache_key in self.cache:
                prev_entry = self.cache[prev_cache_key]
                self.eviction_policy.remove(prev_cache_key, prev_entry)
                self.current_cache_size -= prev_entry.size
                del self.cache[prev_cache_key]
                # logger.debug(f"Evicting before updating {prev_cache_key}")
            current_cache_key = self._generate_cache_key(conversation_id, turn_number)
            if current_cache_key in self.cache:
                current_entry = self.cache[current_cache_key]
                self.eviction_policy.remove(current_cache_key, current_entry)
                self.current_cache_size -= current_entry.size
                del self.cache[current_cache_key]
        
        # Evict if needed
        self._evict_if_needed(entry_size_bytes)
        
        # Create cache entry with total tokens but size based on complete blocks
        entry = CacheEntry(
            conversation_id=conversation_id,
            turn_number=turn_number,
            token_count=token_count,  # Stored tokens
            access_timestamp=current_time,
            size=entry_size_bytes,
            feature=feature,
            creation_time=current_time,
            is_last_prefill=is_last_prefill
        )
        
        # Store in cache
        cache_key = self._generate_cache_key(conversation_id, turn_number)
        if is_last_prefill:
            cache_key += "_last_prefill"
        self.cache[cache_key] = entry
        self.current_cache_size += entry_size_bytes
        
        # Add to eviction policy
        self.eviction_policy.add(cache_key, entry)
    
    def _calculate_turn_analysis_metrics(self):
        """Calculate detailed turn analysis metrics"""
        for turn_count in self.requests_by_turns:
            hits = self.hits_by_turns.get(turn_count, 0)
            requests = self.requests_by_turns[turn_count]
            hit_ratio = hits / requests if requests > 0 else 0.0
            
            # Calculate token hit ratio by turn count
            lookup_tokens = self.lookup_tokens_by_turns.get(turn_count, 0)
            hit_tokens = self.hit_tokens_by_turns.get(turn_count, 0)
            token_hit_ratio = hit_tokens / lookup_tokens if lookup_tokens > 0 else 0.0
            
            # Calculate average stats for this turn count
            # Separate tracking for conversations that ended exactly at turn_count vs continued beyond
            reuse_intervals_exact = []
            query_lengths_exact = []
            response_lengths_exact = []
            reuse_intervals_continued = []
            query_lengths_continued = []
            response_lengths_continued = []
            
            for conv_id, state in self.conversation_tracker.conversation_states.items():
                if state.turn_number >= turn_count:
                    # Only add reuse interval for turns > 1 (turn 1 has no previous turn)
                    if turn_count > 1:
                        interval = state.get_average_reuse_interval_until(turn_count)
                        if interval > 0:
                            if state.turn_number == turn_count:
                                reuse_intervals_exact.append(interval)
                            else:  # state.turn_number > turn_count
                                reuse_intervals_continued.append(interval)
                    
                    # Always add query and response lengths for all turns
                    query_len = state.get_average_query_length_until(turn_count)
                    response_len = state.get_average_response_length_until(turn_count)
                    
                    if state.turn_number == turn_count:
                        query_lengths_exact.append(query_len)
                        response_lengths_exact.append(response_len)
                    else:  # state.turn_number > turn_count
                        query_lengths_continued.append(query_len)
                        response_lengths_continued.append(response_len)
            
            # Calculate averages as dictionaries
            avg_reuse_interval = {
                "exact": (sum(reuse_intervals_exact) / len(reuse_intervals_exact) 
                         if reuse_intervals_exact else 0.0),
                "continued": (sum(reuse_intervals_continued) / len(reuse_intervals_continued) 
                             if reuse_intervals_continued else 0.0)
            }
            avg_query_length = {
                "exact": (sum(query_lengths_exact) / len(query_lengths_exact) 
                         if query_lengths_exact else 0.0),
                "continued": (sum(query_lengths_continued) / len(query_lengths_continued) 
                             if query_lengths_continued else 0.0)
            }
            avg_response_length = {
                "exact": (sum(response_lengths_exact) / len(response_lengths_exact) 
                         if response_lengths_exact else 0.0),
                "continued": (sum(response_lengths_continued) / len(response_lengths_continued) 
                             if response_lengths_continued else 0.0)
            }
            
            # Calculate eviction rate (simplified)
            eviction_rate = 0.0  # Would need more detailed tracking
            
            self.turn_analysis[turn_count] = {
                "hit_ratio": hit_ratio,
                "avg_reuse_interval": avg_reuse_interval,
                "eviction_rate": eviction_rate,
                "token_hit_ratio": token_hit_ratio,
                "avg_query_length": avg_query_length,
                "avg_response_length": avg_response_length
            }
    
    def print_detailed_snapshot(self):
        """Dump detailed cache information for debugging"""
        logger.info("\n" + "="*80)
        logger.info("CACHE DEBUG INFORMATION")
        logger.info("="*80)
        
        # Dump current cache contents
        logger.info(f"\nCurrent cache contents ({len(self.cache)} entries):")
        logger.info(f"Cache size: {self.current_cache_size / (1024**3):.2f} GB")
        logger.info(f"Max cache size: {self.cache_size_bytes / (1024**3):.2f} GB")
        logger.info(f"Utilization: {(self.current_cache_size / self.cache_size_bytes) * 100:.1f}%")
        
        # Group entries by conversation
        conv_entries = {}
        for key, entry in self.cache.items():
            conv_id = entry.conversation_id
            if conv_id not in conv_entries:
                conv_entries[conv_id] = []
            conv_entries[conv_id].append((key, entry))
        
        logger.info(f"\nConversations in cache ({len(conv_entries)} conversations):")
        for conv_id in sorted(conv_entries.keys()):
            entries = conv_entries[conv_id]
            total_tokens = sum(entry.token_count for _, entry in entries)
            total_size = sum(entry.size for _, entry in entries)
            logger.info(f"  Conversation {conv_id}: {len(entries)} entries, "
                       f"{total_tokens} tokens, {total_size / (1024**2):.1f} MB")
            
            # Show individual entries
            for key, entry in sorted(entries, key=lambda x: x[1].turn_number):
                logger.info(f"    {key}: turn {entry.turn_number}, "
                           f"{entry.token_count} tokens, "
                           f"score: {entry.score:.4f}, "
                           f"access: {entry.access_timestamp:.1f}")
        
        # Dump eviction policy information
        if hasattr(self.eviction_policy, 'sorted_scores'):
            logger.info(f"\nEviction policy sorted scores (top 20 lowest scores):")
            for i, (score, key) in enumerate(self.eviction_policy.sorted_scores[:20]):
                if key in self.cache:
                    entry = self.cache[key]
                    logger.info(f"  {i+1}. {key}: score={score:.4f}, "
                               f"conv_{entry.conversation_id}_turn_{entry.turn_number}, "
                               f"tokens={entry.token_count}")
        
        logger.info(f"\nEviction statistics:")
        logger.info(f"  Total evictions: {self.evictions}")

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics with detailed analysis"""
        self._calculate_turn_analysis_metrics()
        
        hit_ratio = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "basic_stats": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_ratio": hit_ratio,
                "evictions": self.evictions,
                "skipped_cache_entries": self.skipped_cache_entries,
                "current_cache_size_gb": self.current_cache_size / (1024**3),
                "cache_utilization": self.current_cache_size / self.cache_size_bytes,
                "eviction_policy": self.config.eviction_policy,
                "lookup_tokens": self.lookup_tokens,
                "hit_tokens": self.hit_tokens,
                "token_hit_ratio": self.hit_tokens / self.lookup_tokens if self.lookup_tokens > 0 else 0.0
            },
            "turn_analysis": dict(self.turn_analysis)
        }


class CacheSimulatorReplay:
    """Main class for replaying logs and simulating cache behavior with detailed analysis"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.simulator = CacheSimulator(config)
        
        # Track conversations whose send events were aborted
        self.aborted_conversations: Set[int] = set()
        self.aborted_requests = 0
        self.processed_requests = 0
        
        # Conversation state management is handled by the simulator's conversation manager
        
        logger.info(f"CacheSimulatorReplay initialized with {config.eviction_policy} policy")
    
    def handle_send_event(self, event: Event):        
        """Handle Send event - perform lookup simulation"""
        conv_tracker = self.simulator.conversation_tracker
        logger.info(f"Send: conv_{event.conversation_id}_{event.turn_number}"
                   f" all_tokens={event.input_tokens}")
        feature = conv_tracker.get_conversation_feature(event.conversation_id)
        if feature is not None:
            logger.info(f"  Previous Feature: turn={event.turn_number-1}, " 
                f"avg_query_len={feature.CustomerQueryLength}, "
                f"avg_resp_len={feature.GenerativeModelResponseLength}, "
                f"avg_reuse_interval={feature.previous_interval:.2f}, "
            )
            state = conv_tracker.get_conversation_state(event.conversation_id)
            _query_len, _resp_len, _reuse_interval = state.get_state_of_turn(event.turn_number-1)
            logger.info(f"  Previous State: turn={event.turn_number-1}, "
                f"query_len={_query_len}, prev_resp_len={_resp_len}, "
                f"previous_interval={_reuse_interval}"
            )
        
        query_len = conv_tracker.calculate_query_len(event.conversation_id, event.input_tokens)

        # Update conversation turn state and input length
        conv_tracker.set_context_length(event.conversation_id, event.input_tokens)
        token_count = event.input_tokens
        
        # Calculate interval from previous event for this conversation
        current_timestamp = event.timestamp if event.timestamp > 0 else time.time()
        
        # Simulate cache lookup with event and interval
        hit, hit_tokens = self.simulator.lookup(
            event.conversation_id, 
            event.turn_number, 
            token_count, 
            current_timestamp,
            query_len
        )
        # Log the result
        if event.turn_number > 1:
            if hit:
                logger.info(f"  ✓ Cache HIT (hit tokens: {hit_tokens})")
            else:
                logger.info(f"  ✗ Cache MISS")

        feature = conv_tracker.update_conversation_feature(
            event.conversation_id, 
            event.turn_number, 
            current_timestamp, 
            query_len,
            0,  # No response tokens available in Send event
            source="send"
        )
        
        self.simulator.store(
            event.conversation_id,
            event.turn_number,
            self.simulator._get_aligned_token_count(token_count),  # aligned tokens
            current_timestamp,
            feature,
            is_last_prefill=False
        )
        self.simulator.store(
            event.conversation_id,
            event.turn_number,
            token_count - self.simulator._get_aligned_token_count(token_count), # tail
            current_timestamp,
            feature,
            is_last_prefill=True
        )
        
        self.processed_requests += 1
    
    def handle_done_event(self, event: Event):
        """Handle Done event - perform store simulation"""
        # Skip if the conversation was aborted
        if event.conversation_id in self.aborted_conversations:
            logger.warning(f"Skipping Done event for aborted conversation {event.conversation_id}")
            return
        
        # Get the current turn_number for this conversation
        conv_tracker = self.simulator.conversation_tracker
        current_turn_number = event.turn_number
        if current_turn_number == 0:
            # If no turn_number tracked, this is likely the first turn
            current_turn_number = 1
        
        # Get the input length from the corresponding send event
        input_tokens = conv_tracker.get_context_length(event.conversation_id)
        output_tokens = event.generated_tokens if event.generated_tokens else 0
        
        token_count = input_tokens + output_tokens
        conv_tracker.set_context_length(event.conversation_id, token_count)
        
        logger.info(f"Done: conv_{event.conversation_id}_{current_turn_number}"
                   f" all_tokens={token_count}")
        
        # Simulate cache store with event
        current_timestamp = event.timestamp if event.timestamp > 0 else time.time()
        self.simulator.store(
            event.conversation_id,
            current_turn_number,
            self.simulator._get_aligned_token_count(token_count),  # cumulative
            current_timestamp,
            conv_tracker.update_conversation_feature(
                event.conversation_id, 
                current_turn_number, 
                current_timestamp,
                0, 
                output_tokens, 
                source="done"
            ),
            is_last_prefill=False
        )
        
        logger.info(f"  ✓ input: {input_tokens}, output: {output_tokens}")
    
    def replay_log_events(self, events: List[Event]):
        """Replay all log events and simulate cache operations"""
        logger.info(f"Starting replay of {len(events)} events")
        
        # Initialize oracle policy if needed (must be done after events are loaded)
        self.simulator.initialize_oracle_policy(events)
        
        start_time = time.perf_counter()
        
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
        
        total_time = time.perf_counter() - start_time
        logger.info(f"\nCompleted replay of {len(events)} events in {total_time:.4f}s")
    
    def print_detailed_statistics(self):
        """Print comprehensive statistics with detailed analysis"""
        # Dump cache debug information
        # self.simulator.print_detailed_snapshot()

        stats = self.simulator.get_detailed_statistics()
        
        logger.info("\n" + "="*80)
        logger.info("CACHE SIMULATOR STATISTICS")
        logger.info("="*80)
        
        # Request processing statistics
        total_send_events = self.processed_requests + self.aborted_requests
        logger.info(f"\nRequest Processing:")
        logger.info(f"  Successfully processed: {self.processed_requests}")
        logger.info(f"  Aborted (length limit): {self.aborted_requests}")
        logger.info(f"  Total Send events: {total_send_events}")
        if total_send_events > 0:
            abort_rate = (self.aborted_requests / total_send_events) * 100
            logger.info(f"  Abort rate: {abort_rate:.1f}%")
        
        # Turn-based analysis
        turn_analysis = stats["turn_analysis"]
        logger.info(f"\nTurn-Based Analysis:")
        logger.info(f"  Hit Ratios by Turn Count:")
        for turn_count in sorted(turn_analysis.keys()):
            data = turn_analysis[turn_count]
            hit_ratio = data["hit_ratio"]
            requests = self.simulator.requests_by_turns.get(turn_count, 0)
            hits = self.simulator.hits_by_turns.get(turn_count, 0)
            
            logger.info(f"    {turn_count} turns: {hit_ratio:.2%} ({hits}/{requests} requests)")
        
        logger.info(f"  Conversation Features by Turn Count:")
        for turn_count in sorted(turn_analysis.keys()):
            data = turn_analysis[turn_count]
            # Print for conversations that ended exactly at this turn count
            logger.info(f"    {turn_count} turns (exact): req={data['avg_query_length']['exact']:.1f},"
                        f" resp={data['avg_response_length']['exact']:.1f}, reuse={data['avg_reuse_interval']['exact']:.1f}")
            # Print for conversations that continued beyond this turn count
            logger.info(f"    {turn_count} turns (continued): req={data['avg_query_length']['continued']:.1f},"
                        f" resp={data['avg_response_length']['continued']:.1f}, reuse={data['avg_reuse_interval']['continued']:.1f}")
        
        logger.info(f"  Token Hit Ratios by Turn Count:")
        for turn_count in sorted(turn_analysis.keys()):
            data = turn_analysis[turn_count]
            token_hit_ratio = data["token_hit_ratio"]
            lookup_tokens = self.simulator.lookup_tokens_by_turns.get(turn_count, 0)
            hit_tokens = self.simulator.hit_tokens_by_turns.get(turn_count, 0)
            logger.info(f"    {turn_count} turns: {token_hit_ratio:.2%} ({hit_tokens}/{lookup_tokens})")
        
        # Basic statistics
        basic_stats = stats["basic_stats"]
        logger.info("\nBasic Statistics:")
        logger.info(f"Eviction Policy: {basic_stats['eviction_policy']}")
        logger.info(f"Total Requests: {basic_stats['total_requests']}")
        logger.info(f"Cache Hits: {basic_stats['cache_hits']}")
        logger.info(f"Cache Misses: {basic_stats['cache_misses']}")
        logger.info(f"Overall Hit Ratio: {basic_stats['hit_ratio']:.2%}")
        logger.info(f"Evictions: {basic_stats['evictions']}")
        logger.info(f"Skipped Cache Entries: {basic_stats['skipped_cache_entries']} "
                   f"({basic_stats['skipped_cache_entries']/(basic_stats['total_requests']+basic_stats['skipped_cache_entries'])*100:.1f}% of all store attempts)")
        logger.info(f"Current Cache Size: {basic_stats['current_cache_size_gb']:.2f} GB")
        logger.info(f"Block Size: {self.config.block_size}")
        logger.info(f"Total Lookup Tokens: {basic_stats['lookup_tokens']}")
        logger.info(f"Total Hit Tokens: {basic_stats['hit_tokens']}")
        logger.info(f"Token Hit Ratio: {basic_stats['token_hit_ratio']:.2%}")
        
        logger.info("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cache simulation for eviction analysis')
    parser.add_argument(
        '--input-file', 
        default='outputs/logs/experiments/qwen-8b_2025-08-12/server_64.0gb_0.9rps_29_5am_6am.log',
        help='Path to the client log file'
    )
    parser.add_argument('--cache-size', type=float, default=64.0,
                       help='Cache size in GB')
    parser.add_argument('--eviction-policy', choices=['lru', 'conversation_aware', 'lightgbm', 'oracle'], 
                       default='conversation_aware',
                       help='Eviction policy to use')
    parser.add_argument('--decay-reference', type=float, default=120.0,
                       help='Decay reference for score decay (default: 120.0)')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process')
    parser.add_argument('--chunk-size', type=int, default=256,
                       help='Chunk size (default: 256)')
    parser.add_argument('--block-size', type=int, default=256,
                       help='Block size for cache entries (default: 256)')
    parser.add_argument('--max-context-length', type=int, default=16384,
                       help='Maximum context length in tokens (default: 16384)')
    parser.add_argument('--model-dir', default='models/',
                       help='Directory to store/load LightGBM models')
    parser.add_argument('--lightgbm-mode', default='regression',
                       help='LightGBM mode (regression, ranking, classification)')
    parser.add_argument('--train-max-events', type=int, default=100000,
                       help='Maximum events to use for LightGBM training (default: 100000)')
    parser.add_argument('--not-force-retrain', action='store_true',
                       help='Force retraining even if model exists')

    args = parser.parse_args()
    # Create configuration
    if 'qwen-8b' in args.input_file:
        model_name = 'Qwen/Qwen3-8B-FP8'
    elif 'llama-8b' in args.input_file:
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    elif 'test_data' in args.input_file:
        # Use Qwen3-8B as default for test_data
        model_name = 'Qwen/Qwen3-8B-FP8'
        print(f"Using default model for test_data: {model_name}")
    else:
        raise ValueError(f"Unknown model: {args.input_file}. Supported models: qwen-8b, llama-8b, test_data")
    _, bytes_per_token = get_bytes_per_token(model_name)
    chat_template_overhead = get_chat_template_overhead(model_name)
    print(f"Bytes per token: {bytes_per_token}")
    print(f"Chat template overhead: {chat_template_overhead} tokens")
    
    print("Cache Simulator for Eviction Analysis")
    print("="*80)
    print(f"Input file: {args.input_file}")
    print(f"Cache size: {args.cache_size} GB")
    print(f"Eviction policy: {args.eviction_policy}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Block size: {args.block_size}")
    print(f"Max context length: {args.max_context_length}")
    print("="*80, flush=True)
    
    # Parse log file
    parser = LogParser(args.input_file)
    if "client" in args.input_file:
        raise ValueError("Client log file not supported")
    else:
        events = parser.parse_log_file(mode='server')
    
    if not events:
        print("No events found in log file!")
        return
    
    # Limit events if specified
    if args.max_events:
        events = events[:args.max_events]
        print(f"Limited to first {len(events)} events")

    # Handle LightGBM model training/loading if needed
    model_path = None
    lightgbm_config = None
    
    if args.eviction_policy == 'lightgbm':
        print("\n" + "="*80)
        print("LightGBM Model Setup")
        print("="*80)
        
        # Generate model path based on input file
        model_path = get_model_path_from_input_file(args.input_file, args.model_dir)
        print(f"Model path: {model_path}")
        
        # Check if model exists and should be trained
        needs_training = not args.not_force_retrain or not os.path.exists(model_path)
        
        if needs_training:
            print("Training new LightGBM model...")
            try:
                # Use the SAME events that will be used for simulation
                # This ensures perfect consistency between training and runtime data
                train_events = events[:args.train_max_events] if args.train_max_events else events
                print(f"Training with {len(train_events)} events (same events used for simulation)")
                
                # Train model using the exact same events
                # Default to ranking mode, but could be made configurable
                trained_scorer = train_model_from_events(train_events, mode=args.lightgbm_mode)
                
                # Save the trained model
                trained_scorer.save_model(model_path)
                print(f"Model training completed and saved to {model_path}")
                
            except Exception as e:
                print(f"Failed to train LightGBM model: {e}")
                print("Falling back to conversation_aware policy")
                args.eviction_policy = 'conversation_aware'
                model_path = None
        else:
            print(f"Using existing model: {model_path}")
        
        # Create LightGBM config
        if model_path:
            lightgbm_config = LightGBMScorerConfig(model_dir=args.model_dir)
        
        print("="*80, flush=True)

    config = SimulatorConfig(
        cache_size_gb=args.cache_size,
        chunk_size=args.chunk_size,
        block_size=args.block_size,
        max_context_length=args.max_context_length,
        eviction_policy=args.eviction_policy,
        decay_reference=args.decay_reference,
        bytes_per_token=bytes_per_token,
        chat_template_overhead=chat_template_overhead,
        lightgbm_config=lightgbm_config,
        model_path=model_path
    )
    
    # Initialize simulator
    replay = CacheSimulatorReplay(config)
    
    try:
        # Replay events
        replay.replay_log_events(events)
        
        # Print final statistics
        replay.print_detailed_statistics()
        
    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nCache Simulation Completed!")

if __name__ == "__main__":
    main() 