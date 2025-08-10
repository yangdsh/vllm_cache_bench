#!/usr/bin/env python3
'''
cd ~/PrefixCacheInternProject/vllm_cache_bench/v1
python cache_simulator.py --log-file experiment_logs/qwen-8b_2025-08-07/server_64.0gb_0.9rps_29_5am_6am.log > replay_sim.log 2>&1
'''

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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import math

import logging

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


from util.common import LogParser, Event
from util.common import get_model_config_and_bytes_per_token, get_chat_template_overhead

@dataclass
class ConversationFeature:
    """Simplified conversation feature for scoring"""
    conversation_id: int
    turn_number: int
    access_timestamp: float
    previous_interval: float = 0.0

@dataclass
class ConversationScoringConfig:
    """Simplified configuration for conversation feature-based scoring."""
    reference_recency: float = 100.0
    reference_time_interval: float = 100.0
    score_factor: float = 1

class ConversationScorer:
    """Simplified conversation scorer for standalone operation"""
    
    def __init__(self, config: Optional[ConversationScoringConfig] = None):
        self.config = config or ConversationScoringConfig()
        
    def calculate_score(
        self, 
        feature: Optional[ConversationFeature],
        obj_access_timestamp: Optional[float] = None
    ) -> float:
        """Calculate an eviction score for a cache entry based on conversation features."""
        if feature is None:
            return 0.0
        
        # Simple scoring based on turn number and time
        recency_score = math.exp(-(time.time() - obj_access_timestamp) / self.config.reference_recency) if obj_access_timestamp else 0.0
        time_interval_score = math.exp(-feature.previous_interval / self.config.reference_time_interval)
        
        # Combine scores
        score = (time_interval_score * self.config.score_factor + 1) * recency_score
        return score

@dataclass
class CacheEntry:
    """Represents a cache entry in the simulator"""
    conversation_id: int
    turn_number: int
    token_count: int  # Total tokens stored
    complete_blocks_count: int  # Number of complete blocks (for lookup)
    size: int
    access_timestamp: float
    feature: Optional[ConversationFeature] = None
    score: float = 0.0
    hit_count: int = 0
    last_access: float = 0.0
    creation_time: float = 0.0
    eviction_time: Optional[float] = None

@dataclass
class ConversationAnalysis:
    """Analysis data for a conversation"""
    conversation_id: int
    total_turns: int = 0
    total_tokens: int = 0
    first_access: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    reuse_intervals: List[float] = field(default_factory=list)
    turn_lengths: List[int] = field(default_factory=list)

@dataclass
class SimulatorConfig:
    """Configuration for the cache simulator"""
    cache_size_gb: float = 10.0
    chunk_size: int = 256
    block_size: int = 256  # Block size for cache entries
    max_context_length: int = 16384
    page_size: int = 16
    bytes_per_token: int = 2048  # Estimated bytes per token for KV cache
    eviction_policy: str = "conversation_aware"  # "lru", "conversation_aware"
    scoring_config: Optional[ConversationScoringConfig] = None
    score_update_interval: float = 10.0
    enable_detailed_analysis: bool = True
    chat_template_overhead: int = 0  # Tokens added by chat template

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
    
    def __init__(self, scoring_config: Optional[ConversationScoringConfig] = None,
                 score_update_interval: float = 10.0):
        self.scorer = ConversationScorer(scoring_config)
        self.score_update_interval = score_update_interval
        self.last_score_update = time.time()
        self.total_size = 0
        
        # Key data storage: key -> entry data
        self.key_data: Dict[str, Dict] = {}
        
        # Sorted list of (score, key) tuples for efficient eviction
        self.sorted_scores: List[Tuple[float, str]] = []
    
    def add(self, key: str, entry: CacheEntry):
        """Add a key with calculated score"""
        current_time = time.time()
        score = self.scorer.calculate_score(entry.feature, entry.access_timestamp)
        
        self.key_data[key] = {
            "entry": entry,
            "score": score,
            "last_score_update": current_time
        }
        self.total_size += entry.size
        # Insert into sorted list
        self._insert_sorted_score(score, key)
    
    def get_evict_candidate(self) -> Optional[str]:
        """Get the key with lowest score"""
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

class CacheSimulator:
    """Cache simulator with detailed analysis capabilities"""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.cache_size_bytes = config.cache_size_gb * 1024 * 1024 * 1024
        self.current_cache_size = 0
        
        # Cache storage: key -> CacheEntry
        self.cache: Dict[str, CacheEntry] = {}
        
        # Initialize eviction policy
        if config.eviction_policy == "lru":
            self.eviction_policy = LRUEvictionPolicy()
        else:
            self.eviction_policy = ConversationAwareEvictionPolicy(
                config.scoring_config, config.score_update_interval
            )
        
        # Basic statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        
        # Conversation tracking
        self.conversation_turns: Dict[int, int] = defaultdict(int)
        self.conversation_features: Dict[int, ConversationFeature] = {}
        
        # Analysis data
        self.conversation_analysis: Dict[int, ConversationAnalysis] = defaultdict(
            lambda: ConversationAnalysis(0)
        )
        
        # Reuse interval analysis
        self.reuse_intervals: Dict[int, List[float]] = defaultdict(list)  # turn_count -> intervals
        self.last_access_times: Dict[str, float] = {}
        
        # Hit ratio by turn count
        self.hits_by_turns: Dict[int, int] = defaultdict(int)
        self.requests_by_turns: Dict[int, int] = defaultdict(int)
        
        # Token-level tracking by turn count
        self.lookup_tokens_by_turns: Dict[int, int] = defaultdict(int)
        self.hit_tokens_by_turns: Dict[int, int] = defaultdict(int)
        
        # Eviction analysis
        self.evicted_conversations: Dict[int, int] = defaultdict(int)  # conv_id -> eviction_count
        self.eviction_reasons: Dict[str, int] = defaultdict(int)  # reason -> count
        
        # Cache efficiency metrics
        self.cache_efficiency_by_turns: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"hit_ratio": 0.0, "avg_reuse_interval": 0.0, "eviction_rate": 0.0, "token_hit_ratio": 0.0}
        )
        # Token-level statistics
        self.lookup_tokens = 0
        self.hit_tokens = 0
        logger.info(f"Cache simulator initialized with {config.cache_size_gb}GB cache")
    
    def _generate_cache_key(self, conversation_id: int, turn_number: int) -> str:
        """Generate a cache key for a conversation turn"""
        return f"conv_{conversation_id}_turn_{turn_number}"
    
    def _get_complete_blocks(self, token_count: int) -> int:
        """Get the number of tokens that form complete blocks"""
        block_size = self.config.block_size
        complete_blocks = token_count // block_size
        return complete_blocks * block_size
    
    def _calculate_entry_size(self, token_count: int) -> int:
        """Calculate the size of a cache entry in bytes, only counting complete blocks"""
        return token_count * self.config.bytes_per_token
    
    def _create_conversation_feature(self, conversation_id: int, turn_number: int, 
                                   current_time: float, previous_interval: float = 0.0) -> ConversationFeature:
        """Create a conversation feature for scoring using actual interval from event"""
        return ConversationFeature(
            conversation_id=conversation_id,
            turn_number=turn_number,
            access_timestamp=current_time,
            previous_interval=previous_interval
        )
    
    def _update_conversation_analysis(self, conversation_id: int, turn_number: int, 
                                    token_count: int, current_time: float, is_hit: bool):
        """Update conversation analysis data"""
        analysis = self.conversation_analysis[conversation_id]
        analysis.conversation_id = conversation_id
        analysis.total_turns = max(analysis.total_turns, turn_number)
        analysis.total_tokens += token_count
        analysis.access_count += 1
        
        if analysis.first_access == 0.0:
            analysis.first_access = current_time
        analysis.last_access = current_time
        
        if is_hit:
            analysis.cache_hits += 1
        else:
            analysis.cache_misses += 1
        
        analysis.turn_lengths.append(token_count)
    
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
            
            # Track eviction analysis
            self.evicted_conversations[entry.conversation_id] += 1
            self.eviction_reasons["cache_full"] += 1
            
            del self.cache[evict_key]
            self.eviction_policy.remove(evict_key, entry)
            self.current_cache_size -= entry.size
            self.evictions += 1
            
            logger.debug(f"Evicted {evict_key}, size: {entry.size}, "
                         f"remaining: {self.current_cache_size}")
    
    def lookup(self, conversation_id: int, turn_number: int, token_count: int, 
               current_time: float) -> Tuple[bool, int]:
        """
        Simulate a cache lookup operation.
        
        Returns:
            Tuple of (is_hit, hit_token_count)
        """
        self.total_requests += 1
        self.lookup_tokens += token_count
        
        # Update conversation turn count
        self.conversation_turns[conversation_id] = max(
            self.conversation_turns[conversation_id], turn_number
        )
        
        # Track requests by turn count
        turn_count = self.conversation_turns[conversation_id]
        self.requests_by_turns[turn_count] += 1
        self.lookup_tokens_by_turns[turn_count] += token_count
        
        # Generate cache key
        lookup_turn_number = max(1, turn_number - 1)
        cache_key = self._generate_cache_key(conversation_id, lookup_turn_number)
        
        # Check for cache hit
        if cache_key in self.cache:
            # Cache hit
            self.cache_hits += 1
            # Use complete blocks count for hit token calculation
            entry = self.cache[cache_key]
            hit_token_count = entry.complete_blocks_count
            self.hit_tokens += hit_token_count
            self.hit_tokens_by_turns[turn_count] += hit_token_count
            
            # Log if hit tokens were limited by complete blocks
            if hit_token_count < entry.token_count:
                logger.debug(f"Hit tokens limited from {entry.token_count} to {hit_token_count} "
                            f"(block_size={self.config.block_size})")
            
            self.hits_by_turns[turn_count] += 1
            
            # Update last access time for reuse interval analysis
            if cache_key in self.last_access_times:
                interval = current_time - self.last_access_times[cache_key]
                self.reuse_intervals[turn_count].append(interval)
                
                # Update conversation analysis
                analysis = self.conversation_analysis[conversation_id]
                analysis.reuse_intervals.append(interval)
            
            self.last_access_times[cache_key] = current_time
            
            # remove the entry in eviction policy
            # This is because the following store will add the entry back with the new size
            # entry = self.cache[cache_key]
            # self.eviction_policy.remove(cache_key, entry)
            # self.current_cache_size -= entry.size
            
            # Update conversation analysis
            self._update_conversation_analysis(conversation_id, turn_number, token_count, current_time, True)
            
            return True, hit_token_count
        else:
            # Cache miss
            self.cache_misses += 1
            
            # Update conversation analysis
            self._update_conversation_analysis(conversation_id, turn_number, token_count, current_time, False)
            
            return False, 0
    
    def store(self, conversation_id: int, turn_number: int, token_count: int, 
              current_time: float, previous_interval: float):
        """Simulate storing an entry in the cache"""
        cache_key = self._generate_cache_key(conversation_id, turn_number)
        
        # Calculate entry size based on complete blocks (for cache management)
        entry_size_bytes = self._calculate_entry_size(token_count)
        
        # Evict if needed
        self._evict_if_needed(entry_size_bytes)
        
        # Create conversation feature
        feature = self._create_conversation_feature(conversation_id,
            turn_number, current_time, previous_interval)
        
        # Create cache entry with total tokens but size based on complete blocks
        entry = CacheEntry(
            conversation_id=conversation_id,
            turn_number=turn_number,
            token_count=token_count,  # Store total tokens
            complete_blocks_count=self._get_complete_blocks(token_count),  # for lookup
            access_timestamp=current_time,
            size=entry_size_bytes,
            feature=feature,
            creation_time=current_time
        )

        if cache_key in self.cache:
            prev_entry = self.cache[cache_key]
            self.eviction_policy.remove(cache_key, prev_entry)
            self.current_cache_size -= prev_entry.size
        
        # Store in cache
        self.cache[cache_key] = entry
        self.current_cache_size += entry_size_bytes
        
        # Add to eviction policy
        self.eviction_policy.add(cache_key, entry)
    
    def _calculate_cache_efficiency_metrics(self):
        """Calculate detailed cache efficiency metrics"""
        for turn_count in self.requests_by_turns:
            hits = self.hits_by_turns.get(turn_count, 0)
            requests = self.requests_by_turns[turn_count]
            hit_ratio = hits / requests if requests > 0 else 0.0
            
            # Calculate token hit ratio by turn count
            lookup_tokens = self.lookup_tokens_by_turns.get(turn_count, 0)
            hit_tokens = self.hit_tokens_by_turns.get(turn_count, 0)
            token_hit_ratio = hit_tokens / lookup_tokens if lookup_tokens > 0 else 0.0
            
            # Calculate average reuse interval
            intervals = self.reuse_intervals.get(turn_count, [])
            avg_reuse_interval = sum(intervals) / len(intervals) if intervals else 0.0
            
            # Calculate eviction rate (simplified)
            eviction_rate = 0.0  # Would need more detailed tracking
            
            self.cache_efficiency_by_turns[turn_count] = {
                "hit_ratio": hit_ratio,
                "avg_reuse_interval": avg_reuse_interval,
                "eviction_rate": eviction_rate,
                "token_hit_ratio": token_hit_ratio
            }
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics with detailed analysis"""
        self._calculate_cache_efficiency_metrics()
        
        hit_ratio = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        
        # Calculate hit ratios by turn count
        hit_ratios_by_turns = {}
        for turn_count in self.requests_by_turns:
            hits = self.hits_by_turns.get(turn_count, 0)
            requests = self.requests_by_turns[turn_count]
            hit_ratios_by_turns[turn_count] = hits / requests if requests > 0 else 0
        
        # Calculate token hit ratios by turn count
        token_hit_ratios_by_turns = {}
        for turn_count in self.lookup_tokens_by_turns:
            lookup_tokens = self.lookup_tokens_by_turns.get(turn_count, 0)
            hit_tokens = self.hit_tokens_by_turns.get(turn_count, 0)
            token_hit_ratios_by_turns[turn_count] = hit_tokens / lookup_tokens if lookup_tokens > 0 else 0
        
        # Calculate average reuse intervals by turn count
        avg_reuse_intervals = {}
        for turn_count, intervals in self.reuse_intervals.items():
            if intervals:
                avg_reuse_intervals[turn_count] = sum(intervals) / len(intervals)
        
        # Conversation pattern analysis
        conversation_stats = {}
        for conv_id, analysis in self.conversation_analysis.items():
            if analysis.access_count > 0:
                conversation_stats[conv_id] = {
                    "total_turns": analysis.total_turns,
                    "total_tokens": analysis.total_tokens,
                    "access_count": analysis.access_count,
                    "hit_ratio": analysis.cache_hits / analysis.access_count if analysis.access_count > 0 else 0,
                    "avg_turn_length": sum(analysis.turn_lengths) / len(analysis.turn_lengths) if analysis.turn_lengths else 0,
                    "avg_reuse_interval": sum(analysis.reuse_intervals) / len(analysis.reuse_intervals) if analysis.reuse_intervals else 0,
                    "duration": analysis.last_access - analysis.first_access if analysis.first_access > 0 else 0
                }
        
        return {
            "basic_stats": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_ratio": hit_ratio,
                "evictions": self.evictions,
                "current_cache_size_gb": self.current_cache_size / (1024**3),
                "cache_utilization": self.current_cache_size / self.cache_size_bytes,
                "eviction_policy": self.config.eviction_policy,
                "lookup_tokens": self.lookup_tokens,
                "hit_tokens": self.hit_tokens,
                "token_hit_ratio": self.hit_tokens / self.lookup_tokens if self.lookup_tokens > 0 else 0.0
            },
            "turn_analysis": {
                "hit_ratios_by_turns": hit_ratios_by_turns,
                "token_hit_ratios_by_turns": token_hit_ratios_by_turns,
                "avg_reuse_intervals": avg_reuse_intervals,
                "requests_by_turns": dict(self.requests_by_turns),
                "hits_by_turns": dict(self.hits_by_turns),
                "lookup_tokens_by_turns": dict(self.lookup_tokens_by_turns),
                "hit_tokens_by_turns": dict(self.hit_tokens_by_turns),
                "cache_efficiency_by_turns": dict(self.cache_efficiency_by_turns)
            },
            "conversation_analysis": conversation_stats,
            "eviction_analysis": {
                "evicted_conversations": dict(self.evicted_conversations),
                "eviction_reasons": dict(self.eviction_reasons)
            }
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
        
        # Track conversation state for proper turn_number mapping
        self.conversation_turn_state: Dict[int, int] = defaultdict(int)
        
        # total input lengths including the most recent send
        self.conversation_input_lengths: Dict[int, int] = {}
        
        # Track conversation timestamps for interval calculation
        self.conversation_last_timestamps: Dict[int, float] = {}
        
        logger.info(f"CacheSimulatorReplay initialized with {config.eviction_policy} policy")
    
    def handle_send_event(self, event: Event):        
        """Handle Send event - perform lookup simulation"""
        logger.info(f"Processing Send event: conv_id={event.conversation_id}, "
                   f"turn_number={event.turn_number}, input_tokens={event.input_tokens}")
        
        # Update conversation turn state
        self.conversation_turn_state[event.conversation_id] = event.turn_number
        
        # Store input length for use in done event
        self.conversation_input_lengths[event.conversation_id] = event.input_tokens
        token_count = event.input_tokens
        
        # Calculate interval from previous event for this conversation
        current_timestamp = event.timestamp if event.timestamp > 0 else time.time()
        previous_interval = 0.0
        if event.conversation_id in self.conversation_last_timestamps:
            previous_interval = current_timestamp - self.conversation_last_timestamps[event.conversation_id]
            logger.info(f"  Interval for conversation {event.conversation_id}: {previous_interval:.2f}s")
        else:
            logger.info(f"  First event for conversation {event.conversation_id}, no interval")
        
        # Update last timestamp for this conversation
        self.conversation_last_timestamps[event.conversation_id] = current_timestamp
        
        logger.info(f"  Cumulative tokens for conversation {event.conversation_id}: {token_count} (added {event.input_tokens})")
        
        # Simulate cache lookup with event and interval
        hit, hit_tokens = self.simulator.lookup(
            event.conversation_id, 
            event.turn_number, 
            token_count, 
            current_timestamp
        )
        
        if hit:
            logger.info(f"  ✓ Cache HIT for conversation {event.conversation_id}, turn {event.turn_number} (hit tokens: {hit_tokens})")
        else:
            logger.info(f"  ✗ Cache MISS for conversation {event.conversation_id}, turn {event.turn_number}")
        
        self.processed_requests += 1
    
    def handle_done_event(self, event: Event):
        """Handle Done event - perform store simulation"""
        # Skip if the conversation was aborted
        if event.conversation_id in self.aborted_conversations:
            logger.warning(f"Skipping Done event for aborted conversation {event.conversation_id}")
            return
        
        # Get the current turn_number for this conversation
        current_turn_number = self.conversation_turn_state[event.conversation_id]
        if current_turn_number == 0:
            # If no turn_number tracked, this is likely the first turn
            current_turn_number = 1
        
        # Get the input length from the corresponding send event
        input_tokens = self.conversation_input_lengths.get(event.conversation_id, 0)
        output_tokens = event.generated_tokens if event.generated_tokens else 0
        
        token_count = input_tokens + output_tokens
        
        logger.info(f"  Cumulative tokens for conversation {event.conversation_id}: {token_count} (added {output_tokens})")
        
        # Simulate cache store with event
        current_timestamp = event.timestamp if event.timestamp > 0 else time.time()
        self.simulator.store(
            event.conversation_id,
            current_turn_number,  # Use the tracked turn_number
            token_count,  # Pass cumulative token count
            current_timestamp,
            0.0  # No interval for store operations
        )
        
        logger.info(f"  ✓ Stored conversation {event.conversation_id} in cache (turn {current_turn_number}, "
                   f"tokens: {token_count}, input: {input_tokens}, output: {output_tokens})")
    
    def replay_log_events(self, events: List[Event]):
        """Replay all log events and simulate cache operations"""
        logger.info(f"Starting replay of {len(events)} events")
        
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
        for turn_count in sorted(turn_analysis["hit_ratios_by_turns"].keys()):
            hit_ratio = turn_analysis["hit_ratios_by_turns"][turn_count]
            requests = turn_analysis["requests_by_turns"].get(turn_count, 0)
            hits = turn_analysis["hits_by_turns"].get(turn_count, 0)
            avg_interval = turn_analysis["avg_reuse_intervals"].get(turn_count, 0)
            logger.info(f"    {turn_count} turns: {hit_ratio:.2%} ({hits}/{requests} requests, "
                       f"avg reuse: {avg_interval:.2f}s)")
        
        logger.info(f"  Token Hit Ratios by Turn Count:")
        for turn_count in sorted(turn_analysis["token_hit_ratios_by_turns"].keys()):
            token_hit_ratio = turn_analysis["token_hit_ratios_by_turns"][turn_count]
            lookup_tokens = turn_analysis["lookup_tokens_by_turns"].get(turn_count, 0)
            hit_tokens = turn_analysis["hit_tokens_by_turns"].get(turn_count, 0)
            logger.info(f"    {turn_count} turns: {token_hit_ratio:.2%} ({hit_tokens}/{lookup_tokens} tokens)")
        
        # Conversation pattern analysis
        conv_analysis = stats["conversation_analysis"]
        if conv_analysis:
            logger.info(f"\nConversation Pattern Analysis:")
            logger.info(f"  Total conversations: {len(conv_analysis)}")
            
            # Calculate summary statistics
            turn_counts = [data["total_turns"] for data in conv_analysis.values()]
            hit_ratios = [data["hit_ratio"] for data in conv_analysis.values()]
            reuse_intervals = [data["avg_reuse_interval"] for data in conv_analysis.values() if data["avg_reuse_interval"] > 0]
            
            if turn_counts:
                logger.info(f"  Average turns per conversation: {sum(turn_counts)/len(turn_counts):.1f}")
                logger.info(f"  Max turns in a conversation: {max(turn_counts)}")
                logger.info(f"  Min turns in a conversation: {min(turn_counts)}")
            
            if hit_ratios:
                logger.info(f"  Average hit ratio per conversation: {sum(hit_ratios)/len(hit_ratios):.2%}")
                logger.info(f"  Best conversation hit ratio: {max(hit_ratios):.2%}")
                logger.info(f"  Worst conversation hit ratio: {min(hit_ratios):.2%}")
            
            if reuse_intervals:
                logger.info(f"  Average reuse interval: {sum(reuse_intervals)/len(reuse_intervals):.2f}s")
                logger.info(f"  Max reuse interval: {max(reuse_intervals):.2f}s")
                logger.info(f"  Min reuse interval: {min(reuse_intervals):.2f}s")
        
        # Eviction analysis
        eviction_analysis = stats["eviction_analysis"]
        if eviction_analysis["evicted_conversations"]:
            logger.info(f"\nEviction Analysis:")
            logger.info(f"  Total conversations evicted: {len(eviction_analysis['evicted_conversations'])}")
            
            eviction_counts = list(eviction_analysis["evicted_conversations"].values())
            if eviction_counts:
                logger.info(f"  Average evictions per conversation: {sum(eviction_counts)/len(eviction_counts):.1f}")
                logger.info(f"  Max evictions for a conversation: {max(eviction_counts)}")
        
        # Basic statistics
        basic_stats = stats["basic_stats"]
        logger.info("\nBasic Statistics:")
        logger.info(f"Eviction Policy: {basic_stats['eviction_policy']}")
        logger.info(f"Total Requests: {basic_stats['total_requests']}")
        logger.info(f"Cache Hits: {basic_stats['cache_hits']}")
        logger.info(f"Cache Misses: {basic_stats['cache_misses']}")
        logger.info(f"Overall Hit Ratio: {basic_stats['hit_ratio']:.2%}")
        logger.info(f"Evictions: {basic_stats['evictions']}")
        logger.info(f"Current Cache Size: {basic_stats['current_cache_size_gb']:.2f} GB")
        logger.info(f"Block Size: {self.config.block_size}")
        logger.info(f"Total Lookup Tokens: {basic_stats['lookup_tokens']}")
        logger.info(f"Total Hit Tokens: {basic_stats['hit_tokens']}")
        logger.info(f"Token Hit Ratio: {basic_stats['token_hit_ratio']:.2%}")
        
        logger.info("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cache simulation for eviction analysis')
    parser.add_argument('--log-file', default='logs/client_32.0gb_1.0rps_test_data.log',
                       help='Path to the client log file')
    parser.add_argument('--cache-size', type=float, default=64.0,
                       help='Cache size in GB')
    parser.add_argument('--eviction-policy', choices=['lru', 'conversation_aware'], 
                       default='lru',
                       help='Eviction policy to use')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process')
    parser.add_argument('--chunk-size', type=int, default=256,
                       help='Chunk size (default: 256)')
    parser.add_argument('--block-size', type=int, default=256,
                       help='Block size for cache entries (default: 256)')
    parser.add_argument('--max-context-length', type=int, default=16384,
                       help='Maximum context length in tokens (default: 16384)')
    parser.add_argument('--model-name', default='Qwen/Qwen3-8B-FP8')
    
    args = parser.parse_args()
    # Create configuration
    model_name = args.model_name
    _, bytes_per_token = get_model_config_and_bytes_per_token(model_name)
    chat_template_overhead = get_chat_template_overhead(model_name)
    print(f"Bytes per token: {bytes_per_token}")
    print(f"Chat template overhead: {chat_template_overhead} tokens")
    
    print("Cache Simulator for Eviction Analysis")
    print("="*80)
    print(f"Log file: {args.log_file}")
    print(f"Cache size: {args.cache_size} GB")
    print(f"Eviction policy: {args.eviction_policy}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Block size: {args.block_size}")
    print(f"Max context length: {args.max_context_length}")
    print("="*80, flush=True)
    
    # Parse log file
    parser = LogParser(args.log_file)
    if "client" in args.log_file:
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

    config = SimulatorConfig(
        cache_size_gb=args.cache_size,
        chunk_size=args.chunk_size,
        block_size=args.block_size,
        max_context_length=args.max_context_length,
        eviction_policy=args.eviction_policy,
        bytes_per_token=bytes_per_token,
        chat_template_overhead=chat_template_overhead
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