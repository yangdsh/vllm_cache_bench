#!/usr/bin/env python3
"""
Oracle Scorer for Cache Eviction Policies

This module provides an "oracle" scorer that uses actual future reuse intervals
as eviction scores. This serves as an upper bound baseline since it has perfect
knowledge of when conversations will be reused next.
"""

import logging
from typing import Optional, List, Dict

from simulation.conversation_tracker import ConversationFeature
from simulation.common import Event

logger = logging.getLogger(__name__)


class OracleScorer:
    """Oracle scorer that uses actual future reuse intervals as eviction scores"""
    
    def __init__(self):
        self.current_time = 0.0
        self.future_reuse_map: Dict[str, float] = {}  # conversation_id -> next_reuse_time
        self.is_initialized = False
        
    def set_current_time(self, current_time: float):
        """Set the current simulation time"""
        self.current_time = current_time
    
    def initialize_from_events(self, events: List[Event]) -> None:
        """Initialize the oracle with future reuse information from events"""
        logger.info(f"Initializing Oracle scorer from {len(events)} events...")
        
        # Build a map of conversation reuse times
        conversation_events = {}  # conversation_id -> list of timestamps
        
        # Get the earliest timestamp to use as reference
        start_time = min(event.timestamp for event in events if event.event_type == "send")
        
        for event in events:
            if event.event_type == "send":
                if event.conversation_id not in conversation_events:
                    conversation_events[event.conversation_id] = []
                # Convert to relative time
                relative_time = event.timestamp - start_time
                conversation_events[event.conversation_id].append(relative_time)
        
        # Sort timestamps for each conversation
        for conv_id in conversation_events:
            conversation_events[conv_id].sort()
        
        self.conversation_events = conversation_events
        self.is_initialized = True
        
        logger.info(f"Oracle initialized with {len(conversation_events)} conversations")
        
        # Log some statistics
        reuse_counts = [len(timestamps) for timestamps in conversation_events.values()]
        logger.info(f"Conversation reuse stats: min={min(reuse_counts)}, max={max(reuse_counts)}, "
                   f"mean={sum(reuse_counts)/len(reuse_counts):.2f}")
    
    def _get_next_reuse_time(self, conversation_id: str, current_time: float) -> Optional[float]:
        """Get the next reuse time for a conversation after current_time"""
        if not self.is_initialized:
            logger.warning("Oracle scorer not initialized with events")
            return None
            
        # Convert string conversation_id to int for lookup
        conv_id_int = int(conversation_id)
            
        if conv_id_int not in self.conversation_events:
            return None
            
        timestamps = self.conversation_events[conv_id_int]
        
        # Find the next timestamp after current_time
        for timestamp in timestamps:
            if timestamp > current_time:
                return timestamp
                
        return None  # No future reuse
    
    def calculate_score(self, feature: Optional[ConversationFeature]) -> float:
        """Calculate eviction score using oracle knowledge of future reuse"""
        if feature is None:
            return -1e8
            
        if not self.is_initialized:
            logger.warning("Oracle scorer not initialized, returning default score")
            return -1e8
        
        # Get next reuse time for this conversation
        next_reuse_time = self._get_next_reuse_time(feature.conversation_id, self.current_time)
        
        if next_reuse_time is None:
            # No future reuse - assign maximum eviction score (most likely to evict)
            return -1e8
        
        # Calculate time interval until next reuse
        reuse_interval = next_reuse_time - self.current_time
        
        # Ensure non-negative (should always be positive due to _get_next_reuse_time logic)
        reuse_interval = max(0.0, reuse_interval)
        
        # Transform the interval to be negative
        return -reuse_interval
    
    def cache_admission(self, feature: Optional[ConversationFeature]) -> bool:
        """Determine if a conversation should be admitted to cache"""
        return feature is not None


def create_oracle_scorer_from_events(events: List[Event]) -> OracleScorer:
    """
    Create and initialize an Oracle scorer from events
    
    Args:
        events: List of log events to extract future reuse information from
        
    Returns:
        Initialized OracleScorer
    """
    logger.info(f"Creating Oracle scorer from {len(events)} events")
    
    if not events:
        raise ValueError("No events provided for oracle initialization")
    
    scorer = OracleScorer()
    scorer.initialize_from_events(events)
    
    return scorer
