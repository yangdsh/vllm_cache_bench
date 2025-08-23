#!/usr/bin/env python3
"""Conversation state management for cache simulator"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from collections import defaultdict


class ConversationFeature:

    # the request is a duplicate of an existing request
    is_duplicate: bool = False

    # Base features (time & length)
    access_timestamp: float
    previous_interval: float = 0.0
    CustomerQueryLength: int = 0
    GenerativeModelResponseLength: int = 0

    # Engineered features
    turn_number: int
    cumulative_time: float = 0.0
    avg_interval_so_far: float = 0.0
    avg_query_length_so_far: float = 0.0
    avg_response_length_so_far: float = 0.0

    # Metadata
    conversation_id: str
    request_id: str = ""
    total_turns: int = 0

    def __init__(self, conversation_id: str, turn_number: int, access_timestamp: float,
                 previous_interval: float = 0.0, CustomerQueryLength: int = 0,
                 GenerativeModelResponseLength: int = 0, cumulative_time: float = 0.0,
                 avg_interval_so_far: float = 0.0, avg_query_length_so_far: float = 0.0,
                 avg_response_length_so_far: float = 0.0, request_id: str = "",
                 total_turns: int = 0, is_duplicate: bool = False):
        self.conversation_id = conversation_id
        self.turn_number = turn_number
        self.access_timestamp = access_timestamp
        self.previous_interval = previous_interval
        self.CustomerQueryLength = CustomerQueryLength
        self.GenerativeModelResponseLength = GenerativeModelResponseLength
        self.cumulative_time = cumulative_time
        self.avg_interval_so_far = avg_interval_so_far
        self.avg_query_length_so_far = avg_query_length_so_far
        self.avg_response_length_so_far = avg_response_length_so_far
        self.request_id = request_id
        self.total_turns = total_turns
        self.is_duplicate = is_duplicate
    
    def __str__(self) -> str:
        """Return a string representation of the feature"""
        return (f"ConversationFeature(conv_{self.conversation_id}_{self.turn_number}, "
                f"query_len={self.CustomerQueryLength},  resp_len={self.GenerativeModelResponseLength}, "
                f"previous_interval={self.previous_interval:.2f}s)")

@dataclass
class ConversationState:
    """Internal conversation state tracking"""
    
    conversation_id: int
    turn_number: int = 0
    cumulative_time: float = 0.0
    query_lengths: List[int] = field(default_factory=list)
    response_lengths: List[int] = field(default_factory=list)
    reuse_intervals: List[float] = field(default_factory=list)
    last_access_time: float = 0.0
    context_length: int = 0

    def get_state_of_turn(self, turn_number: int):
        if turn_number == 0:
            return 0, 0, 0
        
        # Get query length
        query_len = self.query_lengths[turn_number-1] if turn_number-1 < len(self.query_lengths) else 0
        
        # Get response length (turn_number-1 because current response_lengths is not available yet)
        resp_len = self.response_lengths[turn_number-1] if turn_number-1 < len(self.response_lengths) else 0
        
        # Get reuse interval (turn_number-1 because first interval is for turn 2)
        # Turn 1: no interval (index would be -1), Turn 2: interval[0], Turn 3: interval[1], etc.
        reuse_interval = 0
        if turn_number >= 2 and turn_number-2 < len(self.reuse_intervals):
            reuse_interval = self.reuse_intervals[turn_number-2]
        
        return query_len, resp_len, reuse_interval
                
    
    def update_state(self, turn_number, current_time, query_len,
        response_len = 0, source: str = "unknown"):
        """Update conversation state with new turn information"""
        self.turn_number = max(self.turn_number, turn_number)
        
        # Calculate reuse interval from previous access
        reuse_interval = 0.0
        if self.last_access_time > 0:
            reuse_interval = current_time - self.last_access_time
            # Only append intervals on send events to measure Done->Send gaps
            if source == "send":
                self.reuse_intervals.append(reuse_interval)
                self.cumulative_time += reuse_interval
        
        self.last_access_time = current_time
        
        # Store lengths
        if source == "send":
            self.query_lengths.append(query_len)
        if source == "done":
            self.response_lengths.append(response_len)
    
    def get_average_reuse_interval(self) -> float:
        """Get the average reuse interval for all turns"""
        if not self.reuse_intervals:
            return 0.0
        return sum(self.reuse_intervals) / len(self.reuse_intervals)
    
    def get_average_query_length(self) -> float:
        """Get the average query length"""
        if not self.query_lengths:
            return 0.0
        return sum(self.query_lengths) / len(self.query_lengths)
    
    def get_average_response_length(self) -> float:
        """Get the average response length"""
        if not self.response_lengths:
            return 0.0
        return sum(self.response_lengths) / len(self.response_lengths)
    
    def get_average_query_length_until(self, turn_number: int) -> float:
        """Get the average query length up to the specified turn"""
        if not self.query_lengths:
            return 0.0
        # Take query lengths up to the specified turn
        relevant_queries = self.query_lengths[:turn_number]
        if not relevant_queries:
            return 0.0
        return sum(relevant_queries) / len(relevant_queries)
    
    def get_average_response_length_until(self, turn_number: int) -> float:
        """Get the average response length up to the specified turn"""
        if not self.response_lengths:
            return 0.0
        # Take response lengths up to the specified turn
        relevant_responses = self.response_lengths[:turn_number]
        if not relevant_responses:
            return 0.0
        return sum(relevant_responses) / len(relevant_responses)

    def get_average_reuse_interval_until(self, turn_number: int) -> float:
        """Get the average reuse interval up to the specified turn"""
        if not self.reuse_intervals:
            return 0.0
        # Take intervals up to the specified turn (turn 2 uses first interval, etc.)
        relevant_intervals = self.reuse_intervals[:turn_number-1] if turn_number > 1 else []
        if not relevant_intervals:
            return 0.0
        return sum(relevant_intervals) / len(relevant_intervals)
    

@dataclass
class ConversationTracker:
    """Tracks conversation state and creates conversation features"""
    
    def __init__(self):
        # Track conversation states
        self.conversation_states: Dict[int, 'ConversationState'] = {}
    
    # used in send event
    def set_context_length(self, conversation_id: int, input_tokens: int):
        """Update input length for a conversation"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = \
                ConversationState(conversation_id=conversation_id)
        self.conversation_states[conversation_id].context_length = input_tokens
    
    # used in done event
    def get_context_length(self, conversation_id: int) -> int:
        """Get the current input length for a conversation"""
        if conversation_id not in self.conversation_states:
            return 0
        return self.conversation_states[conversation_id].context_length
    
    def calculate_query_len(self, conversation_id: int, input_tokens: int) -> int:
        """Calculate request length for scoring purposes"""
        if conversation_id not in self.conversation_states:
            # First turn - the full input is the request length
            return input_tokens
        return input_tokens - self.conversation_states[conversation_id].context_length
    
    def update_conversation_feature(self, conversation_id: int, turn_number: int, 
        current_time: float, query_len: int = 0, response_len: int = 0,
        source: str = "unknown") -> Optional[ConversationFeature]:
        """Get the conversation feature for a conversation with average values"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = \
                ConversationState(conversation_id=conversation_id)
        
        state = self.conversation_states[conversation_id]
        
        # Update the conversation state with this turn
        state.update_state(
            turn_number, current_time, query_len, response_len, source=source
        )
        
        return self.get_conversation_feature(conversation_id)
    
    def get_conversation_feature(self, conversation_id: int) -> Optional[ConversationFeature]:
        """Get the conversation feature for a conversation"""
        if conversation_id not in self.conversation_states:
            return None
        state = self.conversation_states[conversation_id]
        # Get specific values for the current turn
        query_len, resp_len, reuse_interval = state.get_state_of_turn(state.turn_number)
        
        feature = ConversationFeature(
            conversation_id=str(conversation_id),
            turn_number=state.turn_number,
            access_timestamp=state.last_access_time,
            previous_interval=float(reuse_interval),
            CustomerQueryLength=int(query_len),
            GenerativeModelResponseLength=int(resp_len),
            cumulative_time=state.cumulative_time,
            avg_interval_so_far=state.get_average_reuse_interval(),
            avg_query_length_so_far=state.get_average_query_length(),
            avg_response_length_so_far=state.get_average_response_length(),
            total_turns=state.turn_number
        )
        
        return feature
    
    def get_conversation_state(self, conversation_id: int) -> Optional[ConversationState]:
        """Get the conversation state for a conversation"""
        if conversation_id not in self.conversation_states:
            return None
        return self.conversation_states[conversation_id]
