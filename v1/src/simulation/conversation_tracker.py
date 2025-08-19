#!/usr/bin/env python3
"""Conversation state management for cache simulator"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class ConversationFeature:
    """Simplified conversation feature for scoring"""
    conversation_id: int
    turn_number: int
    access_timestamp: float
    previous_interval: float = 0.0
    request_len: int = 0
    respond_len: int = 0
    
    def __str__(self) -> str:
        """Return a string representation of the feature"""
        return (f"ConversationFeature(conv_{self.conversation_id}_{self.turn_number}, "
                f"req_len={self.request_len},  resp_len={self.respond_len}, "
                f"prev_interval={self.previous_interval:.2f}s)")

@dataclass
class ConversationState:
    """Internal conversation state tracking"""
    
    conversation_id: int
    turn_number: int = 0
    last_turn_time: float = 0.0
    cumulative_time: float = 0.0
    query_lengths: List[int] = field(default_factory=list)
    response_lengths: List[int] = field(default_factory=list)
    turn_intervals: List[float] = field(default_factory=list)
    reuse_intervals: List[float] = field(default_factory=list)
    last_access_time: float = 0.0
    last_input_tokens: int = 0
    
    def update_turn(self, turn_number: int, current_time: float, request_len: int, response_len: int = 0):
        """Update conversation state with new turn information"""
        self.turn_number = max(self.turn_number, turn_number)
        
        # Calculate interval from last turn
        if self.last_turn_time > 0:
            interval = current_time - self.last_turn_time
            self.turn_intervals.append(interval)
            self.cumulative_time += interval
        
        self.last_turn_time = current_time
        self.last_access_time = current_time
        
        # Store lengths
        if request_len > 0:
            self.query_lengths.append(request_len)
        if response_len > 0:
            self.response_lengths.append(response_len)
    
    def add_reuse_interval(self, interval: float):
        """Add a reuse interval to the conversation history"""
        if interval > 0:
            self.reuse_intervals.append(interval)
    
    def get_average_interval(self) -> float:
        """Get the average interval between turns"""
        if not self.turn_intervals:
            return 0.0
        return sum(self.turn_intervals) / len(self.turn_intervals)
    
    def get_average_reuse_interval(self) -> float:
        """Get the average reuse interval"""
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

@dataclass
class ConversationTracker:
    """Tracks conversation state and creates conversation features"""
    
    def __init__(self):
        # Track conversation states
        self.conversation_states: Dict[int, 'ConversationState'] = {}
    
    def update_turn_number(self, conversation_id: int, turn_number: int):
        """Update the turn number for a conversation"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = ConversationState(conversation_id=conversation_id)
        
        state = self.conversation_states[conversation_id]
        state.turn_number = max(state.turn_number, turn_number)
    
    def get_turn_number(self, conversation_id: int) -> int:
        """Get the current turn number for a conversation"""
        if conversation_id not in self.conversation_states:
            return 0
        return self.conversation_states[conversation_id].turn_number
    
    def update_input_length(self, conversation_id: int, input_tokens: int):
        """Update input length for a conversation"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = ConversationState(conversation_id=conversation_id)
        
        state = self.conversation_states[conversation_id]
        state.last_input_tokens = input_tokens
    
    def get_input_length(self, conversation_id: int) -> int:
        """Get the current input length for a conversation"""
        if conversation_id not in self.conversation_states:
            return 0
        return self.conversation_states[conversation_id].last_input_tokens
    
    def get_request_len(self, conversation_id: int, input_tokens: int) -> int:
        """Calculate request length (difference from previous input)"""
        if conversation_id not in self.conversation_states:
            return 0
        state = self.conversation_states[conversation_id]
        if state.last_input_tokens == 0:
            return 0
        return input_tokens - state.last_input_tokens
    
    def update_response_length(self, conversation_id: int, response_len: int):
        """Update the conversation state with response length"""
        if conversation_id in self.conversation_states:
            state = self.conversation_states[conversation_id]
            if response_len > 0:
                state.response_lengths.append(response_len)
    
    def calculate_interval(self, conversation_id: int, current_timestamp: float) -> float:
        """Calculate interval from previous event and update timestamp"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = ConversationState(conversation_id=conversation_id)
        
        state = self.conversation_states[conversation_id]
        previous_interval = 0.0
        
        if state.last_access_time > 0:
            previous_interval = current_timestamp - state.last_access_time
        
        state.last_access_time = current_timestamp
        return previous_interval
    
    def get_conversation_feature(self, conversation_id: int, turn_number: int, 
                               current_time: float, previous_interval: float = 0.0,
                               request_len: int = 0) -> Optional[ConversationFeature]:
        """Get the conversation feature for a conversation with average values"""
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = ConversationState(conversation_id=conversation_id)
        
        state = self.conversation_states[conversation_id]
        
        # Update the conversation state with this turn
        state.update_turn(turn_number, current_time, request_len)
        
        # Add reuse interval if meaningful
        if previous_interval > 0:
            state.add_reuse_interval(previous_interval)
        
        # Create feature with average values
        feature = ConversationFeature(
            conversation_id=conversation_id,
            turn_number=turn_number,
            access_timestamp=current_time,
            previous_interval=state.get_average_interval(),
            request_len=int(state.get_average_query_length())
        )
        
        return feature
