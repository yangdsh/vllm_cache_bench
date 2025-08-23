#!/usr/bin/env python3
"""
Conversation Scorer for Cache Eviction Policies

This module provides scoring functionality for conversation-aware cache eviction.
It includes configuration and calculation of eviction scores based on conversation
features like recency, activity patterns, and turn characteristics.
"""

import math
from dataclasses import dataclass
from typing import Optional
import logging

from simulation.conversation_tracker import ConversationFeature

logger = logging.getLogger(__name__)

@dataclass
class RuleScorerConfig:
    """Configuration for scoring based only on request length and reuse interval."""
    length_norm: float = 100.0           # Normalization factor for request length (tokens)
    reuse_reference: float = 2000.0      # Reference seconds for reuse interval decay
    base_score: float = 0.0              # Base score for all entries
    length_weight: float = 0.5           # Weight for normalized request length
    reuse_weight: float = 0.5            # Weight for reuse interval score

class RuleScorer:
    """Conversation scorer based on request length and reuse interval only"""
    
    def __init__(self, config: Optional[RuleScorerConfig] = None):
        self.config = config or RuleScorerConfig()
        self.current_time = 0.0
        
    def set_current_time(self, current_time: float):
        """Set the current simulation time"""
        self.current_time = current_time
    
    def cache_admission(self, feature: Optional[ConversationFeature]) -> bool:
        """All turns are cacheable when features are available."""
        return feature is not None
        
    def calculate_score(
        self, 
        feature: Optional[ConversationFeature]
    ) -> float:
        """
        Calculate eviction score using only:
        - Request length (query_len + respond_len) normalized by length_norm
        - Reuse interval transformed via exponential decay with reuse_reference
        """
        if feature is None:
            return 0.0

        # Request length component [0, 1]
        request_length = float(feature.CustomerQueryLength)
        length_score = min(request_length / self.config.length_norm, 1.0)

        # Reuse interval component in [0, 1]; lower interval => higher score
        if feature.previous_interval == 0.0:
            reuse_score = 0.0
        else:
            reuse_score = math.exp(-feature.previous_interval / self.config.reuse_reference)

        # Weighted combination
        return (
            self.config.base_score +
            self.config.length_weight * length_score +
            self.config.reuse_weight * reuse_score
        )
