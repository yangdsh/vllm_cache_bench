#!/usr/bin/env python3
"""
LightGBM-based Conversation Scorer for Cache Eviction Policies

This module provides LightGBM-based scoring functionality for conversation-aware cache eviction.
It uses machine learning to predict eviction scores based on conversation features.
"""

import os
import pickle
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import logging
import lightgbm as lgb

from simulation.conversation_tracker import ConversationFeature
from simulation.common import Event

logger = logging.getLogger(__name__)

@dataclass
class LightGBMScorerConfig:
    """Configuration for LightGBM-based scoring"""
    model_dir: str = "models/"
    n_estimators: int = 32
    learning_rate: float = 0.1
    num_leaves: int = 31
    verbose: int = -1
    mode: str = "ranking"  # "ranking", "regression", or "classification"

class LightGBMScorer:
    """LightGBM-based conversation scorer for cache eviction"""
    
    def __init__(self, config: Optional[LightGBMScorerConfig] = None):
        self.config = config or LightGBMScorerConfig()
        self.model: Optional[lgb.Booster] = None
        self.is_trained = False
        self.current_time = 0.0
        self.feature_names = [
            'turn_number', 'previous_interval', 
            'CustomerQueryLength', 'GenerativeModelResponseLength', 'cumulative_time',
            'avg_interval_so_far', 'avg_query_length_so_far', 'avg_response_length_so_far',
        ]
        
        # Ensure model directory exists
        os.makedirs(self.config.model_dir, exist_ok=True)
    
    def set_current_time(self, current_time: float):
        """Set the current simulation time"""
        self.current_time = current_time
    
    def _extract_features(self, feature: ConversationFeature) -> np.ndarray:
        """Extract feature vector from ConversationFeature"""
        
        features = np.array([
            float(feature.turn_number),
            float(feature.previous_interval),
            float(feature.CustomerQueryLength),
            float(feature.GenerativeModelResponseLength),
            float(feature.cumulative_time),
            float(feature.avg_interval_so_far),
            float(feature.avg_query_length_so_far),
            float(feature.avg_response_length_so_far),
        ])
        
        return features
    
    def prepare_training_data(self, events: List[Event]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare training data from log events for machine learning
        
        Args:
            events: List of log events to extract features from
            
        Returns:
            Tuple of (features, labels, group_sizes) where:
            - features: feature vectors
            - labels: target values (ranking scores, regression values, or classification labels)
            - group_sizes: sizes of sequential groups for LightGBM ranking (None for regression/classification)
        """
        from .conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        features_list = []
        labels_list = []
        
        logger.info(f"Preparing {self.config.mode} training data from {len(events)} events...")
        
        # First pass: collect all done events with features
        done_events_with_features = []
        send_event_map = {}  # Map (conversation_id, turn_number) -> input_tokens from send event
        
        for i, event in enumerate(events):
            if event.event_type not in ["send", "done"]:
                continue
                
            # Update conversation tracker
            if event.event_type == "send":
                # Store the input_tokens for this conversation/turn  
                send_event_map[(event.conversation_id, event.turn_number)] = event.input_tokens
                # Don't set context_length yet - do it after query_len calculation
                # Process send event to create intervals
                query_len = tracker.calculate_query_len(event.conversation_id, event.input_tokens)
                feature = tracker.update_conversation_feature(
                    event.conversation_id,
                    event.turn_number,
                    event.timestamp,
                    query_len=query_len,
                    response_len=0,  # No response yet
                    source="send"
                )
            elif event.event_type == "done":
                # Get the input_tokens from the corresponding send event
                send_input_tokens = send_event_map.get((event.conversation_id, event.turn_number), 0)
                # Calculate query_len using the original logic but with correct input_tokens
                query_len = tracker.calculate_query_len(event.conversation_id, send_input_tokens)
                response_len = event.generated_tokens
                # NOW set the context length for this turn (input + response for accumulating context)
                tracker.set_context_length(event.conversation_id, send_input_tokens + response_len)
                
                feature = tracker.update_conversation_feature(
                    event.conversation_id,
                    event.turn_number, 
                    event.timestamp,
                    query_len=query_len,
                    response_len=response_len,
                    source="done"
                )
                
                if feature is not None:
                    done_events_with_features.append((i, event, feature))
        
        logger.info(f"Found {len(done_events_with_features)} done events with features")
        
        # Second pass: generate features and time-to-next-reuse labels
        for event_idx, (i, event, feature) in enumerate(done_events_with_features):
            # Extract features
            feature_vector = self._extract_features(feature)
            features_list.append(feature_vector)
            
            # Calculate time to next reuse of this conversation
            time_to_next_reuse = float('inf')  # Default: no future reuse
            current_time = event.timestamp
            
            # Look for the next send event for this conversation
            for future_i, future_event in enumerate(events[i+1:], i+1):
                if (future_event.conversation_id == event.conversation_id and 
                    future_event.event_type == "send"):
                    time_to_next_reuse = future_event.timestamp - current_time
                    break
            
            # Generate labels based on mode
            if self.config.mode == "ranking":
                if time_to_next_reuse == float('inf'):
                    label = 0  # No future reuse = lowest relevance
                else:
                    # higher values for shorter times
                    if time_to_next_reuse <= 60:
                        label = 3
                    elif time_to_next_reuse <= 180:
                        label = 2
                    else:
                        label = 1
            elif self.config.mode == "regression":
                # Regression: Predict relevance score based on time-to-next-reuse
                # Higher score = more relevant to keep in cache
                if time_to_next_reuse == float('inf'):
                    label = 0.0  # No future reuse = lowest relevance
                else:
                    # Short times → high scores, long times → low scores
                    # Use log(time + 1) to avoid negative values when time is very small
                    label = 10 - math.log(time_to_next_reuse + 1.0)
                    if label < 0:
                        raise ValueError(f"Label is negative: {label}")
            elif self.config.mode == "classification":
                # Binary classification: Should we keep this in cache?
                # 1 = keep (will be reused soon), 0 = evict (won't be reused soon)
                if time_to_next_reuse == float('inf'):
                    label = 0  # No future reuse = evict
                else:
                    # Threshold: keep if reused within 10 minutes
                    label = 1 if time_to_next_reuse <= 600 else 0
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
            
            labels_list.append(label)
        
        if not features_list:
            raise ValueError("No training data could be extracted from events")
        
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        # Group samples sequentially with fixed group size for ranking mode only
        group_sizes = None
        if self.config.mode == "ranking":
            group_size = 64  # Optimal group size for LightGBM ranking
            num_samples = len(features)
            num_groups = (num_samples + group_size - 1) // group_size  # Ceiling division
            
            group_sizes_list = []
            for group_idx in range(num_groups):
                start_idx = group_idx * group_size
                end_idx = min((group_idx + 1) * group_size, num_samples)
                current_group_size = end_idx - start_idx
                group_sizes_list.append(current_group_size)
            
            group_sizes = np.array(group_sizes_list)
            logger.info(f"Created {num_groups} sequential ranking groups (group_size={group_size})")
        
        # Log statistics based on mode
        logger.info(f"Prepared {len(features)} training samples for {self.config.mode}")
        if self.config.mode == "ranking":
            logger.info(f"Label stats: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.2f}")
        elif self.config.mode == "regression":
            logger.info(f"Relevance score stats: min={labels.min():.6f}, max={labels.max():.6f}, mean={labels.mean():.6f}")
            logger.info(f"High relevance (>0.1): {np.sum(labels > 0.1)} samples ({np.mean(labels > 0.1)*100:.1f}%)")
        elif self.config.mode == "classification":
            logger.info(f"Label distribution: keep={np.sum(labels == 1)}, evict={np.sum(labels == 0)}")
            logger.info(f"Keep ratio: {np.mean(labels):.3f}")
        
        # Debug: Feature statistics
        logger.info("=== FEATURE STATISTICS ===")
        for i, feature_name in enumerate(self.feature_names):
            feature_col = features[:, i]
            logger.info(f"Feature '{feature_name}': min={feature_col.min():.3f}, max={feature_col.max():.3f}, "
                       f"mean={feature_col.mean():.3f}, std={feature_col.std():.3f}, "
                       f"zeros={np.sum(feature_col == 0)}/{len(feature_col)}")
        
        # Debug: Sample a few feature vectors
        logger.info("=== SAMPLE FEATURE VECTORS ===")
        for i in range(min(5, len(features))):
            logger.info(f"Sample {i+1}: {features[i]}")
        
        # Debug: Check for constant features
        constant_features = []
        for i, feature_name in enumerate(self.feature_names):
            if features[:, i].std() == 0:
                constant_features.append(feature_name)
        if constant_features:
            logger.warning(f"Constant features (zero variance): {constant_features}")
        
        return features, labels, group_sizes
    
    def train(self, features: np.ndarray, labels: np.ndarray, group_sizes: Optional[np.ndarray] = None) -> None:
        """Train the LightGBM model based on the configured mode"""
        logger.info(f"Training LightGBM {self.config.mode} model...")
        
        # Create LightGBM dataset based on mode
        if self.config.mode == "ranking":
            if group_sizes is None:
                raise ValueError("group_sizes is required for ranking mode")
            train_data = lgb.Dataset(features, label=labels, group=group_sizes, feature_name=self.feature_names)
        else:
            train_data = lgb.Dataset(features, label=labels, feature_name=self.feature_names)
        
        # LightGBM parameters based on mode
        if self.config.mode == "ranking":
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'boosting_type': 'gbdt',
                'num_leaves': self.config.num_leaves,
                'learning_rate': self.config.learning_rate,
                'verbose': self.config.verbose,
                'lambdarank_truncation_level': 64,  # Top-k for ranking evaluation
                'lambdarank_norm': True
            }
        elif self.config.mode == "regression":
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': self.config.num_leaves,
                'learning_rate': self.config.learning_rate,
                'verbose': self.config.verbose,
            }
        elif self.config.mode == "classification":
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': self.config.num_leaves,
                'learning_rate': self.config.learning_rate,
                'verbose': self.config.verbose,
                'is_unbalance': True,  # Handle class imbalance
            }
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            callbacks=[lgb.log_evaluation(50)]
        )
        
        self.is_trained = True
        logger.info("LightGBM model training completed. Feature importance:")
        
        # Print feature importance
        logger.info("=== FEATURE IMPORTANCE (GAIN) ===")
        importance = self.model.feature_importance(importance_type='gain')
        total_importance = importance.sum()
        for i, imp in enumerate(importance):
            percentage = (imp / total_importance * 100) if total_importance > 0 else 0
            logger.info(f"Feature '{self.feature_names[i]}': {imp:.2f} ({percentage:.1f}%)")
        
        # Debug: Test prediction on a sample
        logger.info("=== SAMPLE PREDICTIONS ===")
        sample_features = features[:5] if len(features) > 0 else []
        if len(sample_features) > 0:
            predictions = self.model.predict(sample_features)
            for i, pred in enumerate(predictions):
                logger.info(f"Sample {i+1} prediction: {pred:.4f}")
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model to disk"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save both the model and config
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data.get('config', self.config)
        self.feature_names = model_data.get('feature_names', self.feature_names)
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Model loaded from {model_path}")
    
    def calculate_score(self, feature: Optional[ConversationFeature]) -> float:
        """Calculate eviction score using the trained LightGBM ranking model"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, returning default score")
            return 0.0
        
        if feature is None:
            return 0.0
        
        # Extract features and predict
        feature_vector = self._extract_features(feature)
        prediction = self.model.predict([feature_vector])[0]
        
        # The ranking model predicts time-to-next-reuse
        # For eviction: higher predicted time = higher eviction score (more likely to evict)
        # We can use the prediction directly as eviction score
        # Add small constant to avoid zero scores
        eviction_score = float(prediction) + 1e-6
        
        # Don't apply decay here - it will be applied in the eviction policy
        return eviction_score
    
    def cache_admission(self, feature: Optional[ConversationFeature]) -> bool:
        """Determine if a conversation should be admitted to cache"""
        return feature is not None

def get_model_path_from_input_file(input_file: str, model_dir: str = "models/") -> str:
    """Generate model path based on input file name"""
    # Extract base name and create model filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    model_name = f"lightgbm_{base_name}.pkl"
    return os.path.join(model_dir, model_name)

def train_model_from_events(events: List[Event], mode: str = "ranking") -> LightGBMScorer:
    """
    Train a LightGBM model from already-parsed events
    
    Args:
        events: List of log events to train on
        mode: Training mode - "ranking", "regression", or "classification"
        
    Returns:
        Trained LightGBMScorer
    """
    logger.info(f"Training LightGBM {mode} model from {len(events)} events")
    
    if not events:
        raise ValueError("No events provided for training")
    
    # Create scorer with specified mode
    config = LightGBMScorerConfig(mode=mode)
    scorer = LightGBMScorer(config)
    features, labels, group_sizes = scorer.prepare_training_data(events)
    
    # Train model
    scorer.train(features, labels, group_sizes)
    
    return scorer
