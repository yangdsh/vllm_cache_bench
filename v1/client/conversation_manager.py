#!/usr/bin/env python3
"""
This module provides utility functions for conversation management and prompt generation
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any

from lmcache.logging import init_logger

logger = init_logger(__name__)

class ConversationManager:
    """Manages conversation history and message handling"""
    
    def __init__(self, tokenizer):
        self.conversation_history: Dict[int, List[Dict]] = defaultdict(list)
        self.conversation_last_time: Dict[int, float] = {}
        self.sonnet_text: Optional[str] = None
        self.tokenizer = tokenizer
        self._load_sonnet_text()
    
    def _load_sonnet_text(self) -> None:
        """Load sonnet text for generating realistic prompts."""
        # Use relative path from the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sonnet_path = os.path.join(current_dir, 'sonnet.txt')
        
        try:
            with open(sonnet_path, 'r', encoding='utf-8') as f:
                self.sonnet_text = f.read()
            logger.info(f"Loaded sonnet text: {len(self.sonnet_text)} characters")
        except FileNotFoundError:
            logger.warning(f"Warning: Could not find sonnet.txt at {sonnet_path}, falling back to simple text")
            # Fallback text if sonnet.txt is not found
            self.sonnet_text = "The quick brown fox jumps over the lazy dog. " * 1000
    
    def generate_prompt(self, prompt_tokens: int, conversation_id: int) -> str:
        """
        Generate a prompt with the specified number of tokens using sonnet text.
        
        Adapted from benchmark_dataset.py _generate_prompt method.
        """
        if not self.sonnet_text:
            raise ValueError("Sonnet text not loaded")
        
        prompt_tokens = max(prompt_tokens, 8)
        
        # Calculate starting position in sonnet text, wrapping around if needed
        sonnet_start = conversation_id % len(self.sonnet_text)
        
        # Extract text segment and create tokens
        # append conv id for debugging
        text_segment = str(conversation_id)
        # prompt engineering to make the response reaching the expected length
        text_segment += "\n write a long story about: "
        text_segment += self.sonnet_text[sonnet_start:sonnet_start + prompt_tokens * 10]
        if len(text_segment) < prompt_tokens * 10:  # If near end of file, wrap around
            text_segment += self.sonnet_text[:prompt_tokens * 10]
        
        # Tokenize the text segment and get the desired number of tokens
        if self.tokenizer:
            token_ids = self.tokenizer.encode(text_segment, add_special_tokens=False)[:prompt_tokens]
            
            # Pad with repeated text if we don't have enough tokens
            while len(token_ids) < prompt_tokens:
                additional_text = text_segment
                additional_tokens = self.tokenizer.encode(additional_text, add_special_tokens=False)
                token_ids.extend(additional_tokens[:prompt_tokens - len(token_ids)])
            
            # Ensure we have exactly the right number of tokens
            token_ids = token_ids[:prompt_tokens]
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            return ''
    
    def add_user_message(self, conversation_id: int, prompt: str, 
                    multi_modal_content: Optional[Dict] = None) -> List[Dict]:
        """
        Get conversation messages for a given conversation.
        """
        user_message = {
            "role": "user",
            "content": prompt,
        }
        
        if conversation_id <= 0:
            return

        # Handle multi-modal content if provided
        if multi_modal_content:
            content = [{"type": "text", "text": prompt}]
            content.append(multi_modal_content)
            user_message["content"] = content
        
        self.conversation_history[conversation_id].append(user_message)

    def get_all_messages(self, conversation_id: int) -> List[Dict]:
        """
        Get all messages for a given conversation.
        """
        #print(f"get_all_messages for: {conversation_id}")
        #for message in self.conversation_history[conversation_id]:
        #    print(f"{len(self.tokenizer.encode(message['content']))}")
        return self.conversation_history.get(conversation_id, [])
    
    def add_gpt_message(self, conversation_id: int, generated_text: str):
        """
        Update conversation with assistant response.
        
        Adapted from backend_request_func.py update_conversation function.
        """
        # a hack only for benchmarking qwen3
        # todo: should remove all tokens in the think block
        if '</think>' in generated_text:
            print("Warning: we removed </think> in generated text")
            generated_text = generated_text.replace("</think>", "<think>")
        
        assistant_message = {
            "role": "assistant",
            "content": generated_text,
        }
        
        self.conversation_history[conversation_id].append(assistant_message)
        self.conversation_last_time[conversation_id] = time.time()
    
    def print_conversation_history(self, conversation_id: int):
        """Print conversation history for debugging"""
        if conversation_id not in self.conversation_history:
            print(f"No conversation history for ID: {conversation_id}")
            return
        
        print(f"Conversation {conversation_id} history:")
        for message in self.conversation_history[conversation_id]:
            print(f"role: {message['role']}")
            if isinstance(message['content'], list):
                for content_item in message['content']:
                    if isinstance(content_item, dict) and content_item.get('type') == 'text':
                        print(f"{content_item['text']}")
                    else:
                        print(f"{content_item}")
            else:
                print(f"{message['content']}")
            print()
    
    def get_conversation_count(self) -> int:
        """Get total number of conversations"""
        return len(self.conversation_history)
    
    def get_total_messages(self) -> int:
        """Get total number of messages across all conversations"""
        return sum(len(messages) for messages in self.conversation_history.values())
    
    def get_conversation_messages_count(self, conversation_id: int) -> int:
        """Get message count for a specific conversation"""
        return len(self.conversation_history.get(conversation_id, []))

    def clear_conversation(self, conversation_id: int):
        """Clear a specific conversation"""
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
        if conversation_id in self.conversation_last_time:
            del self.conversation_last_time[conversation_id]
    
    def clear_all_conversations(self):
        """Clear all conversation history"""
        self.conversation_history.clear()
        self.conversation_last_time.clear()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get comprehensive conversation statistics"""
        total_conversations = self.get_conversation_count()
        total_messages = self.get_total_messages()
        
        # Calculate message distribution
        message_counts = [len(messages) for messages in self.conversation_history.values()]
        avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
        
        # Turn distribution (similar to what's shown in dataset statistics)
        turn_distribution = defaultdict(int)
        for messages in self.conversation_history.values():
            # Count user messages to get turn count
            user_messages = sum(1 for msg in messages if msg.get('role') == 'user')
            turn_distribution[user_messages] += 1
        
        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'avg_messages_per_conversation': avg_messages,
            'turn_distribution': dict(turn_distribution),
            'message_counts': message_counts
        }

# Convenience functions for backward compatibility
def create_conversation_manager() -> ConversationManager:
    """Create a new conversation manager instance"""
    return ConversationManager()