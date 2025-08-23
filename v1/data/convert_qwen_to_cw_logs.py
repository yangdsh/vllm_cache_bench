#!/usr/bin/env python3
"""
Convert JSONL format to cw_logs CSV format.

Input JSONL fields:
- chat_id: unique identifier
- timestamp: float (seconds since epoch)
- input_length: integer
- output_length: integer
- Other fields (ignored): parent_chat_id, type, turn, hash_ids

Output CSV fields:
- @timestamp: timestamp string in format "YYYY-MM-DD HH:MM:SS.mmm"
- ConversationId: conversation identifier as 1-indexed integer
- CustomerQueryLength: input length as float
- GenerativeModelResponseLength: output length as float
"""

import json
import csv
import sys
from datetime import datetime


class UnionFind:
    """Union-Find data structure to group chat messages into conversations."""
    
    def __init__(self):
        self.parent = {}
    
    def make_set(self, x):
        """Initialize a new set containing only x."""
        if x not in self.parent:
            self.parent[x] = x
    
    def find(self, x):
        """Find the root of the set containing x with path compression."""
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """Union the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            # Union by rank
            if root_x < root_y:
                self.parent[root_x] = root_y
            elif root_x > root_y:
                self.parent[root_y] = root_x


def convert_timestamp(timestamp_float):
    """Convert float timestamp to datetime string format."""
    # Assuming the timestamp is seconds since epoch
    dt = datetime.fromtimestamp(timestamp_float)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Remove last 3 digits of microseconds


def convert_jsonl_to_csv(input_file, output_file):
    """Convert JSONL file to CSV format matching cw_logs."""
    
    print("First pass: Building conversation groups...")
    
    # First pass: Build conversation groups using Union-Find
    uf = UnionFind()
    messages = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                messages.append(data)
                
                chat_id = data['chat_id']
                parent_chat_id = data['parent_chat_id']
                
                # Initialize the chat_id in union-find
                uf.make_set(chat_id)
                
                # If this message has a parent, union them
                if parent_chat_id != -1:
                    uf.make_set(parent_chat_id)
                    uf.union(chat_id, parent_chat_id)
                
                # Progress indicator
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num} lines...")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print("Second pass: Assigning conversation IDs and writing CSV...")
    
    # Create mapping from conversation root to 1-indexed conversation ID
    conversation_roots = set()
    for msg in messages:
        root = uf.find(msg['chat_id'])
        conversation_roots.add(root)
    
    # Sort roots to ensure deterministic assignment
    sorted_roots = sorted(conversation_roots)
    root_to_conv_id = {root: idx + 1 for idx, root in enumerate(sorted_roots)}
    
    # Dictionary to track accumulated lengths for each conversation
    accumulated_lengths = {}
    
    # CSV headers matching the cw_logs format
    headers = ["@timestamp", "ConversationId", "CustomerQueryLength", 
               "GenerativeModelResponseLength"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        
        for line_num, data in enumerate(messages, 1):
            try:
                # Find the conversation root for this message
                conversation_root = uf.find(data['chat_id'])
                conversation_id = root_to_conv_id[conversation_root]
                
                # Extract required fields
                timestamp = convert_timestamp(data['timestamp'])
                input_length = float(data['input_length'])
                output_length = float(data['output_length'])
                
                # Initialize accumulated length for this conversation if not exists
                if conversation_id not in accumulated_lengths:
                    accumulated_lengths[conversation_id] = 0.0
                
                # Calculate customer query length as current input_length minus accumulated length
                customer_query_length = input_length - accumulated_lengths[conversation_id]
                
                # Update accumulated length for this conversation
                accumulated_lengths[conversation_id] += customer_query_length + output_length
                
                # Write the row
                writer.writerow([
                    timestamp,
                    str(conversation_id),
                    customer_query_length,
                    output_length
                ])
                
                # Progress indicator
                if line_num % 10000 == 0:
                    print(f"  Written {line_num} lines...")
                    
            except (KeyError) as e:
                print(f"Warning: Error processing message {line_num}: {e}")
                continue
    
    print(f"Conversion completed. Output written to {output_file}")
    print(f"Total conversations identified: {len(sorted_roots)}")
    print(f"Total messages processed: {len(messages)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_qwen_to_cw_logs.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Converting {input_file} to {output_file}...")
    convert_jsonl_to_csv(input_file, output_file)

