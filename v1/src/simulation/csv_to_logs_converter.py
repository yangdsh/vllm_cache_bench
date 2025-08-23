#!/usr/bin/env python3
"""
Convert test_data.csv to server.log and client.log files.

This script reads a CSV file with conversation data and generates corresponding
server and client log files in the format expected by the log parsing system.
"""

import argparse
import csv
import uuid
from datetime import datetime, timedelta
from pathlib import Path


def parse_csv_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp from CSV format: 2025-05-29 05:00:07.053"""
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")


def format_log_timestamp(dt: datetime) -> str:
    """Format datetime as log timestamp: [2025-08-19 01:03:15,881]"""
    return f"[{dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}]"


def format_client_timestamp(dt: datetime) -> str:
    """Format datetime as client log timestamp: 2025-08-19 01:03:15,881"""
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}"


def generate_lookup_id() -> str:
    """Generate a lookup ID similar to the format in server logs"""
    return f"chatcmpl-{uuid.uuid4().hex}"


def convert_csv_to_logs(csv_file: str, model_name: str, tokens_per_second: float = 50.0, 
                        output_dir: str = "outputs/logs"):
    """
    Convert CSV data to server.log and client.log files.
    
    Args:
        csv_file: Path to the input CSV file
        model_name: Model name to include in output filenames
        tokens_per_second: Rate of token generation for calculating done event timestamps
        output_dir: Directory to write the log files
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    server_log_path = output_path / f"server_{model_name}.log"
    client_log_path = output_path / f"client_{model_name}.log"
    
    # Track conversation IDs and turn numbers
    conversation_map = {}
    conversation_counter = 1
    
    # Track empty values statistics
    empty_query_count = 0
    empty_response_count = 0
    long_query_count = 0
    server_entries = []
    client_entries = []
    
    with open(csv_file, 'r') as f:
        # Skip header line that starts with @
        first_line = f.readline()
        if not first_line.startswith('@'):
            f.seek(0)  # Reset if no header
        
        reader = csv.DictReader(f, fieldnames=['timestamp', 'ConversationId', 
            'CustomerQueryLength', 'GenerativeModelResponseLength', 'ModelName'])
        
        for row in reader:
            # Skip header row if it exists
            if row['timestamp'] == '@timestamp':
                continue
                
            timestamp_str = row['timestamp']
            conversation_id_str = row['ConversationId']
            
            # Handle empty CustomerQueryLength
            if row['CustomerQueryLength'] != '':
                query_tokens = int(float(row['CustomerQueryLength']))
            else:
                query_tokens = 50  # Default input tokens if empty
                empty_query_count += 1
            if query_tokens > 16384:
                long_query_count += 1
                continue
                
            # Handle empty GenerativeModelResponseLength
            if row['GenerativeModelResponseLength'] != '':
                output_tokens = int(float(row['GenerativeModelResponseLength']))
            else:
                output_tokens = 128
                empty_response_count += 1
            
            # Map conversation ID to numeric ID and track turns
            if conversation_id_str not in conversation_map:
                conversation_map[conversation_id_str] = {
                    'numeric_id': conversation_counter,
                    'turn_count': 0,
                    'cumulative_tokens': 0  # Track cumulative tokens for this conversation
                }
                conversation_counter += 1
            
            conversation_map[conversation_id_str]['turn_count'] += 1
            conv_numeric_id = conversation_map[conversation_id_str]['numeric_id']
            turn_number = conversation_map[conversation_id_str]['turn_count']
            
            # Calculate input_tokens as cumulative (previous queries + responses + current query)
            current_cumulative = conversation_map[conversation_id_str]['cumulative_tokens']
            input_tokens = current_cumulative + query_tokens
            
            # Update cumulative tokens for this conversation (add current query + response)
            conversation_map[conversation_id_str]['cumulative_tokens'] += query_tokens + output_tokens
            
            # Parse send timestamp
            send_timestamp = parse_csv_timestamp(timestamp_str)
            
            # Calculate done timestamp based on output tokens and generation rate
            generation_time = output_tokens / tokens_per_second
            done_timestamp = send_timestamp + timedelta(seconds=generation_time)
            
            # Generate lookup ID
            lookup_id = generate_lookup_id()
            
            # Create server log entry (observability pattern)
            # Assuming hit_tokens = 0 for simplicity (no cache hits in test data)
            hit_tokens = 0
            accumulated_hit_tokens = 0
            
            server_entry = (
                f"{format_log_timestamp(send_timestamp)} LMCache INFO: "
                f"[Observability] on_lookup: conv_{conv_numeric_id}_{turn_number}, "
                f"num_tokens={input_tokens}, hit_tokens={hit_tokens}, "
                f"accumulated_hit_tokens={accumulated_hit_tokens}, lookup_id={lookup_id}"
            )
            server_entries.append((send_timestamp, server_entry))
            
            # Create client log entries
            # Send event
            waiting_time = 0.0  # Assume no waiting time for test data
            send_entry = (
                f"{format_client_timestamp(send_timestamp)} "
                f"Send conv_id: {conv_numeric_id}, turn_id: {turn_number}, "
                f"waiting_time: {waiting_time:.3f}, input_len: {input_tokens}, "
                f"output_len: {output_tokens}"
            )
            client_entries.append((send_timestamp, send_entry))
            
            # Done event
            done_entry = (
                f"{format_client_timestamp(done_timestamp)} "
                f"Done conv_id: {conv_numeric_id}, output_len: {output_tokens}, "
                f"turn_id: {turn_number}"
            )
            client_entries.append((done_timestamp, done_entry))
    
    # Sort entries by timestamp
    server_entries.sort(key=lambda x: x[0])
    client_entries.sort(key=lambda x: x[0])
    
    # Write server log
    with open(server_log_path, 'w') as f:
        for _, entry in server_entries:
            f.write(entry + '\n')
    
    # Write client log
    with open(client_log_path, 'w') as f:
        for _, entry in client_entries:
            f.write(entry + '\n')
    
    print(f"Generated server log: {server_log_path}")
    print(f"Generated client log: {client_log_path}")
    print(f"Model name: {model_name}")
    print(f"Processed {len(server_entries)} conversations")
    print(f"Token generation rate: {tokens_per_second} tokens/second")
    print(f"Empty queries (CustomerQueryLength): {empty_query_count}")
    print(f"Empty responses (GenerativeModelResponseLength): {empty_response_count}")
    print(f"Long queries (CustomerQueryLength > 16384): {long_query_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert test_data.csv to server.log and client.log files"
    )
    parser.add_argument(
        "csv_file",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "model_name",
        help="Model name to include in output filenames (e.g., 'test_data', 'qwen3-8b')"
    )
    parser.add_argument(
        "--tokens-per-second",
        type=float,
        default=50.0,
        help="Token generation rate for calculating done event timestamps (default: 50.0)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/logs",
        help="Directory to write the log files (default: outputs/logs)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_csv_to_logs(
            csv_file=args.csv_file,
            model_name=args.model_name,
            tokens_per_second=args.tokens_per_second,
            output_dir=args.output_dir
        )
    except FileNotFoundError:
        print(f"Error: CSV file '{args.csv_file}' not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
