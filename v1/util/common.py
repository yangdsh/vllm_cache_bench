import re
import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from transformers import AutoConfig, AutoTokenizer
from datetime import datetime

@dataclass
class Event:
    """Represents a log event for replay (send/done, from either client or server log)"""
    event_type: str  # "send" or "done"
    request_id: str
    conversation_id: int
    turn_number: int
    input_tokens: int  # For send events
    hit_tokens: int    # For send events
    generated_tokens: int  # For done events
    timestamp: float = 0.0

class LogParser:
    """
    Unified log parser for both server and client logs.
    Always returns Event objects with appropriate default values.
    """
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.events: List[Event] = []
        self.aborted_count = 0

    def parse_log_file(self, mode: str = 'server') -> List[Event]:
        """
        Parse the log file and return Event objects.
        mode: 'server' for server logs, 'client' for client logs
        """
        if mode == 'server':
            return self._parse_server_log()
        elif mode == 'client':
            return self._parse_client_log()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'server' or 'client'.")

    def _parse_server_log(self) -> List[Event]:
        """Parse server log file and return Event objects ordered to match client event order"""
        observability_pattern = re.compile(
            r'\[Observability\] on_lookup: conv_(\d+)_(\d+), '
            r'num_tokens=(\d+), hit_tokens=(\d+), accumulated_hit_tokens=(\d+), '
            r'lookup_id=([a-zA-Z0-9-]+)'
        )
        
        # First, get client events to understand the correct order
        client_events = self._parse_client_log_events()
        
        # Create a mapping of server log data by (conversation_id, turn_number)
        server_data = {}
        with open(self.log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                timestamp = parse_timestamp_from_log_line(line)
                
                obs_match = observability_pattern.search(line)
                
                if obs_match:
                    conversation_id = int(obs_match.group(1))
                    turn_number = int(obs_match.group(2))
                    num_tokens_tokenized = int(obs_match.group(3))
                    hit_tokens = int(obs_match.group(4))
                    request_id = obs_match.group(6)  # This is lookup_id
                    
                    key = (conversation_id, turn_number)
                    server_data[key] = {
                        'request_id': request_id,
                        'input_tokens': num_tokens_tokenized,
                        'hit_tokens': hit_tokens,
                        'timestamp': timestamp
                    }
        
        # Now create events in the same order as client events
        events = []
        for client_event in client_events:
            key = (client_event.conversation_id, client_event.turn_number)
            
            if key in server_data:
                server_info = server_data[key]
                
                if client_event.event_type == "send":
                    # Create send event with server data
                    send_event = Event(
                        event_type="send",
                        request_id=server_info['request_id'],
                        conversation_id=client_event.conversation_id,
                        turn_number=client_event.turn_number,
                        input_tokens=server_info['input_tokens'],
                        hit_tokens=server_info['hit_tokens'],
                        generated_tokens=0,
                        timestamp=server_info['timestamp']
                    )
                    events.append(send_event)
                
                elif client_event.event_type == "done":
                    # Create done event with client's generated tokens
                    done_event = Event(
                        event_type="done",
                        request_id=server_info['request_id'],
                        conversation_id=client_event.conversation_id,
                        turn_number=client_event.turn_number,
                        input_tokens=0,
                        hit_tokens=0,
                        generated_tokens=client_event.generated_tokens,
                        timestamp=client_event.timestamp
                    )
                    events.append(done_event)
            else:
                self.aborted_count += 1
                continue
        
        print(f"Aborted count: {self.aborted_count}", flush=True)
        self.events = events
        # Sort events by timestamp
        self.events.sort(key=lambda x: x.timestamp)
        return events

    def _parse_client_log(self) -> List[Event]:
        """Parse client log file and return Event objects"""
        events = self._parse_client_log_events()
        self.events = events
        # Sort events by timestamp
        self.events.sort(key=lambda x: x.timestamp)
        return events

    def _parse_client_log_events(self) -> List[Event]:
        """Parse client log file and return all Event objects"""
        send_pattern = re.compile(
            r'Send conv_id: (\d+), turn_id: (\d+), waiting_time: ([\d.]+), '
            r'input_len: (\d+), output_len: (\d+)'
        )
        done_pattern = re.compile(
            r'Done conv_id: (\d+), output_len: (\d+), turn_id: (\d+)'
        )
        
        events = []
        client_log_path = self.log_file_path.replace('server', 'client')
        
        try:
            with open(client_log_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    timestamp = parse_timestamp_from_log_line(line)
                    
                    # Try to match Send event
                    send_match = send_pattern.search(line)
                    if send_match:
                        event = Event(
                            event_type="send",
                            request_id=str(len(events)),
                            conversation_id=int(send_match.group(1)),
                            turn_number=int(send_match.group(2)),
                            input_tokens=int(send_match.group(4)),
                            hit_tokens=0,  # Client logs don't have hit tokens
                            generated_tokens=0,
                            timestamp=timestamp
                        )
                        events.append(event)
                        continue
                    
                    # Try to match Done event
                    done_match = done_pattern.search(line)
                    if done_match:
                        event = Event(
                            event_type="done",
                            request_id=str(len(events)),
                            conversation_id=int(done_match.group(1)),
                            turn_number=int(done_match.group(3)),
                            input_tokens=0,
                            hit_tokens=0,
                            generated_tokens=int(done_match.group(2)),
                            timestamp=timestamp
                        )
                        events.append(event)
                        continue
        except FileNotFoundError:
            print(f"Warning: Client log file not found: {client_log_path}")
            return []
        
        return events

def parse_timestamp_from_log_line(line: str) -> float:
    """
    Parse timestamp from log line format: [YYYY-MM-DD HH:MM:SS,mmm] or YYYY-MM-DD HH:MM:SS,mmm
    Returns timestamp as float (seconds since epoch)
    """
    # Try pattern with brackets first (server log format)
    timestamp_pattern_with_brackets = re.compile(r'\[([\d-]+) ([\d:,]+)\]')
    match = timestamp_pattern_with_brackets.search(line)
    
    if not match:
        # Try pattern without brackets (client log format)
        timestamp_pattern_without_brackets = re.compile(r'^([\d-]+) ([\d:,]+)')
        match = timestamp_pattern_without_brackets.search(line)
    
    if not match:
        return 0.0
    
    date_str = match.group(1)  # YYYY-MM-DD
    time_str = match.group(2)  # HH:MM:SS,mmm
    
    # Parse date and time
    try:
        # Convert comma to period for microseconds
        time_str_clean = time_str.replace(',', '.')
        datetime_str = f"{date_str} {time_str_clean}"
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp()
    except ValueError:
        print(f"Warning: Failed to parse timestamp from line: {line}")
        return 0.0

def get_bytes_per_token(model_name: str) -> tuple:
    """
    Load model config and compute bytes per token for KV cache, matching benchmark_replay.py logic.
    Returns (config, bytes_per_token)
    """
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    head_dim = config.hidden_size // num_attention_heads
    bytes_per_element = 2  # bfloat16
    cache_size_per_layer_per_token = 2 * num_key_value_heads * head_dim * bytes_per_element
    total_cache_size_per_token = cache_size_per_layer_per_token * num_layers
    return config, total_cache_size_per_token

def get_chat_template_overhead(model_name: str) -> int:
    """
    Calculate the token overhead introduced by the chat template.
    Returns the number of additional tokens added by apply_chat_template().
    """
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create a simple conversation to measure overhead
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        
        # Apply chat template
        chat_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # Tokenize the raw messages (without chat template)
        raw_text = ""
        for msg in messages:
            raw_text += f"{msg['content']}"
        raw_tokens = tokenizer.encode(raw_text)
        
        # Calculate overhead
        overhead = len(chat_tokens) - len(raw_tokens)
        
        print(f"Chat template overhead for {model_name}: {overhead} tokens")
        return overhead
        
    except Exception as e:
        print(f"Warning: Could not calculate chat template overhead for {model_name}: {e}")
        print("Using default overhead of 8 tokens")
        return 8

def kill_server(host):
    """Kill all processes using the GPUs on a host."""
    try:
        # Kill any vllm processes for the current user
        user = os.environ.get('USER')
        if user:
            pkill_cmd = f"pkill -u {user} -f vllm"
            subprocess.run(pkill_cmd, shell=True, check=True)
            print(f"Killed vllm processes for user {user}")
    except subprocess.CalledProcessError as e:
        print(f"Skipping killing vllm processes for user {user}: {e}")
    
    try:
        # Command to find and kill all processes running on NVIDIA GPUs.
        # It first checks if nvidia-smi command exists.
        # Then, it gets the PIDs of GPU processes.
        # If any PIDs are found, it attempts to kill them with SIGKILL.
        kill_cmd = (
            "if command -v nvidia-smi &> /dev/null; then "
            "pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); "
            "if [ -n \"$pids\" ]; then "
            "echo \"$pids\" | xargs -r kill -9; "
            "fi; "
            "fi"
        )
        subprocess.run(kill_cmd, shell=True, check=True)
        print("Killed any processes using GPUs on the local machine.")
        print("Waiting for 10 seconds to ensure all processes are killed...")
        time.sleep(10)
    except subprocess.CalledProcessError as e:
        print(f"No processes to kill or an error occurred: {e}")
        return