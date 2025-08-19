# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import sys
import math
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp
import asyncio
import huggingface_hub.constants
import requests
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from vllm.model_executor.model_loader.weight_utils import get_lock
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Custom formatter to match the exact format [2025-08-07 03:54:01,144]
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            # Format the date/time without microseconds
            formatted = dt.strftime('%Y-%m-%d %H:%M:%S')
            # Get milliseconds and append with comma
            milliseconds = int(dt.microsecond / 1000)
            return f"{formatted},{milliseconds:03d}"
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S,%f')

# Configure the logger to include timestamps
if not logger.handlers:  # Only configure if no handlers exist
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
    handler = logging.StreamHandler()
    formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                              datefmt='[%Y-%m-%d %H:%M:%S,%f]')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

# Add the parent directory to the path so we can import util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import conversation utilities
from utils.conversation_manager import ConversationManager
# Import cache statistics functionality
from utils.print_statistics import print_cache_statistics

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    tokenizer: Optional[AutoTokenizer] = None
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    timestamp: float = 0
    interval: float = 0
    time_limit: float = 10000
    conversation_id: int = -1
    turn_id: int = -1


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    message_len: int = 0  # Total message length including conversation history
    error: str = ""
    server: str = "localhost"
    done_time: float = 0.0


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        params = {
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            "truncate": request_func_input.prompt_len,
            # TGI does not accept ignore_eos flag.
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        # NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    output.generated_text = parsed_resp["text"][0]
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "min_tokens": request_func_input.output_len+1, 
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.message_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

conversation_manager = None
start_time = time.time()

# Statistics tracking class
class RequestStatistics:
    def __init__(self):
        self.total_requests = 0
        self.total_waitings = 0
        self.total_follow_up_turns = 0
        self.running_requests = 0
        self.prompt_lengths = []
        self.message_lengths = []  # Total length including conversation history
        self.output_lengths = []
        self.waiting_times = []
        self.error_counts = {}  # Track different error types
        self.last_print_count = 0
    
    def add_request(self, prompt_len: int, message_len: int, output_len: int, 
                   waiting_time: float = 0, is_follow_up: bool = False):
        self.total_requests += 1
        self.prompt_lengths.append(prompt_len)
        self.message_lengths.append(message_len)
        self.output_lengths.append(output_len)

        if waiting_time > 0:
            self.waiting_times.append(waiting_time)
            self.total_waitings += 1
        
        if is_follow_up:
            self.total_follow_up_turns += 1
    
    def add_error(self, error_type: str):
        """Track different types of errors/timeouts"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
    
    def start_request(self):
        """Track when a request starts processing"""
        self.running_requests += 1
    
    def complete_request(self):
        """Track when a request completes processing"""
        self.running_requests = max(0, self.running_requests - 1)
    
    def get_statistics(self) -> dict:
        if not self.prompt_lengths:
            return {}
        
        return {
            'total_requests': self.total_requests,
            'running_requests': self.running_requests,
            'total_waitings': self.total_waitings,
            'total_follow_up_turns': self.total_follow_up_turns,
            'avg_prompt_len': sum(self.prompt_lengths) / len(self.prompt_lengths),
            'avg_message_len': sum(self.message_lengths) / len(self.message_lengths),
            'avg_output_len': sum(self.output_lengths) / len(self.output_lengths),
            'avg_waiting_time': sum(self.waiting_times) / len(self.waiting_times) if self.waiting_times else 0,
            'max_waiting_time': max(self.waiting_times) if self.waiting_times else 0,
            'min_waiting_time': min(self.waiting_times) if self.waiting_times else 0,
            'waiting_percentage': (self.total_waitings / self.total_requests * 100) if self.total_requests > 0 else 0,
            'follow_up_percentage': (self.total_follow_up_turns / self.total_requests * 100) if self.total_requests > 0 else 0,
            'error_counts': self.error_counts.copy()
        }
    
    def should_print_statistics(self) -> bool:
        """Check if we should print statistics (every 100 requests)"""
        return self.total_requests > 0 and self.total_requests % 100 == 0 and self.total_requests != self.last_print_count
    
    def print_statistics(self, api_url: str):
        """Print comprehensive request statistics including cache metrics"""
        stats = self.get_statistics()
        if not stats:
            return
        
        print("\n" + "="*70)
        print(f"📊 COMPREHENSIVE REQUEST STATISTICS")
        print("="*70)
        print(f"Total requests processed: {stats['total_requests']}")
        print(f"Currently running requests: {stats['running_requests']}")
        print(f"Follow-up turns: {stats['total_follow_up_turns']} ({stats['follow_up_percentage']:.1f}%)")
        print(f"Requests with waiting: {stats['total_waitings']} ({stats['waiting_percentage']:.1f}%)")
        print(f"Average waiting time: {stats['avg_waiting_time']:.4f}s")
        print(f"Max waiting time: {stats['max_waiting_time']:.4f}s")
        
        print("-"*50)
        print(f"Average prompt length: {stats['avg_prompt_len']:.1f} tokens")
        print(f"Average message length: {stats['avg_message_len']:.1f} tokens")
        print(f"Average output length: {stats['avg_output_len']:.1f} tokens")
        
        # Fetch and display cache statistics
        vllm_hit_ratio = print_cache_statistics(api_url)
        
        print("="*70 + "\n")
        self.last_print_count = self.total_requests
        return vllm_hit_ratio

# Global statistics tracker
request_stats = RequestStatistics()

async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    global request_stats
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = ConversationManager(request_func_input.tokenizer)
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    output = RequestFuncOutput()
    output.error = "did not start"
    receive_time = time.time()
    if receive_time - start_time > request_func_input.time_limit:
        return output

    async def wait_for_previous_turns(request_func_input: RequestFuncInput) -> tuple[bool, float, str]:
        """
        Wait for previous turns in the conversation to complete and handle scheduling.
        
        Returns:
            tuple[bool, float, str]: (should_continue, waiting_time, error_reason)
        """
        waiting_time = 0.0
        if time.time() - start_time +request_func_input.output_len / 30 > request_func_input.time_limit:
            logger.info(f"Abort conversation_id: {request_func_input.conversation_id}, "
              f"turn_id: {request_func_input.turn_id}, "
              f"waiting_time: {waiting_time}, "
              f"input_len: {request_func_input.prompt_len}, "
              f"output_len: {request_func_input.output_len}")
            return False, waiting_time, "timeout_generation_would_exceed_limit"
        if request_func_input.turn_id <= 1:
            logger.info(f"Send conv_id: {request_func_input.conversation_id}, "
              f"turn_id: {request_func_input.turn_id}, "
              f"waiting_time: {waiting_time}, "
              f"input_len: {request_func_input.prompt_len}, "
              f"output_len: {request_func_input.output_len}")
            return True, waiting_time, ""
        
        waiting_start = time.time()
        
        # Wait until all previous turns are finished - use conversation manager
        cur_turn_id = conversation_manager.get_conversation_messages_count(request_func_input.conversation_id) // 2
        while cur_turn_id != request_func_input.turn_id - 1:
            await asyncio.sleep(3)
            cur_turn_id = conversation_manager.get_conversation_messages_count(request_func_input.conversation_id) // 2
            logger.debug(f"waiting conv_id: {request_func_input.conversation_id}, "
              f"turn_id: {request_func_input.turn_id}, "
              f"cur_turn_id: {cur_turn_id}, "
              f"waiting_time: {waiting_time}, "
              f"input_len: {request_func_input.prompt_len}, "
              f"output_len: {request_func_input.output_len}")
            waiting_time = time.time() - waiting_start
            if time.time() - start_time > request_func_input.time_limit:
                return False, waiting_time, "timeout_last_turn_did_not_finish"

        # Calculate scheduled time based on previous conversation timing
        time_to_schedule = request_func_input.interval
        scheduled_time = conversation_manager.conversation_last_time.get(request_func_input.conversation_id, time.time()) + time_to_schedule
        
        # Check if request would exceed time limit
        if scheduled_time - start_time + request_func_input.output_len / 30 > request_func_input.time_limit:
            return False, waiting_time, "timeout_generation_would_exceed_limit"

        if time_to_schedule > 0:
            await asyncio.sleep(time_to_schedule)
        
        waiting_time = time.time() - waiting_start
        if request_func_input.interval == 0:
            waiting_time = 0
        logger.info(f"Send conv_id: {request_func_input.conversation_id}, "
              f"turn_id: {request_func_input.turn_id}, "
              f"waiting_time: {waiting_time}, "
              f"input_len: {request_func_input.prompt_len}, "
              f"output_len: {request_func_input.output_len}")
        return True, waiting_time, ""
    
    # Use the helper method for waiting logic
    should_continue, waiting_time, error_reason = await wait_for_previous_turns(request_func_input)
    if not should_continue:
        output.error = error_reason
        request_stats.add_error(error_reason)
        return output
    
    # Update timestamp if waiting occurred
    if waiting_time > 0:
        request_func_input.timestamp += time.time() - receive_time

    # Track follow-up requests
    is_follow_up = request_func_input.turn_id > 1
    
    # Calculate message length (including conversation history) - use conversation manager
    conversation_manager.add_user_message(request_func_input.conversation_id, 
                                          request_func_input.prompt, 
                                          request_func_input.prompt_len,
                                          request_func_input.multi_modal_content)
    messages = conversation_manager.get_all_messages(request_func_input.conversation_id)
    if request_func_input.tokenizer:
        try:
            prompt = request_func_input.tokenizer.apply_chat_template(messages, tokenize=True)
            message_length = len(prompt)
        except Exception as e:
            message_length = 0
    else:
        message_length = 0
    
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "messages": messages,
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "min_tokens": request_func_input.output_len, 
            "stream": True,
            "stream_options": {
                "include_usage": True,
            }
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.message_len = message_length
        output.success = False

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        request_stats.start_request()
        #print(round(time.time()-start_time,2), request_func_input.timestamp, 
        #      request_func_input.conversation_id, request_stats.total_requests)
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")
                        # NOTE: SSE comments (often used as pings) start with a colon.
                        # These are not JSON data payload and should be skipped.
                        if chunk_bytes.startswith(":"):
                            continue

                        chunk = chunk_bytes.removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    # Use conversation manager to update conversation
                    conversation_manager.add_gpt_message(request_func_input.conversation_id, 
                                                         generated_text, 
                                                         output.output_tokens)
                    output.success = True
                    output.latency = most_recent_timestamp - st
                    #print(request_func_input.conversation_id, request_func_input.turn_id,
                    #    len(output.itl), request_func_input.output_len, output.latency)
                else:
                    output.error = response.reason or ""
                    output.success = False
                    error_text = await response.text()
                    # Use conversation manager to update conversation with error
                    conversation_manager.add_gpt_message(request_func_input.conversation_id, 
                                                         "Error", 
                                                         0)
                    request_stats.add_error(f"HTTP_{response.status}_{response.reason or 'Unknown'}")
                    # print("Response status:", response.status)
                    # print("Response headers:", response.headers)
                    # print("Response body:", error_text)
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            request_stats.add_error("Exception")
            logger.info(output.error)
        
        request_stats.complete_request()
        
        # Add request to statistics
        actual_output_len = output.output_tokens if output.output_tokens else -1
        if actual_output_len != request_func_input.output_len:
            logger.info(f"[Bug] Mismatch output_len: {request_func_input.output_len}, "
                  f"actual_output_len: {actual_output_len}"
            )
        logger.info(f"Done conv_id: {request_func_input.conversation_id}, "
              f"output_len: {request_func_input.output_len}, "
              f"turn_id: {request_func_input.turn_id}"
        )
        if output.success:    
            request_stats.add_request(
                prompt_len=request_func_input.prompt_len,
                message_len=message_length,
                output_len=actual_output_len,
                waiting_time=waiting_time,
                is_follow_up=is_follow_up
            )
        
        # Print comprehensive statistics every 100 requests
        if request_stats.should_print_statistics():
            request_stats.print_statistics(api_url)

    # if pbar:
    #     pbar.update(1)
    return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            model_path = snapshot_download(
                model_id=pretrained_model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])

            return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    if tokenizer_mode == "mistral":
        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer
        except ImportError as e:
            raise ImportError("MistralTokenizer requires vllm package.\n"
                              "Please install it with `pip install vllm` "
                              "to use mistral tokenizer mode.") from e
        return MistralTokenizer.from_pretrained(
            str(pretrained_model_name_or_path))
    else:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}