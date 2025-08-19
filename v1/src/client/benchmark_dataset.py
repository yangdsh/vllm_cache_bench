# SPDX-License-Identifier: Apache-2.0
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena

TODO: Implement CustomDataset to parse a JSON file and convert its contents into
SampleRequest instances, similar to the approach used in ShareGPT.
"""

import base64
import io
import heapq
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer

import sys
import os
# Add the parent directory to the path so we can import util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import conversation utilities
from utils.conversation_manager import ConversationManager

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Union[MultiModalDataDict, dict]] = None
    lora_request: Optional[LoRARequest] = None
    conversation_id: int = -1
    turn_id: int = -1
    timestamp: int = 0
    next_timestamp: int = 1e9
    interval: int = 0


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0

    # num_requests has default 1000 in both the benchmark_serving.py and
    # benchmark_throughput.py

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.data = None

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.
        
        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.
        
        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    def get_random_lora_request(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Optionally select a random LoRA request and return its associated
        tokenizer.
        
        This method is used when LoRA parameters are provided.  It randomly
        selects a LoRA based on max_loras and retrieves a cached tokenizer for
        that LoRA if available. Otherwise, it returns the base tokenizer.
        
        Args:
            tokenizer (PreTrainedTokenizerBase): The base tokenizer to use if no
            LoRA is selected.  max_loras (Optional[int]): The maximum number of
            LoRAs available. If None, LoRA is not used.  lora_path
            (Optional[str]): Path to the LoRA parameters on disk. If None, LoRA
            is not used.
        
        Returns:
            tuple[Optional[LoRARequest], AnyTokenizer]: A tuple where the first
            element is a LoRARequest (or None if not applicable) and the second
            element is the tokenizer associated with the LoRA request (or the
            base tokenizer).
        """
        if max_loras is None or lora_path is None:
            return None, tokenizer

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, max_loras)
        lora_request = LoRARequest(
            lora_name=str(lora_id),
            lora_int_id=lora_id,
            lora_path=lora_path_on_disk(lora_path),
        )
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        # Return lora_request and the cached tokenizer if available; otherwise,
        # return the base tokenizer
        return lora_request, lora_tokenizer_cache[lora_id] or tokenizer

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.
        
        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.
        
        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.
        
        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: dict[int, AnyTokenizer] = {}


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    For a PIL.Image.Image input:
      - Converts the image to RGB.
      - Saves the image as a JPEG in-memory.
      - Encodes the JPEG data as a base64 string.
      - Returns a dictionary with the image as a base64 data URL.

    For a string input:
      - Treats the string as a URL or file path.
      - Prepends "file://" if the string doesn't start with "http://" or
        "file://".
      - Returns a dictionary with the image URL.

    Raises:
      ValueError: If the input is neither a PIL.Image.Image nor a string.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

    if isinstance(image, str):
        image_url = (image if image.startswith(
            ("http://", "file://")) else f"file://{image}")
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image or str.")


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 1.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               prefix_len: int = DEFAULT_PREFIX_LEN,
               range_ratio: float = DEFAULT_RANGE_RATIO,
               input_len: int = DEFAULT_INPUT_LEN,
               output_len: int = DEFAULT_OUTPUT_LEN,
               **kwargs) -> list[SampleRequest]:

        vocab_size = tokenizer.vocab_size

        prefix_token_ids = (np.random.randint(
            0, vocab_size, size=prefix_len).tolist() if prefix_len > 0 else [])

        input_low = int(input_len * range_ratio)
        output_low = int(output_len * range_ratio)

        input_lens = np.random.randint(input_low,
                                       input_len + 1,
                                       size=num_requests)
        output_lens = np.random.randint(output_low,
                                        output_len + 1,
                                        size=num_requests)
        offsets = np.random.randint(0, vocab_size, size=num_requests)

        requests = []
        for i in range(num_requests):
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) %
                         vocab_size).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            # After decoding the prompt we have to encode and decode it again.
            # This is done because in some cases N consecutive tokens
            # give a string tokenized into != N number of tokens.
            # For example for GPT2Tokenizer:
            # [6880, 6881] -> ['Ġcalls', 'here'] ->
            # [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
            # To avoid uncontrolled change of the prompt length,
            # the encoded sequence is truncated before being decode again.
            re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[
                : input_lens[i]
            ]
            prompt = tokenizer.decode(re_encoded_sequence)
            total_input_len = len(re_encoded_sequence)
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                ))
        return requests


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()
        self.conversation_id = 0

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        if '.json' in self.dataset_path:
            with open(self.dataset_path, encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            ds = load_dataset(self.dataset_path)
            self.data = ds["train"].to_pandas().to_dict(orient="records")
        if 'chatbot_arena' in self.dataset_path:
            self.conv_tag = 'conversation_a'
            self.value_tag = 'content'
            self.role_tag = 'role'
        elif 'lmsys' in self.dataset_path:
            self.conv_tag = 'conversation'
            self.value_tag = 'content'
            self.role_tag = 'role'
        else: # sharegpt
            self.conv_tag = 'conversations'
            self.value_tag = 'value'
            self.role_tag = 'from'
        
        # Filter entries with at least two conversation turns.
        self.data = [
            entry for entry in self.data
            if self.conv_tag in entry and len(entry[self.conv_tag]) >= 2
        ]
        random.seed(self.random_seed)
        # random.shuffle(self.data)

    def sample(self,
               tokenizer: 'PreTrainedTokenizerBase', # Use quotes if class not defined yet
               num_requests: int,
               lora_path: Optional[str] = None,
               max_loras: Optional[int] = None,
               output_len: Optional[int] = None,
               conv_scale: float = 0.25, # Avg time between start of conversations
               req_scale: float = 10,    # Avg time between requests within a conv
               human_delay: float = 0,   # Fixed delay added for human turns
               max_active_conversations: int = 100,
               time_limit: int = 10000,
               **kwargs) -> list:
        samples: list = []
        # Stores the timestamp of the *last* request for potentially active conversations.
        # Using a min-heap allows efficient retrieval of the earliest finish time.
        active_conv_finish_times = [] # Store finish times as a min-heap
        self.conversation_id = 0 # Ensure conversation ID starts fresh
        last_conv_start_timestamp = 0.0 # Track the start time of the previous conv

        for entry_index, entry in enumerate(self.data):
            if len(samples) >= num_requests:
                print('got enough samples')
                break
            if last_conv_start_timestamp > time_limit:
                print('replay timestamp reached limit')
                break

            # --- Calculate potential start time for this new conversation ---
            # Start with a base time slightly after the previous conversation *started*,
            # plus the random interval.
            potential_conv_timestamp = last_conv_start_timestamp + np.random.exponential(conv_scale)

            # --- Prune conversations that are no longer active ---
            # Remove finish times from the heap that are <= the *potential* start time
            # of the new conversation. These conversations have finished before the new one might start.
            while active_conv_finish_times and active_conv_finish_times[0] <= potential_conv_timestamp:
                heapq.heappop(active_conv_finish_times)

            # --- Check if active conversation limit is reached ---
            while len(active_conv_finish_times) >= max_active_conversations:
                # Wait until the earliest active conversation finishes.
                earliest_finish_time = heapq.heappop(active_conv_finish_times)
                # The new conversation cannot start before this time.
                potential_conv_timestamp = max(potential_conv_timestamp, earliest_finish_time)
                # Re-prune based on the potentially updated (later) start time.
                # (This loop handles cases where multiple conversations finish at nearly the same time)
                while active_conv_finish_times and active_conv_finish_times[0] <= potential_conv_timestamp:
                   heapq.heappop(active_conv_finish_times)

            # --- Finalize start time for this conversation ---
            conv_timestamp = potential_conv_timestamp
            last_conv_start_timestamp = conv_timestamp # Update for the next iteration
            req_timestamp = conv_timestamp # Timestamp for the *first* request

            i = 0 # Start checking from the first turn in the entry
            turn_id = 0 # Counter for valid turns found in this entry
            last_req_timestamp = conv_timestamp
            total_output_len = 0

            # --- Inner loop to process turns within the current conversation entry ---
            while True:
                # --- Termination conditions for inner loop ---
                if len(samples) >= num_requests:
                    break # Exit inner loop, outer loop will handle final break
                #if turn_id >= 3: # MAX_TURNS=3 limit per conversation entry
                #    break
                if i >= len(entry[self.conv_tag]) - 1:
                    break # No more pairs possible in this entry

                # --- Core Logic: Check if 'i' is user and 'i+1' is not user ---
                is_current_user = entry[self.conv_tag][i][self.role_tag] in ['human', 'user']
                is_next_not_user = (i + 1 < len(entry[self.conv_tag])) and \
                                   (entry[self.conv_tag][i + 1][self.role_tag] not in ['human', 'user'])

                if is_current_user and is_next_not_user:
                    prompt = entry[self.conv_tag][i][self.value_tag]
                    # ---- microbenchmark controlling hit ratios ----
                    # the input prompt contains 'hello ' repeated 1000 times, which is 1000 tokens
                    #prompt += prompt
                    #prompt = prompt[:1500*6] + str(len(samples)) + prompt[:1000*6] # context and new prompt
                    #if len(samples) % 10 < 8:
                    #    prompt = str(len(samples)) + prompt # this prompt is going to miss entirely
                    # -----------------------------------------------
                    completion = entry[self.conv_tag][i + 1][self.value_tag]

                    lora_request, current_tokenizer = self.get_random_lora_request( # Use current_tokenizer
                        tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)
                    prompt_ids = current_tokenizer(prompt).input_ids
                    completion_ids = current_tokenizer(completion).input_ids
                    prompt_len = len(prompt_ids)
                    new_output_len = len(completion_ids)
                    if output_len is not None:
                        new_output_len += output_len
                    total_output_len += new_output_len

                    if "timestamp" in entry[self.conv_tag][i]:
                        req_timestamp = conv_timestamp + entry[self.conv_tag][i]["timestamp"] * req_scale
                    # Create and add the sample
                    current_sample = SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=new_output_len,
                        lora_request=lora_request,
                        conversation_id=self.conversation_id,
                        turn_id=turn_id,
                        timestamp=req_timestamp, # Assign current request timestamp
                        interval=req_timestamp-last_req_timestamp
                    )
                    samples.append(current_sample)
                    last_req_timestamp = req_timestamp # Update last timestamp for *this* entry
                    turn_id += 1

                    # --- Update timestamp for the *next* request *within this conversation* ---
                    req_timestamp += np.random.exponential(req_scale)

                    # Store the calculated timestamp for the *next* request in the *current* sample
                    if i + 2 < len(entry[self.conv_tag]):
                        if "timestamp" in entry[self.conv_tag][i+2]:
                            req_timestamp = conv_timestamp + entry[self.conv_tag][i+2]["timestamp"] * req_scale
                        samples[-1].next_timestamp = req_timestamp

                    # Advance index past the processed pair
                    i += 2
                else:
                    # Condition not met, advance to the next potential start turn
                    i += 1
                if req_timestamp > time_limit:
                    break
            # --- End of inner while loop (processing turns for one entry) ---

            # If samples were generated for this conversation, record its finish time
            if last_req_timestamp >= 0: # assume token throughput is 30
                heapq.heappush(active_conv_finish_times, last_req_timestamp + total_output_len / 30)
            # Increment conversation ID for the next entry
            self.conversation_id += 1

        # --- End of outer for loop (processing entries) ---

        print(f"Finished processing {entry_index+1} entries. Generated {len(samples)} samples.", flush=True)

        # Sort all collected samples by their request timestamp
        samples.sort(key=lambda x: x.timestamp)
        # --- Filter out samples with timestamp ---
        original_count = len(samples)
        samples = [s for s in samples if s.timestamp <= time_limit]
        filtered_count = len(samples)
        if original_count != filtered_count:
             print(f"{original_count - filtered_count} samples with timestamp > {time_limit}.")
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(self,
               tokenizer,
               num_requests: int,
               prefix_len: int = DEFAULT_PREFIX_LEN,
               input_len: int = DEFAULT_INPUT_LEN,
               output_len: int = DEFAULT_OUTPUT_LEN,
               return_prompt_formatted: bool = False,
               **kwargs) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens)
                      for tokens in \
                        tokenized_lines) / len(tokenized_lines)

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(base_msg,
                                                 add_generation_prompt=True,
                                                 tokenize=False)
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset}).")

        # Determine how many poem lines to use.
        num_input_lines = round((input_len - base_offset) / avg_len)
        num_prefix_lines = round((prefix_len - base_offset) / avg_len)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        for _ in range(num_requests):
            extra_lines = random.choices(self.data,
                                         k=num_input_lines - num_prefix_lines)
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            samples.append(
                SampleRequest(
                    prompt=prompt_formatted
                    if return_prompt_formatted else prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.  Loads data from a CSV file and generates
    sample requests based on synthetic prompt generation. Only rows with Model
    "GPT-4" and positive response tokens are used.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self, ):
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        df = pd.read_csv(self.dataset_path)
        # Filter to keep only GPT-4 rows.
        gpt4_df = df[df["Model"] == "GPT-4"]
        # Remove failed requests (where Response tokens is 0 or less).
        gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
        # Sample the desired number of rows.
        self.data = gpt4_df

    def _sample_loaded_data(self, num_requests: int) -> list:
        if num_requests <= len(self.data):
            data = self.data.sample(n=num_requests,
                                    random_state=self.random_seed)
        else:
            data = self.data.sample(
                n=num_requests,
                random_state=self.random_seed,
                replace=True,
            )
        # Convert the dataframe to a list of lists.
        return data.values.tolist()

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               max_loras: Optional[int] = None,
               lora_path: Optional[str] = None,
               **kwargs) -> list[SampleRequest]:
        samples = []
        data = self._sample_loaded_data(num_requests=num_requests)
        for i in range(num_requests):
            input_len = int(data[i][2])
            output_len = int(data[i][3])
            lora_req, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)
            vocab_size = tokenizer.vocab_size
            # Generate a synthetic prompt: a list of token IDs computed as (i +
            # j) modulo vocab_size.
            token_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    lora_request=lora_req,
                ))
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Implementation
# -----------------------------------------------------------------------------


class HuggingFaceDataset(BenchmarkDataset):
    """
    Dataset class for processing a HuggingFace dataset with conversation data
    and optional images.
    """
    DEFAULT_NUM_REQUESTS = 1000

    def __init__(
        self,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset

        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided for loading data.")

        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )

        if "conversations" not in self.data.features:
            raise ValueError("HF Dataset must have a 'conversations' column.")

        # Shuffle and filter examples with at least 2 conversations.
        self.data = self.data.shuffle(seed=self.random_seed).filter(
            lambda x: len(x["conversations"]) >= 2)

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               lora_path: Optional[str] = None,
               max_loras: Optional[int] = None,
               output_len: Optional[int] = None,
               **kwargs) -> list:
        sampled_requests = []
        dynamic_output = output_len is None

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break

            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            lora_request, tokenizer = self.get_random_lora_request(
                tokenizer, lora_path=lora_path, max_loras=max_loras)

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(
                    prompt_len, completion_len):
                continue

            mm_content = process_image(
                item["image"]) if "image" in item else None
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    lora_request=lora_request,
                ))
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(BenchmarkDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    DEFAULT_NUM_REQUESTS = 1000
    VISION_ARENA_DATASET_PATH = "lmarena-ai/vision-arena-bench-v0.1"

    def __init__(
        self,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset

        if self.dataset_path != self.VISION_ARENA_DATASET_PATH:
            raise ValueError(f"Only support Vision Arena dataset.\
                    This data path {self.dataset_path} is not valid.")
        if self.dataset_subset is None and self.dataset_split != "train":
            raise ValueError("Dataset split must be 'train'.")

        self.load_data()

    def load_data(self) -> None:
        dataset = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = dataset.shuffle(seed=self.random_seed)

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: int = DEFAULT_OUTPUT_LEN,
               **kwargs) -> list:
        # TODO (jenniferzhao): Add support for offline benchmark sampling
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["turns"][0][0]["content"]
            prompt_len = len(tokenizer(prompt).input_ids)
            mm_content = process_image(item["images"][0])
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                ))
        return sampled_requests


# -----------------------------------------------------------------------------
# Conversational CSV Dataset Implementation
# -----------------------------------------------------------------------------


class ConversationalCSVDataset(BenchmarkDataset):
    """
    Implements a dataset from conversational CSV file with conversation data structure.
    Uses real timestamps from the CSV but generates random tokens for content.
    Preserves the original order of requests from the trace.
    
    Expected CSV format:
    @timestamp,ConversationId,CustomerQueryLength,GenerativeModelResponseLength
    """

    def __init__(self, mock_decoding: bool = False, **kwargs) -> None:
        self.mock_decoding = mock_decoding
        super().__init__(**kwargs)
        self.load_data()
        self.conversation_manager = None

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading conversational CSV data.")
        
        # Load CSV data
        df = pd.read_csv(self.dataset_path)
        
        # Rename columns to standardize
        df = df.rename(columns={
            '@timestamp': 'timestamp',
            'ConversationId': 'conversation_id',
            'CustomerQueryLength': 'prompt_tokens',
            'GenerativeModelResponseLength': 'response_tokens'
        })
        
        # Parse timestamp to Unix timestamp (seconds since epoch)
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Fill missing response_tokens with 128 and convert to int
        df['response_tokens'] = df['response_tokens'].fillna(128).astype(int)
        df['prompt_tokens'] = df['prompt_tokens'].fillna(0).astype(int)
        
        # Filter out rows with invalid data (only filter out prompt_tokens <= 0)
        df = df[df['prompt_tokens'] > 0]
        
        # Sort by timestamp to preserve original order
        df = df.sort_values('timestamp')
        
        # Convert to list of request data
        self.data = []
        for _, row in df.iterrows():
            self.data.append({
                'timestamp': row['timestamp'],
                'conversation_id': row['conversation_id'],
                'prompt_tokens': row['prompt_tokens'],
                'response_tokens': row['response_tokens']
            })
        
        random.seed(self.random_seed)

    def sample(self,
               tokenizer: 'PreTrainedTokenizerBase',
               num_requests: int,
               lora_path: Optional[str] = None,
               max_loras: Optional[int] = None,
               output_len: Optional[int] = None,
               conv_scale: float = 0.25,  # Ignored - kept for compatibility
               req_scale: float = 10,
               human_delay: float = 0,    # Ignored - kept for compatibility
               max_active_conversations: int = 100,  # Ignored - kept for compatibility
               time_limit: int = 10000,
               **kwargs) -> list:
        if self.conversation_manager is None:
            self.conversation_manager = ConversationManager(tokenizer)
        
        samples: list = []
        vocab_size = tokenizer.vocab_size
        
        # Get the earliest timestamp to use as baseline
        if not self.data:
            raise ValueError("No valid data found in dataset")
            
        base_timestamp = min(req['timestamp'] for req in self.data)
        
        # Combined conversation tracking: maps original_id -> {'int_id': int, 'turn_count': int}
        conversation_tracker = {}
        next_conversation_int_id = 1
        
        # Process requests in original order
        for i, req_data in enumerate(self.data):
            if len(samples) >= num_requests:
                break
                
            # Use real timestamp, adjusted to baseline and scaled by req_scale
            req_timestamp = (req_data['timestamp'] - base_timestamp) * req_scale
            
            if req_timestamp > time_limit:
                break
            
            # Get token lengths from the data
            prompt_tokens = int(req_data['prompt_tokens'])
            response_tokens = int(req_data['response_tokens'])
            
            if output_len is not None:
                response_tokens = output_len
            
            # --- MOCK DECODING LOGIC ---
            if self.mock_decoding:
                prompt_tokens = prompt_tokens + response_tokens
                response_tokens = 1
            # --------------------------
            
            # Generate random prompt with the specified token length
            lora_request, current_tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)

            # Track conversation and turns
            original_conversation_id = req_data['conversation_id']
            int_conversation_id, turn_id = self._track_conversation(
                original_conversation_id, conversation_tracker, next_conversation_int_id)
            
            if int_conversation_id == next_conversation_int_id:
                next_conversation_int_id += 1
            
            # Generate prompt using sonnet text - use conversation utility function
            prompt = self.conversation_manager.generate_prompt( 
                prompt_tokens, 
                int_conversation_id
            )
            actual_prompt_len = len(current_tokenizer.encode(prompt, add_special_tokens=False))
            
            current_sample = SampleRequest(
                prompt=prompt,
                prompt_len=actual_prompt_len,
                expected_output_len=response_tokens,
                lora_request=lora_request,
                conversation_id=int_conversation_id,
                turn_id=turn_id,
                timestamp=int(req_timestamp)
            )
            samples.append(current_sample)

        return samples

    def _track_conversation(self, original_conversation_id: str, conversation_tracker: dict, 
                          next_int_id: int) -> tuple[int, int]:
        """Track conversation and return (mapped_id, turn_id)."""
        if original_conversation_id not in conversation_tracker:
            # New conversation
            conversation_tracker[original_conversation_id] = {
                'int_id': next_int_id,
                'turn_id': 1
            }
            return next_int_id, 1
        else:
            # Existing conversation - increment turn count
            conversation_tracker[original_conversation_id]['turn_id'] += 1
            return (conversation_tracker[original_conversation_id]['int_id'],
                   conversation_tracker[original_conversation_id]['turn_id'])
