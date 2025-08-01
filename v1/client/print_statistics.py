#!/usr/bin/env python3
"""
Cache Statistics Utilities

This module provides functionality to fetch and print cache statistics
from vLLM and LMCache services.
"""

import json
import os
import re
import requests
from typing import Tuple, Dict, Any


def print_vllm_statistics(api_url: str) -> float:
    """
    Fetch and print vLLM cache statistics
    
    Args:
        api_url: The API URL to derive metrics endpoints from
        
    Returns:
        vllm_hit_ratio: The hit ratio from vLLM cache
    """
    vllm_stats = {
        'queries': 0,
        'hits': 0,
        'hit_ratio': 0.0,
        'preemptions': 0
    }
    
    try:
        # Get vLLM metrics
        metrics_url = api_url.replace('v1/chat/completions', 'metrics')
        response = requests.get(metrics_url, timeout=5)
        
        for line in response.text.split("\n"):
            if "prefix_cache_queries_total{" in line:
                vllm_stats['queries'] = int(float(line.split(' ')[-1]))
            elif "prefix_cache_hits_total{" in line:
                vllm_stats['hits'] = int(float(line.split(' ')[-1]))
            elif "num_preemptions_total{" in line:
                vllm_stats['preemptions'] = int(float(line.split(' ')[-1]))
        
        # Calculate vLLM hit ratio
        if vllm_stats['queries'] > 0:
            vllm_stats['hit_ratio'] = vllm_stats['hits'] / vllm_stats['queries']
            
    except Exception as e:
        print(f"Could not fetch vLLM statistics: {e}")
    
    # Print vLLM statistics
    print(f"vllm_queries: {vllm_stats['queries']}")
    print(f"vllm_hits: {vllm_stats['hits']}")
    print(f"vllm_hit_rate: {vllm_stats['hit_ratio']:.4f}")
    print(f"vllm_preemptions: {vllm_stats['preemptions']}")
    
    return vllm_stats['hit_ratio']


def print_lmcache_statistics(api_url: str) -> Dict[str, Any]:
    """
    Fetch and print LMCache statistics and conversation analytics
    
    Args:
        api_url: The API URL to derive metrics endpoints from
        
    Returns:
        Dictionary containing LMCache statistics
    """    
    try:
        # Try to get LMCache statistics using CUDA device-based port calculation
        lmcache_port = None
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible and cuda_visible != '':
            # Parse device IDs and calculate port
            device_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
            if device_ids:
                lmcache_port = 9000 + device_ids[0]
        else:
            lmcache_port = 9000
        
        # Only try to fetch LMCache statistics if we found a valid port
        if lmcache_port:
            # Extract host from api_url
            host_match = re.search(r'(https?://[^:/]+)', api_url)
            if host_match:
                host_url = host_match.group(1)
                lmcache_stats_url = f"{host_url}:{lmcache_port}/lmcache/stats"
                
                lmcache_response = requests.get(lmcache_stats_url, timeout=2)
                if lmcache_response.status_code == 200:
                    lmcache_data = lmcache_response.json()
                    
                    # Extract LMCache stats from the proper wrapper
                    lmcache_stats = lmcache_data.get('lmcache_stats', {})
                    conversation_stats = lmcache_data.get('conversation_stats', {})
                else:
                    print("LMCache statistics 404")
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError):
        print("LMCache statistics error")
    
    # Print LMCache statistics
    print(f"lmcache_requests: {lmcache_stats['requests']}")
    print(f"lmcache_queries: {lmcache_stats['query_tokens']}")
    print(f"lmcache_hits: {lmcache_stats['hit_tokens']}")
    print(f"lmcache_retrieved_tokens: {lmcache_stats['retrieved_tokens']}")
    print(f"lmcache_hit_rate: {lmcache_stats['hit_rate']:.4f}")
    print(f"lmcache_retrieved_rate: {lmcache_stats['retrieved_rate']:.4f}")
    
    # Output conversation analytics in parseable format
    if conversation_stats:
        for key, value in conversation_stats.items():
            print(f"conversation_{key}: {value}")


def print_cache_statistics(api_url: str) -> float:
    """
    Fetch and print prefix cache statistics and conversation analytics
    
    Args:
        api_url: The API URL to derive metrics endpoints from
        
    Returns:
        vllm_hit_ratio: The hit ratio from vLLM cache
    """
    # Output structured statistics for log analysis
    print("\n" + "="*50)
    print("CLIENT_STATISTICS_BEGIN")
    
    # Print vLLM statistics
    vllm_hit_ratio = print_vllm_statistics(api_url)
    
    # Print LMCache statistics
    print_lmcache_statistics(api_url)
    
    print("CLIENT_STATISTICS_END")
    print("="*50)
    print("-"*50)
    
    return vllm_hit_ratio
