# LMCache Metrics Collection & Processing System

This document explains how metrics are collected, processed, and analyzed across the LMCache and the benchmark infrastructure.

## 🏗️ Architecture Components

### 1. Core Metrics Collection (`lmcache/observability.py`)

**Primary Classes:**
- `LMCStatsMonitor`: Thread-safe statistics collector
- `ConversationStats`: Conversation-specific metrics storage
- `LMCacheStatsHandler`: HTTP endpoint for metrics serving

**Key Metrics Collected:**
```python
# Cache Performance
- lookup_requests: int
- lookup_tokens: int  
- hit_tokens: int
- hit_rate: float

# Conversation Analytics
- conversation_requests: int
- continue_turns: int (follow-up conversations)
- avg_interval: float (time between turns)
- avg_query_length: float
- avg_response_length: float
```

**HTTP Endpoint:**
- **URL**: `http://localhost:{9000 + CUDA_device_id}/lmcache/stats`
- **Method**: GET
- **Response**: JSON with `lmcache_stats` and `conversation_stats`

### 2. Conversation Feature Extraction (`lmcache/v1/conversation.py`)

**Purpose**: Analyzes token sequences to extract conversation patterns

**Key Components:**
```python
ConversationTurnDetector:
  - detect_conversation_turns(): Counts user/assistant turns
  - update_meta_and_get_feature(): Extracts conversation features
  - _extract_turn_lengths(): Measures query/response lengths

ConversationFeature:
  - turn_number: int
  - previous_interval: float  
  - CustomerQueryLength: int
  - GenerativeModelResponseLength: int
  - cumulative_time: float
```

**Data Flow:**
1. **Token Input** → Conversation detector analyzes structure
2. **Pattern Recognition** → Identifies user/assistant boundaries using chat templates
3. **Feature Extraction** → Computes temporal and length statistics
4. **State Tracking** → Maintains conversation history across turns

### 3. Cache Engine Integration (`lmcache/v1/cache_engine.py`)

**Metrics Collection Points:**
```python
# In LMCacheEngine class:
def lookup(tokens) -> int:
    # Records lookup requests and hit tokens
    self.stats_monitor.on_lookup(request_id, num_tokens, matched_tokens, feature)

def store(tokens, **kwargs):
    # Tracks store operations
    monitor_req_id = self.stats_monitor.on_store_request(num_tokens)
    # ... storage logic ...
    self.stats_monitor.on_store_finished(monitor_req_id, stored_tokens)

def retrieve(tokens, **kwargs) -> torch.Tensor:
    # Monitors retrieve operations
    monitor_req_id = self.stats_monitor.on_retrieve_request(num_tokens)
    # ... retrieval logic ...
    self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
```

**Statistics Server:**
- Started automatically when LMCacheEngine initializes
- Port calculation: `9000 + CUDA_device_id` (from `CUDA_VISIBLE_DEVICES`)
- Thread-safe HTTP server serving JSON metrics

### 4. Client-Side Metrics (`vllm_cache_bench/v1/client/backend_request_func.py`)

**Client Statistics Collection:**
```python
class RequestStatistics:
    - total_requests: int
    - running_requests: int  
    - total_follow_up_turns: int
    - prompt_lengths: List[int]
    - message_lengths: List[int]
    - waiting_times: List[float]
    - error_counts: Dict[str, int]
```

### 5. Client Getting Stats from the Server (`PrefixCacheInternProject/vllm_cache_bench/v1/client/print_statistics.py`)

```python
def print_cache_statistics(api_url: str):
    # Fetches vLLM metrics from /metrics endpoint
    # Fetches LMCache metrics from /lmcache/stats endpoint
    # Outputs structured CLIENT_STATISTICS_BEGIN/END blocks
```

**Output Format:**
```
CLIENT_STATISTICS_BEGIN
vllm_queries: 1000
vllm_hits: 750
vllm_hit_rate: 0.7500
lmcache_requests: 500
lmcache_queries: 45000
lmcache_hits: 32000
lmcache_hit_rate: 0.7111
conversation_requests (conversational): 500
conversation_nonzero_turn_ratio: 0.4200
CLIENT_STATISTICS_END
```

### 6. Log Analysis & Processing (`vllm_cache_bench/v1/execution/log_analyzer.py`)

**LogAnalyzer Class:**
```python
def extract_metrics_from_logs(config, client_log_path, server_log_path):
    # Parses structured log output
    # Extracts benchmark results, cache statistics, conversation features
    # Returns ExperimentMetrics dataclass

def _parse_client_log(client_log_path):
    # Regex patterns to extract CLIENT_STATISTICS_BEGIN/END blocks
    # Separates cache stats vs conversation features
    # Handles type conversion (int/float/string)
```

**Metrics Categories:**
- **Benchmark Results**: Throughput, latency, token counts
- **Cache Statistics**: Hit rates, query patterns
- **Conversation Features**: Turn patterns, temporal analysis

### 7. Experiment Orchestration (`vllm_cache_bench/v1/execution/runner.py`)

**ExperimentExecutor:**
```python
async def execute_experiment(config):
    # 1. Start server process with logging
    # 2. Execute client benchmark
    # 3. Analyze logs and extract metrics
    # 4. Save metrics to JSON
```

**Data Persistence:**
- Saves to `summary.json` in experiment log directory  
- Appends metrics from multiple experiment runs
- Structured JSON format for downstream analysis

## 🔄 Complete Data Flow

### 1. **Runtime Collection**
```
LMCache Operations → LMCStatsMonitor → HTTP Stats Server
    ↓
Conversation Analysis → ConversationFeature → Statistics Aggregation
```

### 2. **Client Monitoring**
```
vLLM Requests → RequestStatistics → Periodic Statistics Printing
```

### 3. **Post-Processing**
```
Raw Logs → LogAnalyzer → ExperimentMetrics → JSON Storage
```

## 📍 Key Collection Points

### A. **Cache Engine Level** (`cache_engine.py`)
- **When**: Every lookup/store/retrieve operation
- **What**: Token counts, hit rates, timing
- **Storage**: In-memory `LMCStatsMonitor`

### B. **HTTP Stats Server** (`observability.py`)
- **When**: On-demand via HTTP GET requests
- **What**: Aggregated statistics, conversation analytics  
- **Format**: JSON response with nested categories

### C. **Client Request Level** (`backend_request_func.py`)
- **When**: Every 100 requests during benchmark
- **What**: vLLM + LMCache metrics, request patterns
- **Output**: Structured log blocks for parsing

### D. **Conversation Analysis** (`conversation.py`)
- **When**: During cache lookup operations
- **What**: Turn detection, temporal patterns, length analysis
- **Context**: Maintains conversation state across turns