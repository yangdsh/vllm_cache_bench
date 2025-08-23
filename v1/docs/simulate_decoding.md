# CSV to Logs Converter Tool - Simulate vLLM decoding

## Overview
Developed a Python utility (`csv_to_logs_converter.py`) to convert conversation data from CSV format into server and client log files compatible with our cache simulation system.

## Purpose
- **Problem**: Need to simulate cache behavior using conversation data from CSV exports
- **Solution**: Convert CSV data into the specific log formats expected by our cache simulator
- **Impact**: Enables testing cache policies against real conversation patterns without requiring live server logs

## Key Features

### Core Functionality
- **Input**: CSV files with conversation data (timestamps, conversation IDs, query/response lengths)
- **Output**: Paired `server_{model}.log` and `client_{model}.log` files
- **Model Integration**: Incorporates model names into output filenames for organized simulation runs

### Configurable Parameters
- **Token Generation Rate**: Configurable decode speed (default: 50 tokens/second) for realistic timing simulation
- **Output Directory**: Flexible output location for different experiment setups
- **Model Names**: Support for different model identifiers in filename structure

### Recent Enhancements
- **Robust Data Handling**: Added support for empty/missing values in CSV data
  - Default to 50 tokens for empty query lengths
  - Default to 128 tokens for empty response lengths
- **Cumulative Token Tracking**: Implements proper conversation context modeling where input tokens represent cumulative conversation history
- **Statistics Reporting**: Tracks and reports data quality metrics (empty fields count)

## Technical Implementation
- **Timestamp Handling**: Preserves original CSV timestamps for send events, calculates done events based on response length and decode rate
- **Conversation Management**: Maps UUID conversation IDs to numeric format with proper turn tracking
- **Log Format Compliance**: Generates logs matching exact patterns expected by existing parsing infrastructure

## Usage Examples
```bash
# Basic usage
python src/simulation/csv_to_logs_converter.py data/conversations.csv qwen3-8b

# With custom generation rate
python src/simulation/csv_to_logs_converter.py data/conversations.csv llama-8b --tokens-per-second 75
```

## Business Value
- **Testing Capability**: Enables cache simulation testing without live system dependencies
- **Flexibility**: Supports various models and generation speeds for comprehensive testing scenarios  
- **Data Quality**: Handles real-world data inconsistencies gracefully
- **Integration**: Seamlessly works with existing cache simulation infrastructure

## Next Steps
Tool is production-ready and actively used for cache policy evaluation and performance testing.
