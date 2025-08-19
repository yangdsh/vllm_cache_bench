#!/usr/bin/env python3
"""
Hit Token Comparison Tool

This script compares hit tokens between replay_sim.log and server logs
"""

import re
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class HitTokenComparator:
    def __init__(self, sim_log: str, server_log: str):
        self.sim_log = sim_log
        self.server_log = server_log
        self.sim_lines = []
        self.server_lines = []
        
    def load_logs(self):
        """Load both log files."""
        print("Loading logs...")
        with open(self.sim_log, 'r') as f:
            self.sim_lines = f.readlines()
        with open(self.server_log, 'r') as f:
            self.server_lines = f.readlines()
        print(f"Sim log: {len(self.sim_lines)} lines")
        print(f"Server log: {len(self.server_lines)} lines")
    
    def extract_sim_entries(self) -> List[Dict]:
        """Extract entries from simulator log with eviction information."""
        sim_entries = []
        eviction_events = []
        
        # Pattern for sim log (new format)
        # Format: Send: conv_12_2 all_tokens=440
        #         ✓ Cache HIT (hit tokens: 256)
        # Format: Send: conv_1_1 all_tokens=35
        #         ✗ Cache MISS
        send_pattern = r'Send: conv_(\d+)_(\d+) all_tokens=(\d+)'
        hit_pattern = r'✓ Cache HIT \(hit tokens: (\d+)\)'
        miss_pattern = r'✗ Cache MISS'
        
        # Pattern for eviction events
        eviction_pattern = r'Evicted conv_(\d+)_turn_(\d+)(?:_last_prefill)?, size: (\d+), remaining: (\d+)'
        
        print("Extracting entries from sim log...")
        
        current_send = None
        
        for i, line in enumerate(self.sim_lines):
            # Check for send entries
            send_match = re.search(send_pattern, line)
            if send_match:
                current_send = {
                    'line': i + 1,
                    'conversation_id': int(send_match.group(1)),
                    'turn_number': int(send_match.group(2)),
                    'all_tokens': int(send_match.group(3)),
                    'full_line': line.strip()
                }
                continue
            
            # Check for hit entries
            hit_match = re.search(hit_pattern, line)
            if hit_match and current_send:
                entry = {
                    'line': i + 1,
                    'conversation_id': current_send['conversation_id'],
                    'turn_number': current_send['turn_number'],
                    'all_tokens': current_send['all_tokens'],
                    'hit_tokens': int(hit_match.group(1)),
                    'total_tokens': current_send['all_tokens'],
                    'event_type': 'hit',
                    'full_line': line.strip()
                }
                sim_entries.append(entry)
                current_send = None
                continue
            
            # Check for miss entries
            miss_match = re.search(miss_pattern, line)
            if miss_match and current_send:
                entry = {
                    'line': i + 1,
                    'conversation_id': current_send['conversation_id'],
                    'turn_number': current_send['turn_number'],
                    'all_tokens': current_send['all_tokens'],
                    'hit_tokens': 0,
                    'total_tokens': current_send['all_tokens'],
                    'event_type': 'miss',
                    'full_line': line.strip()
                }
                sim_entries.append(entry)
                current_send = None
                continue
            
            # Check for eviction events
            evict_match = re.search(eviction_pattern, line)
            if evict_match:
                eviction = {
                    'line': i + 1,
                    'conversation_id': int(evict_match.group(1)),
                    'turn_number': int(evict_match.group(2)),
                    'size': int(evict_match.group(3)),
                    'remaining': int(evict_match.group(4)),
                    'is_prefill': '_last_prefill' in line,
                    'full_line': line.strip()
                }
                eviction_events.append(eviction)
        
        print(f"Sim entries: {len(sim_entries)}")
        print(f"Eviction events: {len(eviction_events)}")
        
        return sim_entries, eviction_events
    
    def extract_server_entries(self) -> List[Dict]:
        """Extract entries from server log with new format."""
        server_entries = []
        
        # Pattern for server log (new observability format)
        # Format: [Observability] on_lookup: conv_1_1, num_tokens=35, hit_tokens=0, accumulated_hit_tokens=0, lookup_id=...
        server_pattern = r'\[Observability\] on_lookup: conv_(\d+)_(\d+), num_tokens=(\d+), hit_tokens=(\d+), accumulated_hit_tokens=(\d+), lookup_id=([^\s]+)'
        
        print("Extracting entries from server log...")
        for i, line in enumerate(self.server_lines):
            if '[Observability] on_lookup:' in line:
                match = re.search(server_pattern, line)
                if match:
                    entry = {
                        'line': i + 1,
                        'conversation_id': int(match.group(1)),
                        'turn_number': int(match.group(2)),
                        'total_tokens': int(match.group(3)),
                        'hit_tokens': int(match.group(4)),
                        'accumulated_hit_tokens': int(match.group(5)),
                        'lookup_id': match.group(6),
                        'full_line': line.strip()
                    }
                    server_entries.append(entry)
        
        print(f"Server entries: {len(server_entries)}")
        return server_entries
    
    def analyze_eviction_timing(self, sim_entries: List[Dict], eviction_events: List[Dict]):
        """Analyze eviction timing and its impact on cache hits."""
        print("\n" + "="*80)
        print("EVICTION TIMING ANALYSIS")
        print("="*80)
        
        # Create a timeline of events
        timeline = []
        
        # Add cache hits/misses to timeline
        for entry in sim_entries:
            timeline.append({
                'line': entry['line'],
                'type': entry['event_type'],
                'conversation_id': entry['conversation_id'],
                'turn_number': entry['turn_number'],
                'hit_tokens': entry['hit_tokens'],
                'description': f"Cache {entry['event_type'].upper()} for conversation {entry['conversation_id']}, turn {entry['turn_number']}"
            })
        
        # Add evictions to timeline
        for eviction in eviction_events:
            timeline.append({
                'line': eviction['line'],
                'type': 'eviction',
                'conversation_id': eviction['conversation_id'],
                'turn_number': eviction['turn_number'],
                'size': eviction['size'],
                'remaining': eviction['remaining'],
                'description': f"Evicted conversation {eviction['conversation_id']}, turn {eviction['turn_number']} (size: {eviction['size']})"
            })
        
        # Sort timeline by line number
        timeline.sort(key=lambda x: x['line'])
        
        # Analyze eviction patterns
        eviction_by_conversation = defaultdict(list)
        for eviction in eviction_events:
            key = (eviction['conversation_id'], eviction['turn_number'])
            eviction_by_conversation[key].append(eviction)
        
        print(f"Total eviction events: {len(eviction_events)}")
        print(f"Unique conversation-turn pairs evicted: {len(eviction_by_conversation)}")
        
        # Show first 20 timeline events
        print(f"\nFirst 20 timeline events:")
        for i, event in enumerate(timeline[:20]):
            print(f"  {i+1:3d}. Line {event['line']:5d}: {event['description']}")
        
        # Analyze eviction sizes
        eviction_sizes = [e['size'] for e in eviction_events if e['size'] > 0]
        if eviction_sizes:
            print(f"\nEviction size statistics:")
            print(f"  Total evictions with size > 0: {len(eviction_sizes)}")
            print(f"  Average eviction size: {sum(eviction_sizes) / len(eviction_sizes):.0f}")
            print(f"  Largest eviction: {max(eviction_sizes)}")
            print(f"  Smallest eviction: {min(eviction_sizes)}")
        
        return timeline, eviction_by_conversation
    
    def compare_with_eviction_context(self, sim_entries: List[Dict], server_entries: List[Dict], 
                                     eviction_by_conversation: Dict):
        """Compare hit tokens with eviction context."""
        print("\n" + "="*80)
        print("HIT TOKEN COMPARISON WITH EVICTION CONTEXT")
        print("="*80)
        
        # Create lookup dictionaries
        sim_lookup = {}
        for entry in sim_entries:
            key = (entry['conversation_id'], entry['turn_number'])
            sim_lookup[key] = entry
        
        server_lookup = {}
        for entry in server_entries:
            key = (entry['conversation_id'], entry['turn_number'])
            server_lookup[key] = entry
        
        # Compare entries
        matches = []
        mismatches = []
        eviction_related_mismatches = []
        
        for key in sim_lookup:
            if key in server_lookup:
                sim_entry = sim_lookup[key]
                server_entry = server_lookup[key]
                
                if sim_entry['hit_tokens'] == server_entry['hit_tokens']:
                    matches.append((key, sim_entry, server_entry))
                else:
                    mismatches.append((key, sim_entry, server_entry))
                    
                    # Check if this conversation was evicted in simulator
                    if key in eviction_by_conversation:
                        eviction_related_mismatches.append((key, sim_entry, server_entry, eviction_by_conversation[key]))
        
        print(f"Total comparisons: {len(matches) + len(mismatches)}")
        print(f"Matches: {len(matches)}")
        print(f"Mismatches: {len(mismatches)}")
        print(f"Mismatches with eviction context: {len(eviction_related_mismatches)}")
        
        # Analyze eviction-related mismatches
        if eviction_related_mismatches:
            print(f"\nEviction-related mismatches (first 10):")
            for i, (key, sim_entry, server_entry, evictions) in enumerate(eviction_related_mismatches[:10]):
                conv_id, turn_num = key
                diff = sim_entry['hit_tokens'] - server_entry['hit_tokens']
                print(f"  {i+1}. Conversation {conv_id}, Turn {turn_num}:")
                print(f"     Sim: {sim_entry['hit_tokens']} hits")
                print(f"     Server: {server_entry['hit_tokens']} hits, {server_entry['total_tokens']} total")
                print(f"     Difference: {diff:+d} hits")
                print(f"     Evicted {len(evictions)} times in simulator")
                for eviction in evictions:
                    print(f"       - Size: {eviction['size']}, Remaining: {eviction['remaining']}")
        
        return matches, mismatches, eviction_related_mismatches
    
    def analyze_cache_state_differences(self, sim_entries: List[Dict], server_entries: List[Dict], 
                                      eviction_by_conversation: Dict):
        """Analyze differences in cache state between simulator and server."""
        print("\n" + "="*80)
        print("CACHE STATE DIFFERENCE ANALYSIS")
        print("="*80)
        
        # Find mismatches where simulator has more hits
        sim_higher_mismatches = []
        server_higher_mismatches = []
        
        sim_lookup = {(e['conversation_id'], e['turn_number']): e for e in sim_entries}
        server_lookup = {(e['conversation_id'], e['turn_number']): e for e in server_entries}
        
        for key in sim_lookup:
            if key in server_lookup:
                sim_entry = sim_lookup[key]
                server_entry = server_lookup[key]
                
                if sim_entry['hit_tokens'] != server_entry['hit_tokens']:
                    diff = sim_entry['hit_tokens'] - server_entry['hit_tokens']
                    if diff > 0:
                        sim_higher_mismatches.append((key, sim_entry, server_entry, diff))
                    else:
                        server_higher_mismatches.append((key, sim_entry, server_entry, diff))
        
        print(f"Simulator has more hits: {len(sim_higher_mismatches)}")
        print(f"Server has more hits: {len(server_higher_mismatches)}")
        
        # Analyze simulator higher cases
        if sim_higher_mismatches:
            print(f"\nCases where simulator has more hits (first 10):")
            for i, (key, sim_entry, server_entry, diff) in enumerate(sim_higher_mismatches[:10]):
                conv_id, turn_num = key
                was_evicted = key in eviction_by_conversation
                print(f"  {i+1}. Conversation {conv_id}, Turn {turn_num}: +{diff} hits")
                print(f"     Sim: {sim_entry['hit_tokens']} hits")
                print(f"     Server: {server_entry['hit_tokens']} hits")
                print(f"     Evicted in simulator: {was_evicted}")
                if was_evicted:
                    evictions = eviction_by_conversation[key]
                    print(f"     Eviction count: {len(evictions)}")
        
        return sim_higher_mismatches, server_higher_mismatches
    
    def generate_enhanced_report(self):
        """Generate a comprehensive comparison report with eviction analysis."""
        print("="*80)
        print("ENHANCED HIT TOKEN COMPARISON REPORT")
        print("="*80)
        
        self.load_logs()
        
        # Extract entries with eviction information
        sim_entries, eviction_events = self.extract_sim_entries()
        server_entries = self.extract_server_entries()
        
        # Analyze eviction timing
        timeline, eviction_by_conversation = self.analyze_eviction_timing(sim_entries, eviction_events)
        
        # Compare with eviction context
        matches, mismatches, eviction_related_mismatches = self.compare_with_eviction_context(
            sim_entries, server_entries, eviction_by_conversation)
        
        # Analyze cache state differences
        sim_higher_mismatches, server_higher_mismatches = self.analyze_cache_state_differences(
            sim_entries, server_entries, eviction_by_conversation)
        
        # Final summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        total_comparisons = len(matches) + len(mismatches)
        match_rate = len(matches) / total_comparisons if total_comparisons > 0 else 0
        
        print(f"Total comparisons: {total_comparisons}")
        print(f"Match rate: {match_rate:.2%}")
        print(f"Total eviction events: {len(eviction_events)}")
        print(f"Mismatches with eviction context: {len(eviction_related_mismatches)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python validator.py <replay_sim.log> <server.log>")
        sys.exit(1)
    
    sim_log = sys.argv[1]
    server_log = sys.argv[2]
    
    comparator = HitTokenComparator(sim_log, server_log)
    comparator.generate_enhanced_report()

if __name__ == "__main__":
    main() 