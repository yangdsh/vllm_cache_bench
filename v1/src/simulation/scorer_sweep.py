#!/usr/bin/env python3
"""
Scorer hyperparameter sweep utility.

Runs a sweep over RuleScorerConfig parameters against a subset of a
server log and prints top configurations ranked by token_hit_ratio and hit_ratio.
"""

import argparse
import logging
import sys
import os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.common import LogParser
from simulation.cache_simulator import CacheSimulatorReplay, SimulatorConfig
from simulation.rule_scorer import RuleScorerConfig


def run_sweep(
    input_file: str,
    max_events: int,
    base_score_grid: List[float],
    length_norm_grid: List[float],
    reuse_ref_grid: List[float],
    length_weight_grid: List[float],
    cache_size_gb: float,
    bytes_per_token: int,
) -> None:
    logging.disable(logging.WARNING)

    parser = LogParser(input_file)
    events = parser.parse_log_file(mode='server')
    if max_events:
        events = events[:max_events]

    results = []
    for bs in base_score_grid:
        for ln in length_norm_grid:
            for rr in reuse_ref_grid:
                for lw in length_weight_grid:
                    rw = 1.0 - lw
                    scoring_cfg = RuleScorerConfig(
                        length_norm=ln,
                        reuse_reference=rr,
                        base_score=bs,
                        length_weight=lw,
                        reuse_weight=rw,
                    )
                    sim_cfg = SimulatorConfig(
                        cache_size_gb=cache_size_gb,
                        eviction_policy='conversation_aware',
                        scoring_config=scoring_cfg,
                        bytes_per_token=bytes_per_token,
                        block_size=256,
                    )
                    replay = CacheSimulatorReplay(sim_cfg)
                    replay.replay_log_events(events)
                    stats = replay.simulator.get_detailed_statistics()
                    b = stats['basic_stats']
                    results.append({
                        'base_score': bs,
                        'length_norm': ln,
                        'reuse_reference': rr,
                        'length_weight': lw,
                        'reuse_weight': rw,
                        'hit_ratio': b['hit_ratio'],
                        'token_hit_ratio': b['token_hit_ratio'],
                        'cache_hits': b['cache_hits'],
                        'cache_misses': b['cache_misses'],
                    })

    # Sort by token_hit_ratio then hit_ratio
    results.sort(key=lambda x: (x['token_hit_ratio'], x['hit_ratio']), reverse=True)

    print('Top configurations:')
    for r in results[:10]:
        print(
            f"base_score={r['base_score']:.2f}, "
            f"len_norm={r['length_norm']}, "
            f"reuse_ref={r['reuse_reference']}, "
            f"len_w={r['length_weight']:.2f}, reuse_w={r['reuse_weight']:.2f} -> "
            f"token_hit={r['token_hit_ratio']:.4f}, hit={r['hit_ratio']:.4f}, "
            f"hits={r['cache_hits']}, misses={r['cache_misses']}"
        )


def main():
    ap = argparse.ArgumentParser(description='Run scorer hyperparameter sweep')
    ap.add_argument('--input-file', required=True, help='Path to server log file')
    ap.add_argument('--max-events', type=int, default=100000)
    ap.add_argument('--cache-size', type=float, default=64.0)
    ap.add_argument('--base-score', type=str, default='0,0.5,1')
    ap.add_argument('--length-norm', type=str, default='100,200,300')
    ap.add_argument('--reuse-ref', type=str, default='500, 1000, 1500, 2000')
    ap.add_argument('--length-weight', type=str, default='0,0.25,0.5,0.75,1')
    ap.add_argument('--bytes-per-token', type=int, default=144 * 1024)

    args = ap.parse_args()

    def parse_floats(csv: str) -> List[float]:
        return [float(x.strip()) for x in csv.split(',') if x.strip()]

    run_sweep(
        input_file=args.input_file,
        max_events=args.max_events,
        base_score_grid=parse_floats(args.base_score),
        length_norm_grid=parse_floats(args.length_norm),
        reuse_ref_grid=parse_floats(args.reuse_ref),
        length_weight_grid=parse_floats(args.length_weight),
        cache_size_gb=args.cache_size,
        bytes_per_token=args.bytes_per_token,
    )


if __name__ == '__main__':
    main()


