"""
Benchmark script — Measure recommendation inference latency.

Runs N recommendation requests and reports:
- Cold start time (model loading)
- Warm inference: median, mean, p95, p99 latency
- Throughput (requests/second)

Usage:
    python scripts/benchmark.py
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backend.recommender import Recommender
from src.backend.config import Config

N_REQUESTS = 100
N_RECOMMENDATIONS = 10


def benchmark():
    """Run latency benchmark."""
    print("=" * 60)
    print("INFERENCE LATENCY BENCHMARK")
    print("=" * 60)

    # --- Cold start ---
    print("\n1. Cold Start (model loading)...")
    cold_start = time.time()
    rec = Recommender(
        model_path=Config.MODEL_PATH,
        movies_path=Config.MOVIES_PATH,
        metrics_path=Config.METRICS_PATH,
    )
    cold_elapsed = (time.time() - cold_start) * 1000
    print(f"   Cold start time: {cold_elapsed:.0f}ms")

    # --- Get sample user IDs ---
    user_ids = rec.get_all_user_ids()
    sample_users = user_ids[:min(20, len(user_ids))]

    # --- Warm-up ---
    print(f"\n2. Warming up (3 requests)...")
    for uid in sample_users[:3]:
        rec.get_recommendations(uid, n=N_RECOMMENDATIONS)

    # --- Benchmark ---
    print(f"\n3. Running {N_REQUESTS} recommendation requests (top-{N_RECOMMENDATIONS})...")
    latencies = []

    for i in range(N_REQUESTS):
        uid = sample_users[i % len(sample_users)]
        _, latency_ms = rec.get_recommendations(uid, n=N_RECOMMENDATIONS)
        latencies.append(latency_ms)

        if (i + 1) % 25 == 0:
            print(f"   Completed {i + 1}/{N_REQUESTS}...")

    # --- Results ---
    latencies.sort()
    median = statistics.median(latencies)
    mean = statistics.mean(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    min_lat = latencies[0]
    max_lat = latencies[-1]
    throughput = 1000 / mean  # requests per second

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':>12}")
    print("-" * 39)
    print(f"{'Requests':<25} {N_REQUESTS:>12}")
    print(f"{'Top-N':<25} {N_RECOMMENDATIONS:>12}")
    print(f"{'Cold start':<25} {cold_elapsed:>10.0f}ms")
    print(f"{'Min latency':<25} {min_lat:>10.1f}ms")
    print(f"{'Median latency':<25} {median:>10.1f}ms")
    print(f"{'Mean latency':<25} {mean:>10.1f}ms")
    print(f"{'P95 latency':<25} {p95:>10.1f}ms")
    print(f"{'P99 latency':<25} {p99:>10.1f}ms")
    print(f"{'Max latency':<25} {max_lat:>10.1f}ms")
    print(f"{'Throughput':<25} {throughput:>9.1f}req/s")

    target_met = median < 100
    print(f"\n{'✓' if target_met else '⚠'} Target: median < 100ms — "
          f"{'MET' if target_met else 'NOT MET'} (actual: {median:.1f}ms)")

    return {
        "cold_start_ms": round(cold_elapsed, 0),
        "median_ms": round(median, 1),
        "mean_ms": round(mean, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "throughput_rps": round(throughput, 1),
    }


if __name__ == "__main__":
    benchmark()
