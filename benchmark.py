"""
Latency Benchmark — p50 / p99
==============================
Usage:
    python benchmark.py                      # 200 requests to localhost:8000
    python benchmark.py --n 1000 --url http://localhost:8000
"""

import argparse
import time
import statistics
import json
import urllib.request
import urllib.error

SAMPLE_PAYLOAD = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46,  "V7": 0.24, "V8": 0.10,
    "V9": 0.36,  "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Time": 406.0, "Amount": 149.62,
}


def bench(url: str, n: int, warmup: int = 10) -> None:
    endpoint = f"{url.rstrip('/')}/predict"
    body = json.dumps(SAMPLE_PAYLOAD).encode()
    headers = {"Content-Type": "application/json"}

    latencies_ms: list[float] = []
    errors = 0

    print(f"Target : {endpoint}")
    print(f"Warmup : {warmup} requests  |  Benchmark : {n} requests")
    print()

    # Warmup
    for _ in range(warmup):
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            pass

    # Timed loop
    for i in range(n):
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            errors += 1
            if i < 5:
                print(f"  [error] {e}")

    if not latencies_ms:
        print("No successful requests — is the API running?")
        return

    latencies_ms.sort()
    p50  = latencies_ms[int(len(latencies_ms) * 0.50)]
    p90  = latencies_ms[int(len(latencies_ms) * 0.90)]
    p99  = latencies_ms[int(len(latencies_ms) * 0.99)]
    mean = statistics.mean(latencies_ms)
    mx   = latencies_ms[-1]

    print("=" * 40)
    print(f"  Requests   : {n}  (errors: {errors})")
    print(f"  Mean       : {mean:.1f} ms")
    print(f"  p50        : {p50:.1f} ms")
    print(f"  p90        : {p90:.1f} ms")
    print(f"  p99        : {p99:.1f} ms")
    print(f"  Max        : {mx:.1f} ms")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud API latency benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n",   type=int, default=200, help="Number of timed requests")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    bench(args.url, args.n, args.warmup)
