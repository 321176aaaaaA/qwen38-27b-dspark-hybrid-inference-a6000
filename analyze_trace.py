#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate torch profiler trace by kernel and CPU op."""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
print(f"loading {path} ...", file=sys.stderr)
with open(path, "r") as f:
    data = json.load(f)

events = data.get("traceEvents", [])
print(f"events={len(events)}", file=sys.stderr)

kernels = defaultdict(lambda: [0, 0.0])  # name -> [count, total us]
cpu_ops = defaultdict(lambda: [0, 0.0])

for e in events:
    if e.get("ph") != "X":
        continue
    name = e.get("name", "")
    dur = e.get("dur", 0) or 0
    cat = e.get("cat", "")
    if cat == "kernel":
        kernels[name][0] += 1
        kernels[name][1] += dur
    elif cat == "cpu_op":
        cpu_ops[name][0] += 1
        cpu_ops[name][1] += dur

print("\n=== TOP 60 CUDA KERNELS BY GPU TIME (us) ===")
for name, (cnt, total) in sorted(kernels.items(), key=lambda kv: kv[1][1], reverse=True)[:60]:
    print(f"{total/1e6:10.3f} s  {total:12.0f} us  cnt={cnt:6d}  {name[:140]}")

print("\n=== TOP 60 CPU OPS BY CPU TIME (us) ===")
for name, (cnt, total) in sorted(cpu_ops.items(), key=lambda kv: kv[1][1], reverse=True)[:60]:
    print(f"{total/1e6:10.3f} s  {total:12.0f} us  cnt={cnt:6d}  {name[:140]}")
