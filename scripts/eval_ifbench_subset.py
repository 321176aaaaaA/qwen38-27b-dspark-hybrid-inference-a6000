#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate IFBench subset responses using IFBench evaluation_lib."""
import argparse
import json
import sys

sys.path.insert(0, "/opt/IFBench")
import evaluation_lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/opt/IFBench/data/IFBench_partial_120.jsonl")
    ap.add_argument("--responses", default="/opt/dspark_dev_vllm/quality_ifbench_subset.jsonl")
    args = ap.parse_args()

    inputs = evaluation_lib.read_prompt_list(args.input)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(args.responses)

    for name, func in [
        ("strict", evaluation_lib.test_instruction_following_strict),
        ("loose", evaluation_lib.test_instruction_following_loose),
    ]:
        outputs = [func(inp, prompt_to_response) for inp in inputs]
        prompt_acc = sum(o.follow_all_instructions for o in outputs) / len(outputs)
        instr_total = sum(len(o.follow_instruction_list) for o in outputs)
        instr_correct = sum(sum(o.follow_instruction_list) for o in outputs)
        print(f"{name}: prompt_level={prompt_acc:.4f} instruction_level={instr_correct / instr_total:.4f}")


if __name__ == "__main__":
    main()
