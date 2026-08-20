#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fair IFBench evaluation wrapper (strict/loose, prompt+instruction level)."""
import argparse
import sys

# drop script's own dir (dspark_dev_vllm site-packages copy) from import path
sys.path = [p for p in sys.path if "dspark_dev_vllm" not in p]
sys.path.insert(0, "/opt/IFBench")
import evaluation_lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    inputs = evaluation_lib.read_prompt_list(args.input)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(args.responses)

    missing = sum(1 for inp in inputs if inp.prompt not in prompt_to_response)
    print(f"[fair_eval] tag={args.tag} n={len(inputs)} missing_resp={missing}")
    for name, func in [
        ("strict", evaluation_lib.test_instruction_following_strict),
        ("loose", evaluation_lib.test_instruction_following_loose),
    ]:
        outputs = [func(inp, prompt_to_response) for inp in inputs]
        prompt_acc = sum(o.follow_all_instructions for o in outputs) / len(outputs)
        instr_total = sum(len(o.follow_instruction_list) for o in outputs)
        instr_correct = sum(sum(o.follow_instruction_list) for o in outputs)
        print(f"[fair_eval] tag={args.tag} {name}: prompt_level={prompt_acc:.4f} "
              f"instruction_level={instr_correct / instr_total:.4f}")


if __name__ == "__main__":
    main()
