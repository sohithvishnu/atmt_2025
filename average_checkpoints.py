#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
average_checkpoints.py
--------------------------------
Averages multiple model checkpoints to reduce variance and improve translation stability.

Usage examples:
---------------
# Average the last 3 checkpoints in a directory
python average_checkpoints.py --checkpoint-dir checkpoints/ --num-last 3

# Or specify specific checkpoints manually
python average_checkpoints.py --inputs checkpoints/checkpoint_epoch_0.pt checkpoints/checkpoint_epoch_1.pt checkpoints/checkpoint_epoch_2.pt --output checkpoints/checkpoint_avg.pt
"""

import os
import re
import torch
import argparse
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser(description="Average the weights of multiple model checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to the directory containing checkpoints.")
    parser.add_argument("--inputs", type=str, nargs="+", default=None,
                        help="List of checkpoint files to average (overrides --num-last).")
    parser.add_argument("--num-last", type=int, default=3,
                        help="Number of last checkpoints to average (if --inputs not provided).")
    parser.add_argument("--output", type=str, default="checkpoint_avg.pt",
                        help="Output file to save the averaged checkpoint.")
    return parser.parse_args()


def sorted_checkpoints(checkpoint_dir):
    """Return a sorted list of checkpoint files by modification time."""
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
             if f.endswith(".pt") and "avg" not in f]
    ckpts = sorted(ckpts, key=os.path.getmtime)
    return ckpts


def average_checkpoints(ckpt_files):
    """Load and average multiple checkpoint state_dicts."""
    assert len(ckpt_files) > 0, "No checkpoints to average."

    print(f"📦 Averaging {len(ckpt_files)} checkpoints:")
    for f in ckpt_files:
        print(f"   - {f}")

    avg_state_dict = None
    for i, ckpt_path in enumerate(ckpt_files):
        state = torch.load(ckpt_path, map_location="cpu")

        if "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state

        if avg_state_dict is None:
            avg_state_dict = OrderedDict()
            for k, v in state_dict.items():
                avg_state_dict[k] = v.clone().float()
        else:
            for k in avg_state_dict.keys():
                avg_state_dict[k] += state_dict[k].float()

    # Average
    for k in avg_state_dict.keys():
        avg_state_dict[k] /= len(ckpt_files)

    print("✅ Averaging complete.")
    return avg_state_dict


def save_averaged_checkpoint(avg_state_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"model": avg_state_dict}, output_path)
    print(f"💾 Saved averaged checkpoint to: {output_path}")


def main():
    args = get_args()

    if args.inputs:
        ckpt_files = args.inputs
    else:
        ckpt_files = sorted_checkpoints(args.checkpoint_dir)
        ckpt_files = ckpt_files[-args.num_last:]

    if len(ckpt_files) == 0:
        raise ValueError("No checkpoints found to average.")

    avg_state_dict = average_checkpoints(ckpt_files)
    save_averaged_checkpoint(avg_state_dict, args.output)


if __name__ == "__main__":
    main()
