#!/usr/bin/env python3
"""Expand the saved Holos Snake activation into standard ONNX operations.

Run ONLY on a private working copy of the existing model bundle. The original
decoder and DLL remain next to the portable graph with explicit backup names.
This enables the saved Windows CPU model to run on Linux without a Windows DLL.
The formula x + b * sin(a * x)^2 matches the original dordiai Snake kernel;
weights and voice profiles are unchanged. Floating point rounding may differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    import onnx
    from onnx import helper
    folder = args.models / "Holos"
    active = folder / "decoder_window.onnx"
    original = folder / "decoder_window.original.onnx"
    graph = onnx.load(active)
    count = sum(n.domain == "dordiai" and n.op_type == "Snake" for n in graph.graph.node)
    if not count:
        if original.is_file():
            print("Portable graph already present")
            return
        raise ValueError("Input has no supported Snake nodes; no conversion performed")
    if original.exists():
        raise FileExistsError("Original backup already exists; refusing replacement")
    if any(n.domain == "dordiai" and n.op_type != "Snake" for n in graph.graph.node):
        raise ValueError("Unsupported custom dordiai operation")
    before = digest(active)
    rewritten = []
    number = 0
    for node in graph.graph.node:
        if node.domain == "dordiai" and node.op_type == "Snake":
            if len(node.input) != 3 or len(node.output) != 1:
                raise ValueError("Unexpected Snake contract")
            x, a, b = node.input
            stem = "portable_snake_" + str(number)
            rewritten.extend([
                helper.make_node("Mul", [x,a], [stem+"_a"]),
                helper.make_node("Sin", [stem+"_a"], [stem+"_s"]),
                helper.make_node("Mul", [stem+"_s",stem+"_s"], [stem+"_ss"]),
                helper.make_node("Mul", [b,stem+"_ss"], [stem+"_b"]),
                helper.make_node("Add", [x,stem+"_b"], list(node.output)),
            ])
            number += 1
        else:
            rewritten.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    temporary = folder / "decoder_window.portable.onnx"
    onnx.save(graph, temporary)
    active.rename(original)
    temporary.rename(active)
    dll = folder / "dordiai_kernels.dll"
    if dll.is_file():
        unused = folder / "dordiai_kernels.dll.unused"
        if unused.exists():
            raise FileExistsError("DLL backup already exists")
        dll.rename(unused)
    report = {"original_sha256": before, "portable_sha256": digest(active),
              "expanded_snake_nodes": count, "weights_changed": False,
              "formula": "x + b * sin(a * x)^2", "floating_point_bit_parity_claimed": False}
    (folder / "linux_portability.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
