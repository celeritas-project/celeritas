#!/usr/bin/env python3
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Convert an ORANGE CSG JSON representation to a GraphViz input.
"""
from itertools import count, repeat
import json

class DoxygenGenerator:
    def __init__(self, f):
        self.f = f
        self.write = f.write
        self.write("""\
strict digraph  {
rankdir=TB;
""")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.write("}\n")

    def write_node(self, i, value):
        self.write(f"{i:02d} [label=\"{value}\"];\n")

    def write_edge(self, i, e):
        self.write(f"{i:02d} -> {e:02d};\n")

def process(gen, tree, labels):
    for (i, node, labs) in zip(count(), tree, labels or repeat(None)):
        if isinstance(node, str):
            # True (or false??)
            gen.write_node(i, node)
            continue
        label = "\\n".join(labs) if labs else ""

        (nodetype, value) = node
        if nodetype == "S":
            # Surface (literal)
            gen.write_node(i, label or f"S{value}")
            continue

        if label:
            label += "\\n"
        label += nodetype
        gen.write_node(i, label)

        if isinstance(value, list):
            # Joined
            for v in value:
                gen.write_edge(i, v)
        else:
            # Aliased/negated
            gen.write_edge(i, value)

def run(infile, outfile, universe):
    tree = json.load(infile)
    metadata = None
    if universe is not None:
        # Load from a 'proto' debug file
        csg_unit = tree[universe]
        tree = csg_unit["tree"]
        metadata = csg_unit["metadata"]

    with DoxygenGenerator(outfile) as gen:
        process(gen, tree, metadata)

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Input filename (- for stdin)")
    parser.add_argument('-u', '--universe', type=int, default=None,
                        help="Universe ID if a 'proto' debug file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output filename (empty for stdout)")
    args = parser.parse_args()

    if args.input == "-":
        infile = sys.stdin
    else:
        infile = open(args.input)

    if not args.output:
        outfile = sys.stdout
    else:
        outfile = open(args.output, 'w')

    run(infile, outfile, args.universe)

if __name__ == "__main__":
    main()
