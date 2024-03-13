#!/usr/bin/env python3
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Convert an ORANGE CSG JSON representation to a GraphViz input.
"""
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

    def write_literal(self, i, value):
        self.write(f"{i:02d} [label=\"{value}\"];\n")

    def write_node(self, i, value, edges):
        self.write_literal(i, f"{i}: {value}")
        for e in edges:
            self.write(f"{i:02d} -> {e:02d};\n")

def process(gen, tree):
    for (i, node) in enumerate(tree):
        if isinstance(node, str):
            # True (or false??)
            gen.write_literal(i, node)
            continue
        (nodetype, value) = node
        if nodetype == "S":
            # Surface
            gen.write_literal(i, f"S{value}")
        elif nodetype in ("~", "="):
            # Negated
            gen.write_node(i, nodetype, [value])
        else:
            # Joined
            gen.write_node(i, nodetype, value)

def run(infile, outfile):
    tree = json.load(infile)
    with DoxygenGenerator(outfile) as gen:
        process(gen, tree)

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Input filename (- for stdin)")
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
        outfile = open(args.output)

    run(infile, outfile)

if __name__ == "__main__":
    main()
