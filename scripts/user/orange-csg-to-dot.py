#!/usr/bin/env python3
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Convert an ORANGE CSG JSON representation to a GraphViz input.
"""
from itertools import count, repeat
import json

class DotGenerator:
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

    def write_volume(self, i, value):
        self.write(f"volume{i:02d} [label=\"{value}\", shape=box];\n")
        self.write(f"volume{i:02d} -> {i:02d} [color=gray, dir=both];\n")

    def write_edge(self, i, e):
        self.write(f"{i:02d} -> {e:02d};\n")



class MermaidGenerator:
    def __init__(self, f):
        self.f = f
        self.write = f.write
        self.write("flowchart TB\n")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.write("")

    def write_node(self, i, value):
        self.write(f"  n{i:02d}[\"{value}\"]\n")

    def write_volume(self, i, value):
        self.write(f"  v{i:02d}([\"{value}\"])\n")
        self.write(f"  v{i:02d} <--> n{i:02d}\n")

    def write_edge(self, i, e):
        self.write(f"  n{i:02d} --> n{e:02d}\n")

def write_tree(gen, csg_unit):
    tree = csg_unit["tree"]
    labels = csg_unit["metadata"] or repeat(None)

    for (i, node, labs) in zip(count(), tree, labels):
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

def write_volumes(gen, volumes):
    for v in volumes:
        gen.write_volume(v["csg_node"], v["label"])

def run(infile, outfile, gencls, universe):
    tree = json.load(infile)
    if universe is not None:
        # Load from a .csg.json debug file
        csg_unit = tree[universe]
    else:
        csg_unit = {
            "tree": tree,
            "metadata": None,
            "label": "CSG tree",
        }

    with gencls(outfile) as gen:
        write_tree(gen, csg_unit)
        if (vols := csg_unit.get("volumes")):
            write_volumes(gen, vols)

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Input filename (- for stdin)")
    parser.add_argument('-T', '--type', default=None,
                        help="Output type: 'dot' or 'mermaid'")
    parser.add_argument('-u', '--universe', type=int, default=None,
                        help="Universe ID if a 'csg.json' debug file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output filename (empty for stdout)")
    args = parser.parse_args()

    if args.input == "-":
        infile = sys.stdin
    else:
        infile = open(args.input)

    if not args.type:
        if args.output and args.output.endswith(".dot"):
            gencls = DotGenerator
        else:
            gencls = MermaidGenerator
    else:
        gencls_dict = {
            "dot": DotGenerator,
            "mermaid": MermaidGenerator,
        }
        try:
            gencls = gencls_dict[args.type]
        except KeyError:
            valid = ",".join(gencls_dict)
            print(f"invalid type {args.type}: valid types are {valid}",
                  file=sys.stderr)


    if not args.output:
        outfile = sys.stdout
    else:
        outfile = open(args.output, 'w')

    run(infile, outfile, gencls, args.universe)

if __name__ == "__main__":
    main()
