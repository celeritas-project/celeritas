#!/usr/bin/env python3
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Generate a GraphViz DAG of GDML logical volume relationships.

The resulting output file can be converted to a PDF file with, e.g.::

    dot -Tpdf:quartz demo.gdml.dot -o demo.pdf

"""

import json
import re
import xml.etree.ElementTree as ET
import networkx as nx

from collections import defaultdict
from pathlib import Path

class PointerReplacer:
    sub = re.compile(r'0x[0-9a-f]{4,}').sub

    def __init__(self):
        self.addrs = {}

    def repl(self, match):
        val = self.addrs.setdefault(match.group(0), len(self.addrs))
        return f"@{val:d}"

    def __call__(self, s):
        return self.sub(self.repl, s)

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = defaultdict(int)
        self.replace_pointers = PointerReplacer()

    def add_volume(self, el):
        edges = self.edges

        pname = self.replace_pointers(el.attrib["name"])
        self.nodes.append(pname)
        for vrel in el.iter("volumeref"):
            dname = self.replace_pointers(vrel.attrib["ref"])
            edges[(pname, dname)] += 1

    def add_world(self, vrel):
        pname = self.replace_pointers(vrel.attrib["ref"])
        self.nodes.append(pname)

    @property
    def weighted_edges(self):
        for ((u, v), weight) in self.edges.items():
            yield (u, v, weight)

    @property
    def labeled_edges(self):
        for ((u, v), weight) in self.edges.items():
            yield (u, v, ("" if weight == 1 else f"×{weight}"))

    @property
    def pointer_addresses(self):
        return self.replace_pointers.addrs

def read_graph(filename):
    tree = ET.parse(filename)
    structure = next(tree.iter("structure"))

    g = Graph()
    for el in structure:
        if el.tag in ('volume', 'assembly'):
            g.add_volume(el)
        else:
            raise ValueError(f"Unrecognized structure tag: {el!r}")
    g.add_world(tree.findall("./setup/world")[0])

    return g

def write_graph(g, filename):
    graph = nx.DiGraph()
    graph.add_nodes_from(reversed(g.nodes))
    graph.add_weighted_edges_from(g.labeled_edges, weight='label')
    graph.graph['graph']={'rankdir':'LR'}
    nx.nx_pydot.write_dot(graph, filename)

    with open(filename, 'a') as f:
        f.write("// Pointer mapping:\n")
        addrs = g.pointer_addresses.items()
        for (idx, addr) in sorted((v, k) for (k, v) in addrs):
            f.write(f"// {idx:04d}: {addr}\n")

def main(*args):
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__, prog="gdml-to-dot")
    parser.add_argument('-o', '--output')
    parser.add_argument('input')
    ns = parser.parse_args(*args)
    input = Path(ns.input)
    g = read_graph(input)
    write_graph(g, ns.output or (input.stem + ".dot"))

if __name__ == "__main__":
    main()
