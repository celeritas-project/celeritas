#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Render the Celeritas volume hierarchy JSON as a GraphViz digraph.

The input is the JSON produced by ``VolumeParamsIO.json`` (or the stream
``operator<<`` of ``VolumeParams``).  Volumes (logical) are rendered as
rounded-rectangle nodes; volume instances (physical placements / edges) are
shown as plain box nodes between parent and child volumes.

.. example::

    Generate a PDF of the hierarchy stored in ``volumes.json``::

        ./volumes-to-dot.py volumes.json | dot -Tpdf -o volumes.pdf

    Read from another tool's output via stdin::

        celer-geo --dump-volumes | ./volumes-to-dot.py - | dot -Tsvg -o out.svg
"""

import argparse
import json
import sys


def escape(s: str) -> str:
    """Escape a string for use inside a GraphViz label."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render(data: dict, out, print_ids: bool = False):
    volumes = data["volumes"]  # list of label strings
    vis = data["volume_instances"]  # list of label strings (may be empty)
    world = data["world"]  # int or null
    children = data["children"]  # children[v] = [vi, ...]
    i2v = data["instance_to_volume"]  # i2v[vi] = v or null

    w = out.write

    w("digraph volumes {\n")
    w("  rankdir=TB\n")
    w(
        '  node [fontname="Helvetica", shape=box, style="rounded,filled",'
        " fillcolor=lightblue]\n"
    )
    w("\n")

    # Volume nodes (logical)
    for v_idx, label in enumerate(volumes):
        lbl = escape(label) if label else ""
        if print_ids:
            id_str = f"V{{{v_idx}}}"
            lbl = f"{lbl}\\n{id_str}" if lbl else id_str
        elif not lbl:
            lbl = f"v{v_idx}"
        w(f'  v{v_idx} [label="{lbl}"]\n')

    w("\n")
    # Edges: one per volume instance that appears in a children list
    for v_idx, child_vis in enumerate(children):
        for vi_idx in child_vis:
            child_v = i2v[vi_idx]
            if child_v is None:
                continue
            vi_label = vis[vi_idx] if vi_idx < len(vis) else ""
            if print_ids:
                id_str = f"VI{{{vi_idx}}}"
                vi_label = f"{escape(vi_label)}\\n{id_str}" if vi_label else id_str
            elif vi_label:
                vi_label = escape(vi_label)
            edge_lbl = f' [label="{vi_label}"]' if vi_label else ""
            w(f"  v{v_idx} -> v{child_v}{edge_lbl}\n")

    w("}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Input filename (- for stdin)")
    parser.add_argument(
        "-o", "--output", default=None, help="Output filename (default: stdout)"
    )
    parser.add_argument(
        "--ids",
        action="store_true",
        default=False,
        help="Append V{NNN}/VI{MMM} IDs to volume/instance labels",
    )
    args = parser.parse_args()

    if args.input == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.input) as f:
            data = json.load(f)

    if args.output:
        with open(args.output, "w") as f:
            render(data, f, print_ids=args.ids)
    else:
        render(data, sys.stdout, print_ids=args.ids)


if __name__ == "__main__":
    main()
