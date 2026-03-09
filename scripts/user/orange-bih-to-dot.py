#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Convert BIH structure metadata within a JSON file to GraphViz .dot files.

Behavior:

1. ``-u/--universe``: write ``bih_<uid>.dot`` for selected universe IDs,
2. ``-a/--all``: write ``bih_<uid>.dot`` for all universes,
3. Neither ``-u/--universe`` nor ``-a/--all``: print univers-wise diagnostics
   only.

In cases 1 and 2 the output directory can be specified with ``-o/--output``.

.. example::

    # Write selected universes
    ./orange-bih-to-dot.py out.json -u 0 3

    # Write all universes
    ./orange-bih-to-dot.py out.json --all

    # Print diagnostics only
    ./orange-bih-to-dot.py out.json
"""

import json
import sys
from pathlib import Path


def make_dot(universe_id, tree):
    lines = []
    lines.append('digraph "bih_{}" {{'.format(universe_id))
    lines.append("  rankdir=TB")
    lines.append("  node [shape=box]")

    for i, node in enumerate(tree):
        if node[0] == "i":
            axis = node[1]
            left_plane = float(node[3][0])
            right_plane = float(node[3][1])
            label = "{}=({:.2f}, {:.2f})".format(axis, left_plane, right_plane)
            lines.append('  n{} [label="{}"]'.format(i, label))
        else:
            vols = ", ".join("{}".format(v) for v in node[1])
            lines.append('  n{} [label="vols=[{}]"]'.format(i, vols))

    for i, node in enumerate(tree):
        if node[0] == "i":
            left_child = node[2][0]
            right_child = node[2][1]
            lines.append('  n{} -> n{} [label="L"]'.format(i, left_child))
            lines.append('  n{} -> n{} [label="R"]'.format(i, right_child))

    lines.append("}")
    return "\n".join(lines)


def write_diagnostic(outfile, bih_data):
    depth = bih_data["depth"]
    nvol = bih_data["num_finite_bboxes"]
    ninf = bih_data["num_infinite_bboxes"]

    template = "{:>8} {:>8} {:>8} {:>8}\n"
    out = template.format("uid", "depth", "num_vols", "num_inf")

    for i in range(len(depth)):
        out += template.format(i, depth[i], nvol[i], ninf[i])

    outfile.write(out)


def run(args):
    data = json.load(open(args.input))
    bih_data = data["internal"]["orange"]["bih_metadata"]

    if args.universe:
        uids = args.universe
    elif args.all:
        uids = list(range(len(bih_data["structure"])))
    else:
        write_diagnostic(sys.stdout, bih_data)
        return

    for uid in uids:
        dot = make_dot(uid, bih_data["structure"][uid]["tree"])
        with open(Path(args.output) / "bih_{}.dot".format(uid), "w") as out:
            out.write(dot)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        default=None,
        help="Input JSON filename (default: stdin)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-u",
        "--universe",
        nargs="+",
        type=int,
        default=None,
        help="Universe ID(s) to export as DOT",
    )
    mode.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Export DOT for all universes",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path.cwd(),
        help="Output directory (default: current directory)",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
