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

# Fill colors by internal-node plane relationship and leaf type
_FILL_COLORS = {
    "gap": "lightgreen",
    "overlap": "#ffaaaa",
    "exact": "white",
    "slight": "#dbc6a7",
    "leaf": "lightgray",
}

_SLIGHT_REL = 1e-4

# Outline colors by partition axis (x/y/z)
_AXIS_COLORS = {
    "x": "red",
    "y": "green",
    "z": "blue",
}


def dump_dot(universe_id, tree, out):
    out.write(f"""\
strict digraph "bih_{universe_id}" {{
  rankdir=LR
  node [shape=box style=filled]
""")

    for i, node in enumerate(tree):
        nodetype = node[0]
        if node[0] == "i":
            (axis, children, planes) = node[1:]

            (left, right) = planes
            delta = abs(right - left)
            if delta == 0.0:
                plane_label = left
                fill = _FILL_COLORS["exact"]
            else:
                # + indicates gap, - indicates overlap
                if left < right:
                    sign = "+"
                    fill = _FILL_COLORS["gap"]
                else:
                    sign = "-"
                    if delta / max(abs(left), abs(right)) < _SLIGHT_REL:
                        fill = _FILL_COLORS["slight"]
                    else:
                        fill = _FILL_COLORS["overlap"]
                plane_label = f"({left:.2g} {sign} {delta:.2g})"
            axis_color = _AXIS_COLORS.get(axis, "black")
            out.write(
                f'  n{i} [label="{axis}: {plane_label}"'
                f' fillcolor="{fill}" color="{axis_color}" penwidth=3]\n'
            )

            for sign, c in zip(["-", "+"], children):
                out.write(f'    n{i} -> n{c} [label="{sign}{axis}"]\n')
        else:
            assert nodetype == "l"
            vols = ", ".join("{}".format(v) for v in node[1])
            out.write(
                f'  n{i} [label="vols=[{vols}]" fillcolor="{_FILL_COLORS["leaf"]}"]\n'
            )

    out.write("}\n")


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
    with open(args.input) as f:
        data = json.load(f)

    if "internal" in data:
        print("Assuming app output was given rather than ORANGE output")
        orange = data["internal"]["orange"]
    else:
        orange = data

    bih_data = orange["bih_metadata"]

    if args.universe:
        uids = args.universe
    elif args.all:
        uids = list(range(len(bih_data["structure"])))
    else:
        write_diagnostic(sys.stdout, bih_data)
        return

    for uid in uids:
        with open(Path(args.output) / "bih_{}.dot".format(uid), "w") as out:
            dump_dot(uid, bih_data["structure"][uid]["tree"], out)


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
