#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Convert BVH structure metadata within a JSON file to GraphViz .dot files.

Behavior:

1. ``-u/--universe``: write ``bvh_<uid>.dot`` for selected universe IDs,
2. ``-a/--all``: write ``bvh_<uid>.dot`` for all universes,
3. Neither ``-u/--universe`` nor ``-a/--all``: print univers-wise diagnostics
   only.

In cases 1 and 2 the output directory can be specified with ``-o/--output``.

.. example::

    # Write selected universes
    ./orange-bvh-to-dot.py out.json -u 0 3

    # Write all universes
    ./orange-bvh-to-dot.py out.json --all

    # Print diagnostics only
    ./orange-bvh-to-dot.py out.json
"""

import json
import sys
from pathlib import Path
from textwrap import dedent

# Fill colors by internal-node plane relationship and leaf type
_FILL_COLORS = {
    "disparate": "#90fc94",
    "slight": "#fcf290",
    "medium": "#fcd690",
    "large": "#fcb090",
    "complete": "#fc9090",
    "leaf": "lightgray",
}

_SLIGHT_REL = 1e-2
_MEDIUM_REL = 1e-1
_COMPLETE_REL = 1 - 1e-2

# Outline colors by partition axis (x/y/z)
_AXIS_COLORS = {
    "x": "darkred",
    "y": "darkgreen",
    "z": "blue",
}


def fill_node(overlap):
    """
    Determine what fill color to use based on overlap fraction.
    """
    if overlap == 0.0:
        return _FILL_COLORS["disparate"]
    elif overlap < _SLIGHT_REL:
        return _FILL_COLORS["slight"]
    elif overlap < _MEDIUM_REL:
        return _FILL_COLORS["medium"]
    elif overlap < _COMPLETE_REL:
        return _FILL_COLORS["large"]
    else:
        return _FILL_COLORS["complete"]


def calc_overlap(bbox_a, bbox_b):
    """Calculate the fraction of the smaller bbox's volume that overlaps with
    the larger bbox.
    """
    (a_lo, a_hi) = bbox_a
    (b_lo, b_hi) = bbox_b

    overlap_vol = 1.0
    for i in range(3):
        lo = max(a_lo[i], b_lo[i])
        hi = min(a_hi[i], b_hi[i])
        if hi <= lo:
            # Bounding boxes do not overlap
            return 0.0
        overlap_vol *= hi - lo

    def calc_volume(lo, hi):
        return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])

    vol_a = calc_volume(a_lo, a_hi)
    vol_b = calc_volume(b_lo, b_hi)

    return overlap_vol / min(vol_a, vol_b)


def dump_dot(universe_id, tree, out):

    def format_bbox(bbox):
        result = ""
        for ax, lo, hi in zip("xyz", bbox[0], bbox[1]):
            result += f"{ax}: [{lo:.3g}, {hi:.3g}]\\l"
        return result

    out.write(
        dedent(f"""\
        strict digraph "bvh_{universe_id}" {{
          rankdir=LR
          node [shape=box style=filled]
        """)
    )

    for i, node in enumerate(tree):
        nodetype = node[0]
        if nodetype == "i":
            (axis, children, bboxes) = node[1:]
            (lo_bbox, hi_bbox) = bboxes

            overlap = calc_overlap(lo_bbox, hi_bbox)

            fill = fill_node(overlap)

            axis_color = _AXIS_COLORS.get(axis, "black")

            label = (
                f"axis: {axis}\\l"
                f"lo:\\l{format_bbox(lo_bbox)}"
                f"hi:\\l{format_bbox(hi_bbox)}"
                f"overlap: {100 * overlap:.1f}%\\l"
            )

            out.write(
                f'  n{i} [label="{label}"'
                f' fillcolor="{fill}" color="{axis_color}" penwidth=3]\n'
            )

            for edge_label, c in zip(["lo", "hi"], children):
                out.write(f'    n{i} -> n{c} [label="{edge_label}"]\n')
        else:
            assert nodetype == "l"
            vols = ", ".join("{}".format(v) for v in node[1])
            out.write(
                f'  n{i} [label="vols=[{vols}]" fillcolor="{_FILL_COLORS["leaf"]}"]\n'
            )
    out.write("}\n")


def write_diagnostic(outfile, bvh_data):
    depth = bvh_data["depth"]
    nvol = bvh_data["num_finite_bboxes"]
    ninf = bvh_data["num_infinite_bboxes"]

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
        orange = data["orange_stats"]

    bvh_data = orange["bvh_metadata"]

    if args.universe:
        uids = args.universe
    elif args.all:
        uids = list(range(len(bvh_data["structure"])))
    else:
        write_diagnostic(sys.stdout, bvh_data)
        return

    for uid in uids:
        with open(Path(args.output) / "bvh_{}.dot".format(uid), "w") as out:
            dump_dot(uid, bvh_data["structure"][uid]["tree"], out)


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
