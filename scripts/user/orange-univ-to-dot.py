#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Convert universe structure metadata within a JSON file to a GraphViz .dot file.

The structure metadata is only written if the ``ORANGE_UNIV_STRUCTURE``
environment variable is set when running when generating the JSON.

Behavior:

1. ``-o/--output``: write the universes embedded in the root universe
   given by ``-u/--universe`` (default: the global universe) to the given
   file,
2. No ``-o/--output``: print universe-wise diagnostics only.

.. example::

    # Print diagnostics only
    ./orange-univ-to-dot.py out.json

    # Write all universes
    ./orange-univ-to-dot.py out.json -o universes.dot

    # Write the universes embedded in universe 3
    ./orange-univ-to-dot.py out.json -u 3 -o universes.dot
"""

import json
import sys
from textwrap import dedent

# Fill colors by universe type
_FILL_COLORS = {
    "simple": "lightgray",
    "rect_array": "lightsteelblue",
}

# Edge colors by daughter transform type
_EDGE_COLORS = {
    "no_transformation": "black",
    "translation": "blue",
    "transformation": "darkred",
}


def find_embedded(structure, root_id):
    """Find all universes embedded in the universe given by root_id."""
    result = set()
    stack = [root_id]
    while stack:
        uid = stack.pop()
        if uid not in result:
            result.add(uid)
            stack.extend(d[1] for d in structure[uid]["daughters"])
    return result


def dump_dot(univ_data, uids, out):
    structure = univ_data["structure"]

    out.write(
        dedent("""\
        digraph "universes" {
          rankdir=TB
          node [shape=box style=filled]
        """)
    )

    for uid in sorted(uids):
        univ_type = univ_data["type"][uid]
        label = (
            f"uid: {uid}\\l"
            f"label: {univ_data['label'][uid]}\\l"
            f"type: {univ_type}\\l"
            f"vols: {univ_data['num_volumes'][uid]}\\l"
            f"surfs: {univ_data['num_surfaces'][uid]}\\l"
        )
        fill = _FILL_COLORS.get(univ_type, "white")
        out.write(f'  n{uid} [label="{label}" fillcolor="{fill}"]\n')

        # Count daughters by universe and transform type to reduce edges
        counts = {}
        for _, daughter, transform in structure[uid]["daughters"]:
            key = (daughter, transform)
            counts[key] = counts.get(key, 0) + 1

        for (daughter, transform), num_vols in sorted(counts.items()):
            label = f"{transform}\\lvols: {num_vols}\\l"
            color = _EDGE_COLORS.get(transform, "black")
            out.write(
                f'    n{uid} -> n{daughter} [label="{label}"'
                f' color="{color}" fontcolor="{color}"]\n'
            )
    out.write("}\n")


def write_diagnostic(outfile, univ_data):
    univ_type = univ_data["type"]
    label = univ_data["label"]
    num_vols = univ_data["num_volumes"]
    num_surfs = univ_data["num_surfaces"]
    structure = univ_data["structure"]

    template = "{:>8} {:>10} {:>8} {:>8} {:>11} {}\n"
    out = template.format("uid", "type", "# vols", "# surfs", "# daughters", "label")

    for i in range(len(univ_type)):
        num_dtrs = len(structure[i]["daughters"])
        out += template.format(
            i, univ_type[i], num_vols[i], num_surfs[i], num_dtrs, label[i]
        )

    outfile.write(out)


def run(args):
    with open(args.input) as f:
        data = json.load(f)

    if "internal" in data:
        print("Assuming app output was given rather than ORANGE output")
        orange = data["internal"]["orange"]
    else:
        orange = data["orange_stats"]

    univ_data = orange["univ_metadata"]
    if "structure" not in univ_data:
        sys.exit("Universe structure is missing: run with ORANGE_UNIV_STRUCTURE=1")

    if args.output is None:
        write_diagnostic(sys.stdout, univ_data)
        return

    uids = find_embedded(univ_data["structure"], args.universe)

    with open(args.output, "w") as out:
        dump_dot(univ_data, uids, out)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="Input JSON filename",
    )
    parser.add_argument(
        "-u",
        "--universe",
        type=int,
        default=0,
        help="Root universe ID (default: global universe)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output DOT filename (default: print diagnostics only)",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
