#!/usr/bin/env python
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Download publications from the Zotero database.
"""

import os
import sys
from pathlib import Path
import zoteropubs as zp
from pyzotero import zotero


def run(group_id, outdir):
    try:
        api_key = os.environ["ZOTERO_TOKEN"]
    except KeyError as e:
        print(f"fatal: Zotero API token not specified in environment variable {e}")
        sys.exit(1)

    zot = zotero.Zotero(group_id, "group", api_key)

    def sorted_collection(name):
        return zp.sorted_data_by_date(zp.collection_items_top(zot, name))

    with open(outdir / "publications.md", "w") as f:
        zp.print_bibliography(sorted_collection, f)

    with open(outdir / "references.md", "w") as f:
        zp.print_references(sorted_collection, f)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--group", type=int, help="Zotero group ID")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    args = parser.parse_args()
    outdir = Path(args.output or ".")
    run(args.group, outdir)


if __name__ == "__main__":
    main()
