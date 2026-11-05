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

CELERITAS_GROUP_ID = "2380941"


def get_token():
    key = "ZOTERO_TOKEN"
    try:
        return os.environ[key]
    except KeyError:
        pass

    token_path = Path.home() / ".config" / "zotero-token"
    if token_path.exists():
        print(f"Reading token from {token_path}")
        return token_path.read_text().strip()

    print(
        "fatal: Zotero API token not specified "
        f"in environment variable {key} or path {token_path}"
    )
    sys.exit(1)


def run(group_id, outdir):
    zot = zotero.Zotero(group_id, "group", api_key=get_token())

    def sorted_collection(name):
        return zp.sorted_data_by_date(zp.collection_items_top(zot, name))

  #  with open(outdir / "publications.md", "w") as f:
  #      zp.print_bibliography(sorted_collection, f)

    with open(outdir / "references.md", "w") as f:
        zp.print_references(sorted_collection, f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-g", "--group", type=int, help="Zotero group ID", default=CELERITAS_GROUP_ID
    )
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    args = parser.parse_args()
    outdir = Path(args.output or ".")
    run(args.group, outdir)


if __name__ == "__main__":
    main()
