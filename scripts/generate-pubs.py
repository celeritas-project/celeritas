#!/usr/bin/env python
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Download publications from the Zotero database.
"""

import json
import os
import sys

from itertools import chain
from pathlib import Path
from pyzotero import zotero
from dateutil.parser import parse as parse_date

try:
    API_KEY = os.environ["ZOTERO_TOKEN"]
except KeyError as e:
    print(f"fatal: Zotero API token not specified in environment variable {e}")
    sys.exit(1)


def log(text):
    sys.stderr.write(text)
    sys.stderr.flush()


def data_from(iterable):
    for entry in iterable:
        yield entry['data']


def cached_collections(zot):
    try:
        ckeys = zot._collection_keys
    except AttributeError:
        ckeys = {e['name']: e['key'] for e in data_from(zot.collections())}
        zot._collection_keys = ckeys
    return ckeys


def collection_items_top(zot, name, *, limit=8):
    """Return a verbose generator for items in a given collection name."""
    ck = cached_collections(zot)[name]
    log(f"Loading {name}")
    items = zot.collection_items_top(ck, limit=limit)
    log(".")
    for group in zot.makeiter(items):
        log("." * len(group))
        yield from group
    log("✔\n")


def format_name(c):
    if (name := c.get('name')):
        return name
    first = " ".join(w[0] + '.' for w in c['firstName'].split())
    last = c['lastName']
    return f"{first} {last}"


def format_names(creators, limit=1):
    creators = [c for c in creators if c['creatorType'] != "contributor"]
    if len(creators) < limit:
        return ", ".join(format_name(c) for c in creators)
    formatted_creators = [format_name(c) for c in creators[:limit]]
    formatted_creators.append("*et al*.")
    return ", ".join(formatted_creators)


def append_names(bits, creators, /, **kwargs):
    names = format_names(creators, **kwargs)
    if not names:
        return
    if names and not names.endswith('.'):
        names += '.'
    bits.append(names)


def format_presentation(e):
    bits = []
    append_names(bits, e['creators'], limit=3)
    bits.append("\"{title}\".".format(**e))
    if (meeting := e.get('meetingName')):
        bits.append(f"*{meeting}*,")
    date = parse_date(e['date'])
    bits.append(date.strftime("%d %b %Y."))
    if (url := e.get('url')):
        pt = e.get('presentationType', "").lower() or "presentation"
        bits.append(f"[{pt}]({url})")
    return " ".join(bits)


def format_paper(e):
    bits = []
    append_names(bits, e['creators'], limit=5)
    bits.append("\"{title}\".".format(**e))
    if (pub := e.get('publicationTitle')):
        bits.append(f"*{pub}*,")
    elif (proc := e.get('proceedingsTitle')):
        bits.append(f"in *{proc}*,")
    date = parse_date(e['date'])
    bits.append(date.strftime("%b %Y.").lstrip())
    if (doi := e.get('DOI')):
        bits.append(f"[{doi}](https://doi.org/{doi})")
    return " ".join(bits)


def format_software(e):
    bits = []
    append_names(bits, e['creators'], limit=100)
    title = e['title']
    if (version := e.get('version')):
        title = f"{title} *v{version}*"
    if (url := e.get('url')):
        title = f"[{title}]({url})"
    bits.append(f"\"{title}\".")
    date = parse_date(e['date'])
    bits.append(date.strftime("%b %Y.").lstrip())
    return " ".join(bits)


def sorted_data_by_date(items):
    entries = data_from(items)
    entries = (e for e in entries if e.get('date'))
    return sorted(entries, key=lambda e: parse_date(e['date']), reverse=True)


def print_bibliography(get_collection_items, out):

    print(f"""\
---
title: Celeritas publications
---
<!--
NOTE: this page is generated automatically from
https://github.com/celeritas-project/celeritas/tree/doc/gh-pages-base/scripts/generate-pubs.py
-->
# Publications

These publications are extracted from the Celeritas team's Zotero database.""", file=out)

    def print_subheader(name):
        print(f"\n## {name}\n", file=out)

    print_subheader("Conference papers")
    for e in get_collection_items("Conference papers"):
        print("-", format_paper(e), file=out)

    print_subheader("Presentations")
    for e in get_collection_items("Presentations"):
        print("-",format_presentation(e), file=out)

    print_subheader("Journal articles")
    for e in get_collection_items("Journal articles"):
        print("-",format_paper(e), file=out)

    print_subheader("Technical reports")
    for e in get_collection_items("Reports"):
        print("-",format_paper(e), file=out)

    print_subheader("Code")
    for e in get_collection_items("Code objects"):
        print("-",format_software(e), file=out)


def print_references(get_collection_items, out):

    print(f"""\
---
title: Celeritas references
---
<!--
NOTE: this page is generated automatically from
https://github.com/celeritas-project/celeritas/tree/doc/gh-pages-base/scripts/generate-pubs.py
-->
# References

These publications are extracted from the Celeritas team's Zotero database.""", file=out)

    def print_subheader(name):
        print(f"\n## {name}\n", file=out)

    print_subheader("Physics models and validation")
    for e in get_collection_items("Physics"):
        print("-", format_paper(e), file=out)

    print_subheader("HEP experiments")
    for e in get_collection_items("HEP experiments"):
        print("-",format_paper(e), file=out)

    print_subheader("Software implementations")
    for e in get_collection_items("Software"):
        print("-",format_paper(e), file=out)

    print_subheader("Computer science and mathematics")
    for e in get_collection_items("Computer science"):
        print("-",format_paper(e), file=out)

    print_subheader("Computational geometry")
    for e in get_collection_items("Geometry"):
        print("-",format_paper(e), file=out)


def run(group_id, outdir):
    zot = zotero.Zotero(group_id, "group", API_KEY)

    def sorted_collection(name):
        return sorted_data_by_date(collection_items_top(zot, name))

    with open(outdir / 'publications.md', 'w') as f:
        print_bibliography(sorted_collection, f)

    with open(outdir / 'references.md', 'w') as f:
        print_references(sorted_collection, f)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--group', type=int,
                        help="Zotero group ID")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory")
    args = parser.parse_args()
    outdir = Path(args.output or '.')
    run(args.group, outdir)


if __name__ == "__main__":
    main()
