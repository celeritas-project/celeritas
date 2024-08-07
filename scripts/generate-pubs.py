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
from pyzotero import zotero
from datetime import datetime
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
        ckeys ={e['name']: e['key'] for e in data_from(zot.collections())}
        zot._collection_keys = ckeys
    return ckeys


def collection_items(zot, name, *, limit=8):
    """Return a generator for items in a given collection name."""
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


def format_presentation(e):
    bits = []
    bits.append(format_names(e['creators'], limit=3))
    if not (last := bits[-1]).endswith('.'):
        bits[-1] = last + '.'
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
    bits.append(format_names(e['creators'], limit=5))
    if not (last := bits[-1]).endswith('.'):
        bits[-1] = last + '.'
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
    bits.append(format_names(e['creators'], limit=100))
    if not (last := bits[-1]).endswith('.'):
        bits[-1] = last + '.'
    bits.append("\"{title}\".".format(**e))
    date = parse_date(e['date'])
    bits.append(date.strftime("%b %Y.").lstrip())
    if (doi := e.get('DOI')):
        bits.append(f"[{doi}](https://doi.org/{doi})")
    return " ".join(bits)


def sorted_collection(zot, name):
    entries = data_from(collection_items(zot, name))
    entries = (e for e in entries if e.get('date'))
    return sorted(entries, key=lambda e: parse_date(e['date']), reverse=True)


def run(group_id, out):
    zot = zotero.Zotero(group_id, "group", API_KEY)

    today = datetime.today().strftime("%d %b %Y")
    print(f"""# Publications

These publications were extracted from the Celeritas team's Zotero database
on {today}.

## Conference papers
""", file=out)
    for e in sorted_collection(zot, "Conference papers"):
        print("-", format_paper(e), file=out)

    print(f"""
## Code
""", file=out)
    for e in sorted_collection(zot, "Code objects"):
        print("-",format_software(e), file=out)

    print(f"""
## Journal articles
""", file=out)
    for e in sorted_collection(zot, "Journal articles"):
        print("-",format_paper(e), file=out)

    print(f"""
## Presentations
""", file=out)
    for e in sorted_collection(zot, "Presentations"):
        print("-",format_presentation(e), file=out)

    print(f"""
## Technical reports
""", file=out)
    for e in sorted_collection(zot, "Reports"):
        print("-",format_paper(e), file=out)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--group', type=int,
                        help="Zotero group ID")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output filename (empty for stdout)")
    args = parser.parse_args()

    if not args.output:
        outfile = sys.stdout
    else:
        outfile = open(args.output, 'w')

    run(args.group, outfile)

if __name__ == "__main__":
    main()

