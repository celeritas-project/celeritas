#!/usr/bin/env python3
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""

import itertools
import os
import subprocess
import sys

try:
    BUILD = os.environ['SCALE_BUILD_DIR']
except KeyError as ke:
    print(f"missing environment variable {ke}", file=sys.stderr)
    sys.exit(1)

sys.path[:0] = [BUILD + dir
                for dir in ["/python", "/src/geometria", "/src/ampx",
                            "/src/nemesis"]]

def main(*args):
    from orangeinp.celeritas import main as generate_celer

    for fn in args:
        basename, _ = os.path.splitext(fn)
        try:
            generate_celer(["-o", basename + ".json", fn])
        except SystemExit:
            pass
        os.unlink(basename + ".xml")

if __name__ == '__main__':
    main(*sys.argv[1:])
