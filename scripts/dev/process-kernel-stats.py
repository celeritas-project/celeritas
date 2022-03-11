#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Write kernel launch bound information to be used by the gen-kernel.py scripts.
"""

from collections import defaultdict
import json

DEVICE_KEYS = ['capability_major', 'capability_minor', 'eu_per_cu', 'name',
               'platform', 'threads_per_warp']
KERNEL_KEY_MAP = {
    #'threads_per_block': 'max_threads_per_block',
    'max_threads_per_block': 'max_threads_per_block',
    'max_blocks_per_cu': 'min_blocks_per_cu',
    'max_warps_per_eu': 'min_warps_per_eu',
}

def run(input, output, key):
    # Convert
    devices = []
    kernels = defaultdict(list)
    for inp_file in input:
        with open(inp_file) as f:
            system = json.load(f)
        if key:
            system = system[key]
        device = system['device']
        devices.append({k: device[k] for k in DEVICE_KEYS})
        for kernel in system['kernels']:
            kernels[kernel['name']].append({newk: kernel[k]
                        for (k, newk) in KERNEL_KEY_MAP.items()})

    with open(output, 'w') as f:
        json.dump({
            'devices': devices,
            'kernels': dict(kernels)
        }, f, indent=1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--key',
        default="system",
        help="System information key in each input JSON file"
    )
    parser.add_argument(
        '--output', '-o',
        default="cmake/CeleritasUtils/launch-bounds.json",
        help="JSON output filename"
    )
    parser.add_argument(
        'input',
        nargs='+',
        help="Runtime JSON outputs with system information"
    )

    run(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()

