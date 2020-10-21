#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
from pprint import pprint
import subprocess
from os import environ
from sys import exit

inp = {
    'grid_params': {
        'block_size': 128,
        'grid_size': 32,
    },
    'run': {
        'seed': 12345,
        'energy': 10, # MeV
        'num_tracks': 128 * 32,
        'max_steps': 128,
        'tally_grid': {
            'size': 1024,
            'front': -1,
            'delta': .25,
        }
    }
}

print("Input:")
pprint(inp)

exe = environ.get('CELERITAS_DEMO_EXE', './demo-interactor')
print("Running", exe)
result = subprocess.run([exe, '-'],
                        input=json.dumps(inp).encode(),
                        stdout=subprocess.PIPE)
if result.returncode:
    print("Run failed with error", result.returncode)
    exit(result.returncode)

print("Received {} bytes of data".format(len(result.stdout)))
out_text = result.stdout.decode()
try:
    out = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)

with open('demo-interactor.json', 'w') as f:
    json.dump(out, f, indent=1)

result = out['result']
num_tracks = result['alive'][0]
num_iters = len(result['edep'])
num_track_steps = sum(result['alive'])
print("Number of steps:", num_iters,
      "(average", num_track_steps / num_tracks, "per track)")
print("Fraction of time in kernel:", sum(result['time']) /
        result['total_time'])
