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
        'block_size': 256,
        'grid_size': 64,
    },
    'run': {
        'seed': 12345,
        'energy': 1e2, # MeV
        'num_tracks': 256 * 256,
        'max_steps': 128,
    }
}

print("Input:")
pprint(inp)

exe = environ.get('CELERITAS_DEMO_EXE', './demo-interactor')
print("Running", exe)
with subprocess.Popen([exe, '-'],
                      stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE) as proc:
    (output, _) = proc.communicate(input=json.dumps(inp).encode())

print("Received {} bytes of data".format(len(output)))
out_text = output.decode()
try:
    result = json.loads(out_text)['result']
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)
pprint(result)

num_tracks = result['alive'][0]
num_iters = len(result['edep'])
num_track_steps = sum(result['alive'])
print("Number of steps:", num_iters,
      "(average", num_track_steps / num_tracks, " per track)")
print("Fraction of time in kernel:", sum(result['time']) /
        result['total_time'])
