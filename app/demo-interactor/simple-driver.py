#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import subprocess
from os import environ
from sys import exit

inp = {
    'grid_params': {
        'threads_per_block': 128,
        'sync': False,
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

exe = environ.get('CELERITAS_DEMO_EXE', './demo-interactor')
environ['CELER_PROFILE_DEVICE'] = "1"

print("Input:")
with open(f'{exe}.inp.json', 'w') as f:
    json.dump(inp, f, indent=1)
print(json.dumps(inp, indent=1))

print("Running", exe)
result = subprocess.run([exe, '-'],
                        input=json.dumps(inp).encode(),
                        stdout=subprocess.PIPE)

if result.returncode:
    print("fatal: run failed with error", result.returncode)
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

with open(f'{exe}.out.json', 'w') as f:
    json.dump(out, f, indent=1)

result = out['result']
num_tracks = result['alive'][0]
num_iters = len(result['edep'])
num_track_steps = sum(result['alive'])
print("Number of steps:", num_iters,
      "(average", num_track_steps / num_tracks, "per track)")
print("Fraction of time in kernel:", sum(result['time']) /
        result['total_time'])

runtime = out['runtime']
if runtime['device'] is not None:
    print("Device: {} with {:.1f} GB memory".format(
        runtime['device']['name'],
        runtime['device']['total_global_mem'] / 1e9)
    )

    kernel_stats = {k.pop('name'): k for k in runtime['kernels']}
    print("Recorded {} kernels".format(len(kernel_stats)))
    k = kernel_stats['interact']
    print(f"interact was called {k['num_launches']} times "
          f"using {k['num_regs']} registers "
          f"with occupancy of {k['occupancy']}")
