#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import subprocess
from os import environ
from sys import exit, argv

try:
    geometry_filename = argv[1]
    (physics_filename,) = argv[2:]
except TypeError:
    print("usage: {} inp.gdml inp.root".format(sys.argv[0]))
    exit(2)

inp = {
    'run': {
        'geometry_filename': geometry_filename,
        'physics_filename': physics_filename,
        'seed': 12345,
        'energy': 10, # MeV
        'num_tracks': 128 * 32,
        'max_steps': 128
    }
}

exe = environ.get('CELERITAS_DEMO_EXE', './demo-loop')

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
    result = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)

print(json.dumps(result, indent=1))
with open(f'{exe}.out.json', 'w') as f:
    json.dump(result, f)
