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
from sys import exit, argv

try:
    (gdml_filename,) = argv[1:]
except TypeError:
    print("usage: {} inp.gdml".format(sys.argv[0]))
    exit(2)

inp = {
    'image': {
        # TODO: input is cm for now; add 'units' argument?
        'lower_left': [-10, -10, 0],
        'upper_right': [10, 10, 0],
        'rightward_ax': [1, 0, 0],
        'vertical_pixels': 32
    },
    'input': gdml_filename,
    'output': 'two-boxes.bin'
}

exe = environ.get('CELERITAS_DEMO_EXE', './demo-rasterizer')

print("Input:")
with open(f'{exe}.inp.json', 'w') as f:
    json.dump(inp, f, indent=1)
print(json.dumps(inp, indent=1))

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
    result = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)
print(json.dumps(result, indent=1))
with open(f'{exe}.out.json', 'w') as f:
    json.dump(result, f)
