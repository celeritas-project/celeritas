#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
from distutils.util import strtobool
import json
import subprocess
from os import environ, path
from sys import exit, argv

try:
    geometry_filename = argv[1]
    hepmc3_filename = argv[2]
except (IndexError, TypeError):
    print("usage: {} inp.gdml inp.hepmc3".format(argv[0]))
    exit(2)

use_device = bool(strtobool(environ.get('CELERITAS_USE_DEVICE', 'true')))
run_name = (path.splitext(path.basename(geometry_filename))[0]
            + '-gpu' if use_device else '-cpu')

geant_exp_exe = environ.get('CELERITAS_GEANT_EXPORTER_EXE', './geant-exporter')
physics_filename = run_name + ".root"

result_ge = subprocess.run([geant_exp_exe,
                            geometry_filename,
                            physics_filename])

if result_ge.returncode:
    print("fatal: geant-exporter failed with error", result_ge.returncode)
    exit(result_ge.returncode)

storage_factor = 10 if use_device else 100
max_num_tracks = 128*32 if use_device else 1

inp = {
    'run': {
        'use_device': use_device,
        'geometry_filename': geometry_filename,
        'physics_filename': physics_filename,
        'hepmc3_filename': hepmc3_filename,
        'seed': 12345,
        'max_num_tracks': max_num_tracks,
        'max_steps': 128,
        'storage_factor': storage_factor
    }
}


print("Input:")
with open(f'{run_name}.inp.json', 'w') as f:
    json.dump(inp, f, indent=1)
print(json.dumps(inp, indent=1))

exe = environ.get('CELERITAS_DEMO_EXE', './demo-loop')
print("Running", exe)
result = subprocess.run([exe, '-'],
                        input=json.dumps(inp).encode(),
                        stdout=subprocess.PIPE)
if result.returncode:
    print("fatal: run failed with error", result.returncode)
    exit(result.returncode)

print("Received {} bytes of data".format(len(result.stdout)))
out_text = result.stdout.decode()
# Filter out spurious HepMC3 output
out_text = out_text[out_text.find('\n{') + 1:]
try:
    result = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)

print(json.dumps(result, indent=1))
with open(f'{run_name}.out.json', 'w') as f:
    json.dump(result, f)
