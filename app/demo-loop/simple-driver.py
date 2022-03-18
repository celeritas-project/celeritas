#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import re
import subprocess
from distutils.util import strtobool
from os import environ, path
from sys import exit, argv

try:
    (geometry_filename, hepmc3_filename) = argv[1:]
except ValueError:
    print("usage: {} inp.gdml inp.hepmc3".format(argv[0]))
    exit(1)

# We reuse the "disable device" environment variable, which prevents the GPU
# from being initialized at runtime.
use_device = not strtobool(environ.get('CELER_DISABLE_DEVICE', 'false'))
run_name = (path.splitext(path.basename(geometry_filename))[0]
            + ('-gpu' if use_device else '-cpu'))

geant_exp_exe = environ.get('CELER_EXPORT_GEANT_EXE', './geant-exporter')

if geant_exp_exe:
    physics_filename = run_name + ".root"
    result_ge = subprocess.run([geant_exp_exe,
                                geometry_filename,
                                physics_filename])
    if result_ge.returncode:
        print(f"fatal: {geant_exp_exe} failed with error {result_ge.returncode}")
        exit(result_ge.returncode)
else:
    # Load directly from Geant4 rather than ROOT file
    physics_filename = geometry_filename

if strtobool(environ.get('CELER_DISABLE_VECGEOM', 'false')):
    print("Replacing .gdml extension since VecGeom is disabled")
    geometry_filename = re.sub(r"\.gdml$", ".org.json", geometry_filename)

num_tracks = 128*32 if use_device else 32
num_primaries = 3 * 15 # assuming test hepmc input

inp = {
    'use_device': use_device,
    'geometry_filename': geometry_filename,
    'physics_filename': physics_filename,
    'hepmc3_filename': hepmc3_filename,
    'seed': 12345,
    'max_num_tracks': num_tracks,
    'max_steps': 128 if use_device else 64, # Just for sake of test time!
    'initializer_capacity': 100 * max([num_tracks, num_primaries]),
    'secondary_stack_factor': 3,
    'enable_diagnostics': True,
    'sync': True,
    # Physics options
    'rayleigh': True,
    'eloss_fluctuation': True,
    'brem_combined': True,
    'brem_lpm': True,
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

outfilename = f'{run_name}.out.json'
with open(outfilename, 'w') as f:
    json.dump(result, f)
print("Results written to", outfilename)

time = result['result']['time'].copy()
time.pop('steps')
print(json.dumps(time, indent=1))
