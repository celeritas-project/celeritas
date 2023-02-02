#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import re
import subprocess
from distutils.util import strtobool
from os import environ, path
from sys import exit, argv, stderr

try:
    (geometry_filename, hepmc3_filename, rootout_filename) = argv[1:]
except ValueError:
    print("usage: {} inp.gdml inp.hepmc3 mctruth.root (use '' for no ROOT output)".format(argv[0]))
    exit(1)

# We reuse the "disable device" environment variable, which prevents the GPU
# from being initialized at runtime.
use_device = not strtobool(environ.get('CELER_DISABLE_DEVICE', 'false'))
use_vecgeom = not strtobool(environ.get('CELER_DISABLE_VECGEOM', 'false'))
geant_exp_exe = environ.get('CELER_EXPORT_GEANT_EXE', './celer-export-geant')

run_name = (path.splitext(path.basename(geometry_filename))[0]
            + ('-gpu' if use_device else '-cpu'))

geant_options = {
    'rayleigh': True,
    'eloss_fluctuation': True,
    'brems': "all",
    'lpm': True,
    'msc': "urban" if use_vecgeom else "none",
}

if geant_exp_exe:
    physics_filename = run_name + ".root"
    print("Running", geant_exp_exe, file=stderr)
    result_ge = subprocess.run(
        [geant_exp_exe, geometry_filename, "-", physics_filename],
        input=json.dumps(geant_options).encode()
    )

    if result_ge.returncode:
        print(f"fatal: {geant_exp_exe} failed with error {result_ge.returncode}")
        exit(result_ge.returncode)
else:
    # Load directly from Geant4 rather than ROOT file
    physics_filename = geometry_filename

if not use_vecgeom:
    print("Replacing .gdml extension since VecGeom is disabled", file=stderr)
    geometry_filename = re.sub(r"\.gdml$", ".org.json", geometry_filename)

num_tracks = 128*32 if use_device else 32
num_primaries = 3 * 15 # assuming test hepmc input
max_steps = 512 if geant_options['msc'] else 128

if not use_device:
    # Way more steps are needed since we're not tracking in parallel, but
    # shorten to an unreasonably small number to reduce test time.
    max_steps = 256

inp = {
    'use_device': use_device,
    'geometry_filename': geometry_filename,
    'physics_filename': physics_filename,
    'hepmc3_filename': hepmc3_filename,
    'mctruth_filename': rootout_filename,
    'seed': 12345,
    'max_num_tracks': num_tracks,
    'max_steps': max_steps,
    'initializer_capacity': 100 * max([num_tracks, num_primaries]),
    'max_events': 1000,
    'secondary_stack_factor': 3,
    'enable_diagnostics': True,
    'sync': True,
    'brem_combined': True,
    'geant_options': geant_options,
}


with open(f'{run_name}.inp.json', 'w') as f:
    json.dump(inp, f, indent=1)

exe = environ.get('CELERITAS_DEMO_EXE', './demo-loop')
print("Running", exe, file=stderr)
result = subprocess.run([exe, '-'],
                        input=json.dumps(inp).encode(),
                        stdout=subprocess.PIPE)
if result.returncode:
    print("fatal: run failed with error", result.returncode)
    try:
        j = json.loads(result.stdout.decode())
    except:
        pass
    else:
        outfilename = f'{run_name}.out.failed.json'
        with open(outfilename, 'w') as f:
            json.dump(j, f, indent=1)
        print("Failure written to", outfilename, file=stderr)

    exit(result.returncode)

print("Received {} bytes of data".format(len(result.stdout)), file=stderr)
out_text = result.stdout.decode()
# Filter out spurious HepMC3 output
out_text = out_text[out_text.find('\n{') + 1:]
try:
    j = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)

outfilename = f'{run_name}.out.json'
with open(outfilename, 'w') as f:
    json.dump(j, f, indent=1)
print("Results written to", outfilename, file=stderr)

time = j['result']['time'].copy()
time.pop('steps')
print(json.dumps(time, indent=1))
