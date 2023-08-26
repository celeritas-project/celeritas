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
    (geometry_filename, event_filename, rootout_filename) = argv[1:]
except ValueError:
    print("usage: {} inp.gdml inp.hepmc3 mctruth.root (use '' for no ROOT output)".format(argv[0]))
    exit(1)

# We reuse the "disable device" environment variable, which prevents the GPU
# from being initialized at runtime.
use_device = not strtobool(environ.get('CELER_DISABLE_DEVICE', 'false'))
core_geo = environ.get('CELER_CORE_GEO', 'ORANGE').lower()
geant_exp_exe = environ.get('CELER_EXPORT_GEANT_EXE', './celer-export-geant')

run_name = (path.splitext(path.basename(geometry_filename))[0]
            + ('-gpu' if use_device else '-cpu'))

geant_options = {
    'coulomb_scattering': False,
    'compton_scattering': True,
    'photoelectric': True,
    'rayleigh_scattering': True,
    'gamma_conversion': True,
    'gamma_general': False,
    'ionization': True,
    'annihilation': True,
    'brems': "all",
    'msc': "urban_extended" if core_geo == "vecgeom" else "none",
    'eloss_fluctuation': True,
    'lpm': True,
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

if core_geo == "orange":
    print("Replacing .gdml extension since VecGeom is disabled", file=stderr)
    geometry_filename = re.sub(r"\.gdml$", ".org.json", geometry_filename)

simple_calo = []
if not rootout_filename:
    simple_calo = ["si_tracker", "em_calorimeter"]

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
    'event_filename': event_filename,
    'mctruth_filename': rootout_filename,
    'seed': 12345,
    'num_track_slots': num_tracks,
    'max_steps': max_steps,
    'initializer_capacity': 100 * max([num_tracks, num_primaries]),
    'max_events': 1000,
    'secondary_stack_factor': 3,
    'action_diagnostic': True,
    'step_diagnostic': True,
    'step_diagnostic_maxsteps': 200,
    'simple_calo': simple_calo,
    'sync': True,
    'merge_events': False,
    'default_stream': False,
    'brem_combined': True,
    'geant_options': geant_options,
}

with open(f'{run_name}.inp.json', 'w') as f:
    json.dump(inp, f, indent=1)

exe = environ.get('CELERITAS_DEMO_EXE', './celer-sim')
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

time = j['result']['runner']['time'].copy()
time.pop('steps')
print(json.dumps(time, indent=1))
