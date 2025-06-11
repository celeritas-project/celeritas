#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import re
import subprocess
from os import environ, getcwd, path
from sys import exit, argv, stderr

try:
    (geometry_filename, event_filename, rootout_filename) = argv[1:]
except ValueError:
    print("usage: {} inp.gdml inp.hepmc3 mctruth.root (use '' for no ROOT output)".format(argv[0]))
    exit(1)

# We reuse the "disable device" environment variable, which prevents the GPU
# from being initialized at runtime if non-empty.
use_device = not environ.get('CELER_DISABLE_DEVICE')
core_geo = environ.get('CELER_CORE_GEO', 'ORANGE').lower()
geant_exp_exe = environ.get('CELER_EXPORT_GEANT_EXE', './celer-export-geant')

run_name = (path.splitext(path.basename(geometry_filename))[0]
            + ('-gpu' if use_device else '-cpu'))

physics_options = {
    'coulomb_scattering': False,
    'compton_scattering': True,
    'photoelectric': True,
    'rayleigh_scattering': True,
    'gamma_conversion': True,
    'gamma_general': False,
    'ionization': True,
    'annihilation': True,
    'brems': "all",
    'msc': "urban" if core_geo == "vecgeom" else "none",
    'eloss_fluctuation': True,
    'lpm': True,
    'optical': {
        'absorption': True,
        'rayleigh_scattering': True,
        'wavelength_shifting': "exponential",
        'wavelength_shifting2': "exponential"
    }
}

physics_filename = None
if geant_exp_exe:
    physics_filename = run_name + ".root"
    inp_file = f'{run_name}.geant.json'
    with open(inp_file, 'w') as f:
        json.dump(physics_options, f, indent=1)

    args = [geant_exp_exe, geometry_filename, inp_file, physics_filename]
    print("Running", args, file=stderr)

    # Test using stdin instead of filename
    args[2] = "-"
    result_ge = subprocess.run(args, input=json.dumps(physics_options).encode())

    if result_ge.returncode:
        print(f"fatal: {geant_exp_exe} failed with error {result_ge.returncode}")
        exit(result_ge.returncode)

if core_geo == "orange-json":
    print("Replacing .gdml extension since VecGeom and Geant4 conversion are disabled", file=stderr)
    env['ORANGE_FORCE_INPUT'] = re.sub(r"\.gdml$", ".org.json", geometry_filename)

simple_calo = []
if not rootout_filename and "cms" in geometry_filename:
    simple_calo = ["si_tracker", "em_calorimeter"]

num_tracks = 128 * 32 if use_device else 32
num_primaries = 3 * 15 # assuming test hepmc input
max_steps = 512 if physics_options['msc'] else 128

if not use_device:
    # Way more steps are needed since we're not tracking in parallel, but
    # shorten to an unreasonably small number to reduce test time.
    max_steps = 256

inp = {
    'use_device': use_device,
    'geometry_file': geometry_filename,
    'event_file': event_filename,
    'mctruth_file': rootout_filename,
    'seed': 12345,
    'num_track_slots': num_tracks,
    'max_steps': max_steps,
    'interpolation': 'linear',
    'initializer_capacity': 100 * num_tracks,
    'secondary_stack_factor': 3,
    'action_diagnostic': True,
    'step_diagnostic': True,
    'step_diagnostic_bins': 200,
    'write_step_times': use_device,
    'simple_calo': simple_calo,
    'action_times': True,
    'merge_events': False,
    'physics_options': physics_options,
    'field': None,
    'slot_diagnostic_prefix': f"slot-diag-{run_name}-",
}

if "lar" in geometry_filename:
    num_optical_tracks = 4096
    inp['max_steps'] = 2
    inp['optical'] = {
        'num_track_slots': num_optical_tracks,
        'buffer_capacity': 3 * max_steps * num_optical_tracks,
        'initializer_capacity': 2048 * num_optical_tracks,
        'max_steps': 4,
        'auto_flush': num_optical_tracks,
    }

if "simple-cms" in geometry_filename:
    inp['merge_events'] = True

if physics_filename:
    inp['physics_file'] = physics_filename

inp_file = f'{run_name}.inp.json'
with open(inp_file, 'w') as f:
    json.dump(inp, f, indent=1)

exe = environ.get('CELERITAS_EXE', './celer-sim')
print("Running", exe, inp_file, "from", getcwd(), file=stderr)
result = subprocess.run(
    [exe, '-'],
    input=json.dumps(inp).encode(),
    stdout=subprocess.PIPE
)
if result.returncode:
    print("fatal: run failed with error", result.returncode)
    try:
        j = json.loads(result.stdout.decode())
    except:
        pass
    else:
        out_file = f'{run_name}.out.failed.json'
        with open(out_file, 'w') as f:
            json.dump(j, f, indent=1)
        print("Failure written to", out_file, file=stderr)

    exit(result.returncode)

print("Received {} bytes of data".format(len(result.stdout)), file=stderr)
out_text = result.stdout.decode()
try:
    j = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)

out_file = f'{run_name}.out.json'
with open(out_file, 'w') as f:
    json.dump(j, f, indent=1)
print("Results written to", out_file, file=stderr)

run_output = j['result']['runner']
time = run_output['time'].copy()
steps = time.pop('steps')
if use_device:
    assert steps
    assert len(steps[0]) == run_output['num_step_iterations'][0], steps[0]
else:
    # Step times disabled on CPU from input
    assert steps is None, steps

internal = j["internal"]
core_sizes = internal["core-sizes"].copy()
num_streams = core_sizes.pop("streams")
if "openmp" not in j["system"]["build"]["config"]["use"]:
    assert num_streams == 1
if inp["merge_events"]:
    assert num_streams == 1

expected_core_sizes = None
expected_opt_sizes = None
if not use_device:
    expected_core_sizes =  {
       "events": 4,
       "initializers": 3200,
       "processes": 1,
       "secondaries": 96,
       "tracks": 32
      }
if not use_device and "lar" in geometry_filename:
    expected_opt_sizes = {
       "generators": 3145728,
       "initializers": 8388608,
       "tracks": 4096
    }


if expected_core_sizes:
    assert core_sizes == expected_core_sizes, core_sizes
if expected_opt_sizes:
    opt_sizes = internal["optical-sizes"].copy()
    assert num_streams == opt_sizes.pop("streams")
    assert opt_sizes == expected_opt_sizes, opt_sizes

print(json.dumps(time, indent=1))
