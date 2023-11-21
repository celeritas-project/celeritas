#!/usr/bin/env python3
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Run celer-g4.
"""
import json
import re
import subprocess
from os import environ, path
from pprint import pprint
from sys import exit, argv, stderr

def strtobool(text):
    text = text.lower()
    if text in {"true", "on", "yes", "1"}:
        return True
    if text in {"false", "off", "no", "0"}:
        return False
    raise ValueError(text)

#### LOAD OPTIONS ####

# Load from environment
use_device = not strtobool(environ.get("CELER_DISABLE_DEVICE", "false"))
use_root = strtobool(environ["CELER_USE_ROOT"])
use_celeritas = not strtobool(environ.get("CELER_DISABLE", "false"))
build = environ.get("CELER_BUILD_TYPE", "unknown").lower()
ext = environ.get("CELER_TEST_EXT", "unknown")

# Load from arguments
try:
    (exe, model_file, events_file) = argv[1:]
except ValueError:
    print(f"usage: {argv[0]} celer-g4 inp.gdml inp.hepmc3")
    exit(1)

problem_name = "-".join([
    path.splitext(path.basename(model_file))[0],
    ext
])

#### BUILD INPUT  ####

offload_file = ".".join([
    problem_name,
    "offloaded",
    "root" if use_root else "hepmc3"
])
inp_file = f"{problem_name}.inp.json"
out_file = f"{problem_name}.out.json"

if use_device:
    if build == "release":
        # GPU release
        max_tracks = 2**19
        init_capacity = 2**22
    else:
        # GPU debug
        max_tracks = 2**11
        init_capacity = 2**16
else:
    # CPU
    max_tracks = 2**10
    init_capacity = 2**15

inp = {
    "geometry_file": model_file,
    "event_file": events_file,
    "output_file": out_file,
    "offload_output_file": offload_file,
    "num_track_slots": max_tracks,
    "max_events": 1024,
    "initializer_capacity": init_capacity,
    "secondary_stack_factor": 2,
    "physics_list": "ftfp_bert",
    "field_type": "uniform",
    "field": [ 0.0, 0.0, 1.0 ],
    "field_options": {
     "minimum_step": 0.000001,
     "delta_chord": 0.025,
     "delta_intersection": 0.00001,
     "epsilon_step": 0.00001
    },
    "sd_type": "event_hit" if use_root else "simple_calo",
    "step_diagnostic": ext == "none",
    "step_diagnostic_bins": 8,
}

with open(inp_file, "w") as f:
    json.dump(inp, f, indent=1)

print("Running", exe, inp_file, file=stderr)
result = subprocess.run([exe, inp_file])

if result.returncode:
    print("fatal: run failed with error", result.returncode)
    exit(result.returncode)

with open(out_file) as f:
    result = json.load(f)

pprint(result["result"])
