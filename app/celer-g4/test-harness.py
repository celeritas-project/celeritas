#!/usr/bin/env python3
# Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Run celer-g4.
"""
import json
import re
import subprocess
from os import environ, getcwd, path
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
use_device = not environ.get("CELER_DISABLE_DEVICE")
use_root = strtobool(environ["CELER_USE_ROOT"])
use_celeritas = not environ.get("CELER_DISABLE")
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
    "initializer_capacity": init_capacity,
    "secondary_stack_factor": 2,
    "physics_list": "celer_ftfp_bert",
    "field_type": "uniform",
    "field": [ 0.0, 0.0, 1.0 ],
    "field_options": {
     "minimum_step": 0.000001,
     "delta_chord": 0.025,
     "delta_intersection": 0.00001,
     "epsilon_step": 0.00001
    },
    "sd_type": "simple_calo",
    "step_diagnostic": ext == "none",
    "step_diagnostic_bins": 8,
}

if ext == "cpu-nonfatal":
    inp.update({
        "max_steps": 30,
        "environ": {
            "CELER_NONFATAL_FLUSH": "1",
        }
    })

kwargs = {}
args = [exe, inp_file]
if use_celeritas:
    # IO through streams should work with celeritas or g4 as driver, but just
    # do it here as an example
    args = [exe, inp_file]
    inp["output_file"] = "-"
    if not use_device:
        inp["slot_diagnostic_prefix"] = f"slot-diag-{ext}-"

    env = dict(environ)
    kwargs = dict(
        input=json.dumps(inp).encode(),
        stdout=subprocess.PIPE,
        env=env,
    )

with open(inp_file, "w") as f:
    json.dump(inp, f, indent=1)

envstr = " ".join(f"{k}={v}"
                  for (k, v) in environ.items()
                  if k.startswith("CELER_"))
print("Running", envstr, exe, inp_file, file=stderr)
print("working directory:", getcwd(), file=stderr)
result = subprocess.run(args, **kwargs)

if use_celeritas:
    # Decode the output, fail if the run failed
    out_text = result.stdout.decode()
    try:
        j = json.loads(out_text)
    except json.decoder.JSONDecodeError as e:
        print(f"error ({e}): expected a JSON object but got the following stdout:")
        print(out_text)
        j = {}
    else:
        with open(out_file, 'w') as f:
            json.dump(j, f, indent=1)

if result.returncode:
    print("fatal: run failed with error", result.returncode)
    if use_celeritas:
        try:
            j = json.loads(result.stdout.decode())
        except:
            pass
        else:
            out_file = f'{problem_name}.out.failed.json'
            with open(out_file, 'w') as f:
                json.dump(j, f, indent=1)
            print("Failure written to", out_file, file=stderr)
            j = {}

    exit(result.returncode)

if not use_celeritas:
    # Load written file
    with open(out_file) as f:
        j = json.load(f)

    # Rewrite with indentation
    with open(out_file, 'w') as f:
        json.dump(j, f, indent=1)

pprint(j["result"])
