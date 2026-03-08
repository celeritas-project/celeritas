#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Test harness for celer-optical for use in CMake."""

import json
import subprocess
from os import environ, getcwd
from pathlib import Path
from sys import exit, argv, stderr

try:
    geometry_file = argv[1]
except IndexError:
    print(f"usage: {argv[0]} inp.gdml")
    exit(1)

core_geo = environ.get("CELER_CORE_GEO", "ORANGE").lower()
use_device = not environ.get("CELER_DISABLE_DEVICE")
run_name = Path(geometry_file).stem + ("-gpu" if use_device else "-cpu")

geant_setup = {
    "absorption": True,
    "rayleigh_scattering": True,
    "mie_scattering": True,
    "wavelength_shifting": {"enable": False},
    "wavelength_shifting2": {"enable": False},
    "cherenkov": {"enable": False},
    "scintillation": {"enable": False},
}

capacity = {
    "generators": 1,
    "primaries": 1,
    "tracks": 8192,
}

generator = {
    "_type": "primary",
    "primaries": 1048576,
    "angle": {"_type": "isotropic"},
    "energy": {"_type": "normal", "mean": 5e-5, "stddev": 5e-6},
    "shape": {"_type": "delta", "value": [0, 0, 0]},
}

problem = {
    "model": {"geometry": geometry_file},
    "generator": generator,
    "capacity": capacity,
    "seed": 12345,
    "timers": {"action": False},
}

inp = {
    "problem": problem,
    "geant_setup": geant_setup,
}

if use_device:
    inp["system"] = {"device": {}}

inp_file = f"{run_name}.inp.json"
with open(inp_file, "w") as f:
    json.dump(inp, f, indent=1)

exe = environ.get("CELERITAS_EXE", "./celer-optical")
print(f"Running {exe} {inp_file} from {getcwd()}", file=stderr)
result = subprocess.run(
    [exe, "-"], input=json.dumps(inp).encode(), stdout=subprocess.PIPE
)
if result.returncode:
    print(f"fatal: run failed with error {result.returncode}")
    try:
        j = json.loads(result.stdout.decode())
    except:
        pass
    else:
        out_file = f"{run_name}.out.failed.json"
        with open(out_file, "w") as f:
            json.dump(j, f, indent=1)
        print(f"Failure written to {out_file}", file=stderr)
    exit(result.returncode)

print(f"Received {len(result.stdout)} bytes of data", file=stderr)
out_text = result.stdout.decode()
try:
    j = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print(f"fatal: {e}")
    exit(1)

out_file = f"{run_name}.out.json"
with open(out_file, "w") as f:
    json.dump(j, f, indent=1)
print(f"Results written to {out_file}", file=stderr)

# Check results
expected_generators = {
    "buffer_size": 0,
    "num_generated": 1048576,
    "num_pending": 0,
}
counters = j["result"]["counters"].copy()
assert counters["flushes"] == 1
assert len(counters["generators"]) == 1
assert counters["generators"][0] == expected_generators

expected_sizes = {
    "generators": 1,
    "tracks": 8192,
}
sizes = j["internal"]["optical-sizes"].copy()
assert sizes.pop("streams") == 1
assert sizes == expected_sizes

print(json.dumps(j["result"]["time"], indent=1))
