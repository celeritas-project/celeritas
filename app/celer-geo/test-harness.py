#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import subprocess
from os import environ, getcwd
from sys import exit, argv, stderr
from pathlib import Path

try:
    (model_file,) = argv[1:]
except TypeError:
    print("usage: {} inp.gdml".format(sys.argv[0]))
    exit(2)

def decode_line(jsonline):
    try:
        return json.loads(jsonline)
    except json.decoder.JSONDecodeError as e:
        print("error: expected a JSON object but got the following stdout:")
        print(jsonline)
        print("fatal:", str(e))
        exit(1)

exe = environ.get("CELERITAS_EXE", "./celer-geo")
ext = environ.get("CELER_TEST_EXT", "unknown")

problem_name = "-".join([Path(model_file).stem, ext])

image = {
    "_units": "cgs",
    "_units": "cgs",
    "lower_left": [-800, 0, -1500],
    "upper_right": [800, 0, 1600],
    "rightward": [1, 0, 0],
    "vertical_pixels": 128,
}

commands = [
    {
        "geometry_file": model_file,
    },
    {
        "image": image,
        "volumes": True,
        "bin_file": f"{problem_name}.orange.bin",
    },
    {
        # Reuse image setup
        "bin_file": f"{problem_name}.geant4.bin",
        "geometry": "geant4",
    },
    {
        "bin_file": f"{problem_name}.vecgeom.bin",
        "geometry": "vecgeom",
    },
]

env = dict(environ)
if env["CMAKE_BUILD_TYPE"].lower() == "release":
    commands[0]["perfetto_file"] = "out.perfetto"
    env["CELER_ENABLE_PROFILING"] = "1"

filename = f"{problem_name}.inp.jsonl"
with open(filename, 'w') as f:
    for c in commands:
        json.dump(c, f)
        f.write('\n')

print("Running", exe, filename, "from", getcwd(), file=stderr)
result = subprocess.run([exe, filename],
                        stdout=subprocess.PIPE,
                        env=env)
if result.returncode:
    print("Run failed with error", result.returncode)
    exit(result.returncode)

print("Received {} bytes of data".format(len(result.stdout)))
with open(f'{problem_name}.out.json', 'wb') as f:
    f.write(result.stdout)
out_lines = result.stdout.decode().splitlines()

# Geometry diagnostic information
print(decode_line(out_lines[0]))

for line in out_lines[1:-1]:
    result = decode_line(line)
    if result.get("_label") == "exception":
        # Note that it's *OK* for the trace to fail e.g. if we have disabled
        # vecgeom or GPU
        print("Ray trace failed:")
        print(json.dumps(result, indent=1))

print(json.dumps(decode_line(out_lines[-1]), indent=1))
