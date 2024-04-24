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
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Will not plot: matplotlib is unavailable")
    plt = None
    viz = None
else:
    import importlib
    spec = importlib.util.spec_from_file_location(
        "viz",
        Path(__file__).parent / "visualize.py"
    )
    viz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz)

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

exe = environ.get("CELERITAS_DEMO_EXE", "./demo-rasterizer")
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

filename = f"{problem_name}.inp.jsonl"
with open(filename, 'w') as f:
    for c in commands:
        json.dump(c, f)
        f.write('\n')
print("Wrote input to", filename)

print("Running", exe)
result = subprocess.run([exe, filename],
                        stdout=subprocess.PIPE)
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
    elif plt is not None:
        (fig, ax) = plt.subplots(layout="constrained")
        viz.load_and_plot_image(ax, result)
        trace = result['trace']
        filename = f"{problem_name}-{trace['geometry']}-{trace['memspace']}.png"
        fig.savefig(filename, dpi=150)
        print(f"Saved to {filename}.png")

print(json.dumps(decode_line(out_lines[-1]), indent=1))
