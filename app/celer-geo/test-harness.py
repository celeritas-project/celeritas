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
    (model_file,) = argv[1:]
except TypeError:
    print("usage: {} inp.gdml".format(sys.argv[0]))
    exit(2)

exe = environ.get("CELERITAS_DEMO_EXE", "./demo-rasterizer")
ext = environ.get("CELER_TEST_EXT", "unknown")

problem_name = "-".join([Path(model_file).stem, ext])

eps = 0.01
inp = {
    'image': {
        # TODO: input is cm for now; add 'units' argument?
        'lower_left': [-800, 0, -1500],
        'upper_right': [800, 0, 1600],
        'rightward': [1, 0, 0],
        'vertical_pixels': 128,
    },
    'input': model_file,
    'output': f"{problem_name}.bin"
}
print("Input:")
with open(f"{problem_name}.inp.json", 'w') as f:
    json.dump(inp, f, indent=1)
print(json.dumps(inp, indent=1))

print("Running", exe)
result = subprocess.run([exe, '-'],
                        input=json.dumps(inp).encode(),
                        stdout=subprocess.PIPE)
if result.returncode:
    print("Run failed with error", result.returncode)
    exit(result.returncode)

print("Received {} bytes of data".format(len(result.stdout)))
out_text = result.stdout.decode()
try:
    result = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)
print(json.dumps(result["metadata"], indent=1))
with open(f'{problem_name}.out.json', 'w') as f:
    json.dump(result, f)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Skipping plot: matplotlib is unavailable")
    exit(0)

import importlib
spec = importlib.util.spec_from_file_location(
    "viz",
    Path(__file__).parent / "visualize.py"
)
viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz)

(fig, ax) = plt.subplots(layout="constrained")
viz.load_and_plot_image(ax, result)
fig.savefig(f"{problem_name}.png", dpi=150)
print(f"Saved to {problem_name}.png")
