#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Run celer-geo as part of a CMake-based test harness."""

import json
import subprocess
from os import environ, getcwd
from sys import exit, argv, stderr, stdout
from pathlib import Path
from functools import partial

try:
    (model_path,) = argv[1:]
except TypeError:
    print("usage: {} inp.gdml".format(argv[0]))
    exit(2)


# Print messages to stderr, not stdout
def log(*args, **kwargs):
    return print(*args, **kwargs, file=stderr)


model_path = Path(model_path)
exe = Path(environ.get("CELERITAS_EXE", "./celer-geo"))
ext = environ.get("CELER_TEST_EXT", "unknown")

problem_name = "-".join([model_path.stem, ext])


def decode_line(jsonline):
    try:
        return json.loads(jsonline)
    except json.decoder.JSONDecodeError as e:
        log("error: expected a JSON object but got the following stdout:")
        log(jsonline)
        log("fatal:", str(e))
        exit(1)


def make_problem_path(ext):
    return Path("".join([problem_name, ext]))


def write_commands(commands):
    inp_path = make_problem_path(".inp.jsonl")
    with open(inp_path, "w") as f:
        for c in commands:
            try:
                json.dump(c, f)
            except TypeError:
                log(repr(c))
                raise
            f.write("\n")
        f.write("\n")
    return inp_path

# NOTE: it's important to inherit the environment rather than replacing it for
# apps; MPI will fail if HOME is undefined, for example
env = dict(environ)
env["CELER_LOG"] = "debug"

# Error check test is a separate case and ignores the model path
if ext == "errcheck":
    inp_path = write_commands(
        [
            {
                "geometry_file": "nonexistent.gdml",
            },
        ]
    )
    log("Running", exe, inp_path, "from", getcwd())
    result = subprocess.run(
        [exe, inp_path],
        stdout=subprocess.PIPE,
        env=env,
    )
    lines = result.stdout.decode().split('\n')
    if len(lines) != 1:
        log("Unexpected number of lines: ", lines)
    exception_obj = json.loads(lines[0])
    log("Result:", exception_obj)
    if result.returncode:
        assert exception_obj['type'] == "RuntimeError"
        log("Initial config run returned with error", result.returncode)
        log("Successfully checked failure mode for the app")
        exit(0)

    exit(result.returncode)

# Run to get configuration
log("Running", exe, "from", getcwd())
result = subprocess.run(
    [exe, "-"], input=b"\n", stdout=subprocess.PIPE, env=env,
)
if result.returncode:
    log("Initial config run failed with error", result.returncode)
    exit(result.returncode)
build_config = json.loads(result.stdout.decode())
log("Build config:", repr(build_config))

image = {
    "_units": "cgs",
    "lower_left": [-800, 0, -1500],
    "upper_right": [800, 0, 1600],
    "rightward": [1, 0, 0],
    "vertical_pixels": 128,
}

commands = [
    {
        "geometry_file": str(model_path),
    },
    {
        "_cmd": "orange_stats",
    },
    {
        "_cmd": "trace",
        "image": image,
        "volumes": True,
        "bin_file": str(make_problem_path(".orange.bin")),
    },
    {
        # Reuse image setup
        "_cmd": "trace",
        "bin_file": str(make_problem_path(".geant4.bin")),
        "geometry": "geant4",
    },
    {
        # DEPRECATED: omitting _cmd should work until v1.0
        "bin_file": str(make_problem_path(".vecgeom.bin")),
        "geometry": "vecgeom",
    },
    # Test error conditions
    {
        "_cmd": "trace",
        "geometry": "bad",
        "bin_file": str(make_problem_path(".bad.bin")),
    },
    {
        "_cmd": "trace",
    },
    {
        "_cmd": "nonexistent",
    },
]

enabled_deps = set(build_config["config"]["use"])

env = dict(environ)
perfetto_path = None
if "perfetto" in enabled_deps:
    perfetto_path = make_problem_path(".out.perfetto")
    perfetto_path.unlink(missing_ok=True)
    commands[0]["perfetto_file"] = str(perfetto_path)
    env["CELER_ENABLE_PROFILING"] = "1"

g4orgopt = None
if "geant4" in enabled_deps:
    g4orgopt = {"config": make_problem_path(".g4orgconf.json")}
    for k in ["csg", "org", "objects"]:
        path = make_problem_path(f".{k}.json")
        path.unlink(missing_ok=True)
        g4orgopt[f"{k}_output_file"] = path

    with open(g4orgopt["config"], "w") as f:
        json.dump({k: str(v) for k, v in g4orgopt.items()}, f)
    env["G4ORG_OPTIONS"] = str(g4orgopt["config"])
env["CELER_LOG"] = "debug"

inp_path = write_commands(commands)
log("Running", exe, inp_path, "from", getcwd())
result = subprocess.run([exe, inp_path], stdout=subprocess.PIPE, env=env)
if result.returncode:
    log("Run failed with error", result.returncode)
    exit(result.returncode)

num_bytes = len(result.stdout)
outfile = make_problem_path(".out.jsonl")
log(f"Received {num_bytes} bytes of data via stdin and echoed to {outfile.absolute()}")
with open(outfile, "wb") as f:
    f.write(result.stdout)
out_lines = result.stdout.decode().splitlines()

# Check that the converted ORANGE output is valid
if g4orgopt is not None:
    g4org_debug_out = {}
    for k, outpath in g4orgopt.items():
        assert outpath.exists(), outpath
        with open(outpath) as f:
            g4org_debug_out[k] = json.load(f)

# Check that profiling information was written
if perfetto_path is not None:
    assert perfetto_path.exists()

# Geometry diagnostic information
log("Setup:", decode_line(out_lines[0]))

for line in out_lines[1:-1]:
    result = decode_line(line)
    if result.get("_label") == "exception":
        # Note that it's *OK* for the trace to fail e.g. if we have disabled
        # vecgeom or GPU
        log("Failure may be ok:", json.dumps(result, indent=1))

log("Run succeeded")

summary = decode_line(out_lines[-1])
summary.pop("runtime")
print("Output (without runtime):", json.dumps(summary, indent=1))
