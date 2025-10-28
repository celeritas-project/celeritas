#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Run celer-geo as part of a CMake-based test harness."""

from itertools import count
import json
import subprocess
from dataclasses import dataclass
from os import environ, getcwd
from sys import exit, stderr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def log(*args: Any, **kwargs: Any) -> None:
    """Print messages to stderr, not stdout."""
    print(*args, **kwargs, file=stderr)


def decode_line(jsonline: str) -> Dict[str, Any]:
    """Decode a JSON line, exiting on error."""
    try:
        return json.loads(jsonline)
    except json.decoder.JSONDecodeError as e:
        log("error: expected a JSON object but got the following stdout:")
        log(jsonline)
        log("fatal:", str(e))
        exit(1)


@dataclass
class Config:
    """Configuration for the test harness."""

    model_path: Path
    exe: Path
    ext: str
    problem_name: str
    env: dict

    @classmethod
    def from_environment(cls, model_path_str: str) -> "Config":
        """Create configuration from environment variables and arguments."""
        env = dict(environ)
        model_path = Path(model_path_str)
        exe = Path(env.setdefault("CELERITAS_EXE", "./celer-geo"))
        ext = env.get("CELER_TEST_EXT")
        if ext is None:
            raise RuntimeError("CELER_TEST_EXT environment variable must be defined")
        problem_name = "-".join([model_path.stem, ext])

        env["CELER_LOG"] = "debug"

        return cls(
            model_path=model_path, exe=exe, ext=ext, problem_name=problem_name, env=env
        )

    def make_problem_path(self, ext: str) -> Path:
        """Generate a path for problem output files."""
        return Path("".join([self.problem_name, ext]))


class CelerGeoDriver:
    """Driver for running celer-geo executable."""

    def __init__(self, config: Config):
        self.config = config
        self._command_counter = count()

        (returncode, lines) = self([])
        if returncode or len(lines) != 1:
            log("Initial config run failed with error", returncode)
            exit(returncode)

        self.build_config = build_config = json.loads(lines[0])
        log("Build config:", repr(build_config))
        self.enabled_deps = set(build_config["config"]["use"])

    def __call__(
        self, commands: List[Dict[str, Any]], env: Optional[Dict[str, str]] = None
    ) -> Tuple[int, List[str]]:
        """Write commands to a file and run the celer-geo executable."""
        # Write commands to a uniquely named input file
        ctr = next(self._command_counter)
        inp_path = self.config.make_problem_path(f".{ctr}.inp.jsonl")

        with open(inp_path, "w") as f:
            for c in commands:
                try:
                    json.dump(c, f)
                except TypeError:
                    log(repr(c))
                    raise
                f.write("\n")
            f.write("\n")

        # Run the executable
        log("Running", self.config.exe, inp_path, "from", getcwd())

        runenv = self.config.env
        if env is not None:
            runenv = runenv.copy()
            runenv.update(env)

        result = subprocess.run(
            [self.config.exe, inp_path],
            stdout=subprocess.PIPE,
            env=runenv,
        )

        num_bytes = len(result.stdout)
        out_path = self.config.make_problem_path(f".{ctr}.out.jsonl")
        log(
            f"Received {num_bytes} bytes of data via stdout and echoed to {out_path.absolute()}"
        )

        with open(out_path, "wb") as f:
            f.write(result.stdout)

        lines = result.stdout.decode().split("\n")
        if lines and not lines[-1]:
            lines.pop()

        return (result.returncode, lines)


IMAGES = {
    "simple-cms": {
        "_units": "cgs",
        "lower_left": [-800, 0, -1500],
        "upper_right": [800, 0, 1600],
        "rightward": [1, 0, 0],
        "vertical_pixels": 128,
    },
    "nonexistent": {},
}


def run_error_check_test(config: Config, run_celer_geo: CelerGeoDriver) -> int:
    """Run the error checking test case."""
    returncode, lines = run_celer_geo(
        [
            {"geometry_file": "nonexistent.gdml"},
        ]
    )

    if len(lines) != 1:
        log("Unexpected number of lines: ", lines)
        return 1

    exception_obj = json.loads(lines[0])
    log("Result:", exception_obj)

    if returncode:
        assert exception_obj["type"] == "RuntimeError"
        log("Successfully checked failure mode for the app")
    else:
        log("Did not fail as expected")
        return 1

    # Try deprecated/erroneous commands
    commands = [
        {"geometry_file": str(config.model_path)},
        {
            # DEPRECATED: omitting _cmd should work until v1.0
            "image": IMAGES[config.model_path.stem],
            "volumes": True,
            "bin_file": str(config.make_problem_path(".orange.bin")),
        },
        {
            "_cmd": "trace",
            "geometry": "bad",
            "bin_file": str(config.make_problem_path(".bad.bin")),
        },
    ]
    if "vecgeom" not in run_celer_geo.enabled_deps:
        commands.append(
            {
                "_cmd": "trace",
                "bin_file": str(config.make_problem_path(".vecgeom.bin")),
                "geometry": "vecgeom",
            }
        )
    commands.extend(
        [
            # Test error conditions
            {"_cmd": "trace"},
            {"_cmd": "nonexistent"},
        ]
    )

    returncode, lines = run_celer_geo(commands)
    if returncode:
        log("Failures should have been graceful")
    else:
        log("Success: warnings and errors above were nonfatal")

    return returncode


def run_main_test(config: Config, run_celer_geo: CelerGeoDriver) -> int:
    """Run the main test with trace commands."""

    commands = [
        {"geometry_file": str(config.model_path)},
        {"_cmd": "orange_stats"},
        {
            "_cmd": "trace",
            "image": IMAGES[config.model_path.stem],
            "volumes": True,
            "bin_file": str(config.make_problem_path(".orange.bin")),
        },
        {
            "_cmd": "trace",
            # Reuse same image
            "bin_file": str(config.make_problem_path(".geant4.bin")),
            "geometry": "geant4",
        },
    ]
    if "vecgeom" in run_celer_geo.enabled_deps:
        commands.append(
            {
                "_cmd": "trace",
                "bin_file": str(config.make_problem_path(".vecgeom.bin")),
                "geometry": "vecgeom",
            }
        )

    perfetto_path: Optional[Path] = None
    g4orgopt = {}
    env = {}

    if "perfetto" in run_celer_geo.enabled_deps:
        perfetto_path = config.make_problem_path(".out.perfetto")
        perfetto_path.unlink(missing_ok=True)
        commands[0]["perfetto_file"] = str(perfetto_path)
        env["CELER_ENABLE_PROFILING"] = "1"

    if "geant4" in run_celer_geo.enabled_deps:
        g4orgopt["config"] = g4cfg = config.make_problem_path(".g4orgconf.json")
        env["G4ORG_OPTIONS"] = str(g4cfg)

        for k in ["csg", "org", "objects"]:
            path = config.make_problem_path(f".{k}.json")
            path.unlink(missing_ok=True)
            g4orgopt[f"{k}_output_file"] = path

        with open(g4cfg, "w") as f:
            json.dump({k: str(v) for k, v in g4orgopt.items()}, f)

    (returncode, out_lines) = run_celer_geo(commands, env=env)

    if returncode:
        log("Run failed with error", returncode)
        return returncode

    # Validate that expected output files exist
    if g4orgopt:
        g4org_debug_out = {}
        for k, outpath in g4orgopt.items():
            assert outpath.exists(), outpath
            with open(outpath) as f:
                g4org_debug_out[k] = json.load(f)

    if perfetto_path is not None:
        assert perfetto_path.exists()

    # Process and log results from output lines
    log("Setup:", decode_line(out_lines[0]))

    for line in out_lines[1:-1]:
        result = decode_line(line)
        if result.get("_label") == "exception":
            log("Failure may be ok:", json.dumps(result, indent=1))

    log("Run succeeded")

    summary = decode_line(out_lines[-1])
    summary.pop("runtime")
    print("Output (without runtime):", json.dumps(summary, indent=1))

    return 0


TEST_FIXTURES = {
    "errcheck": run_error_check_test,
    "cpu": run_main_test,
    "gpu": run_main_test,
}


def main() -> int:
    """Main entry point for the test harness."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run celer-geo as part of a CMake-based test harness."
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the input geometry file (e.g., inp.gdml)",
    )

    args = parser.parse_args()

    # Validate that the model path exists (unless running error check)
    config = Config.from_environment(args.model_path)
    run_celer_geo = CelerGeoDriver(config)
    run_fixture = TEST_FIXTURES[config.ext]
    return run_fixture(config, run_celer_geo)


if __name__ == "__main__":
    exit(main())
