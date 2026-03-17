#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Test harness for celer-sim for use in CMake."""

import argparse
import json
import subprocess
from dataclasses import dataclass
from os import environ
from pathlib import Path
from sys import exit, stderr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test harness for celer-sim for use in CMake."
    )
    parser.add_argument("geometry_file", help="Input geometry file (GDML)")
    parser.add_argument("event_file", help="Input event file (HepMC3)")
    parser.add_argument(
        "rootout_file",
        help="Output ROOT MC truth file (use empty string for none)",
    )
    return parser.parse_args()


@dataclass
class RunConfig:
    geometry_file: str
    event_file: str
    rootout_file: str
    run_name: str
    use_device: bool
    physics_options: dict
    physics_filename: str | None


def get_em_physics_options(core_geo: str):
    return {
        "coulomb_scattering": False,
        "compton_scattering": True,
        "photoelectric": True,
        "rayleigh_scattering": True,
        "gamma_conversion": True,
        "gamma_general": False,
        "ionization": True,
        "annihilation": True,
        "brems": "all",
        "msc": "urban" if core_geo == "vecgeom" else "none",
        "eloss_fluctuation": True,
        "lpm": True,
    }


def run_geant_export(
    geometry_file: str, run_name: str, physics_options: dict
) -> str | None:
    """Run celer-export-geant and return the output physics filename, or None."""
    geant_exp_exe = environ.get("CELER_EXPORT_GEANT_EXE", "./celer-export-geant")
    if not geant_exp_exe:
        return None

    physics_filename = run_name + ".root"
    inp_file = Path(f"{run_name}.geant.json")
    inp_file.write_text(json.dumps(physics_options, indent=1))

    # Test using stdin instead of filename
    args = [geant_exp_exe, geometry_file, "-", physics_filename]
    print("Running", args, file=stderr)
    result = subprocess.run(args, input=json.dumps(physics_options).encode())
    if result.returncode:
        print(f"fatal: {geant_exp_exe} failed with error {result.returncode}")
        exit(result.returncode)

    return physics_filename


def build_input(cfg: RunConfig) -> dict:
    num_tracks = 128 * 32 if cfg.use_device else 32
    max_steps = 512 if cfg.physics_options["msc"] else 128

    if not cfg.use_device:
        # Way more steps are needed since we're not tracking in parallel, but
        # shorten to an unreasonably small number to reduce test time.
        max_steps = 256

    simple_calo: list[str] = []
    if not cfg.rootout_file and "cms" in cfg.geometry_file:
        simple_calo = ["si_tracker", "em_calorimeter"]
        # Volumes (needed for the simple calo) are currently only loaded if Geant4
        # import is enabled
        cfg.physics_filename = None

    inp = {
        "use_device": cfg.use_device,
        "geometry_file": cfg.geometry_file,
        "event_file": cfg.event_file,
        "mctruth_file": cfg.rootout_file,
        "seed": 12345,
        "num_track_slots": num_tracks,
        "max_steps": max_steps,
        "interpolation": "linear",
        "initializer_capacity": 100 * num_tracks,
        "secondary_stack_factor": 3,
        "action_diagnostic": True,
        "step_diagnostic": True,
        "step_diagnostic_bins": 200,
        "write_step_times": cfg.use_device,
        "simple_calo": simple_calo,
        "action_times": True,
        "merge_events": False,
        "physics_options": cfg.physics_options,
        "field": None,
        "slot_diagnostic_prefix": f"slot-diag-{cfg.run_name}-",
    }

    if "lar" in cfg.geometry_file:
        # Disable physics export: not implemented for optical
        cfg.physics_filename = None

        cfg.physics_options["optical"] = {
            "absorption": True,
            "rayleigh_scattering": True,
            "wavelength_shifting": None,
            "wavelength_shifting2": None,
        }

        num_optical_tracks = 8192
        optical_capacity = {
            "tracks": num_optical_tracks,
            "generators": 4 * max_steps * num_optical_tracks,
            "primaries": num_optical_tracks,
        }
        inp["optical"] = {
            "capacity": optical_capacity,
            "limits": {"steps": 7, "step_iters": 10},
        }
        inp["max_steps"] = 4

    if "simple-cms" in cfg.geometry_file:
        inp["merge_events"] = True
        inp["optical"] = None

    if cfg.physics_filename:
        inp["physics_file"] = cfg.physics_filename

    return inp


def run_celer_sim(inp: dict, run_name: str) -> dict:
    """Write the input file, run celer-sim, and return the parsed JSON output."""
    exe = environ.get("CELERITAS_EXE", "./celer-sim")
    inp_file = Path(f"{run_name}.inp.json")
    print("Running", exe, str(inp_file), "from", Path.cwd(), file=stderr)
    inp_file.write_text(json.dumps(inp, indent=1))

    result = subprocess.run(
        [exe, "-"], input=json.dumps(inp).encode(), stdout=subprocess.PIPE
    )

    if result.returncode:
        print("fatal: run failed with error", result.returncode)
        try:
            j = json.loads(result.stdout.decode())
        except Exception:
            pass
        else:
            out_file = Path(f"{run_name}.out.failed.json")
            out_file.write_text(json.dumps(j, indent=1))
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

    out_file = Path(f"{run_name}.out.json")
    out_file.write_text(json.dumps(j, indent=1))
    print("Results written to", out_file, file=stderr)
    return j


def validate_output(j: dict, inp: dict, use_device: bool) -> None:
    run_output = j["result"]["runner"]
    time = run_output["time"].copy()
    steps = time.pop("steps")
    if use_device:
        assert steps
        assert len(steps[0]) == run_output["num_step_iterations"][0], steps[0]
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
        expected_core_sizes = {
            "events": 4,
            "initializers": 3200,
            "processes": 1,
            "secondaries": 96,
            "tracks": 32,
        }
    if not use_device and "lar" in inp["geometry_file"]:
        expected_opt_sizes = {"generators": 8388608, "tracks": 8192}
        assert "optical" in run_output
        cuts = sum(em_step["num_cut"] for em_step in run_output["optical"])
        assert cuts > 0

    if expected_core_sizes:
        assert core_sizes == expected_core_sizes, core_sizes
    if expected_opt_sizes:
        opt_sizes = internal["optical-sizes"].copy()
        assert num_streams == opt_sizes.pop("streams")
        assert opt_sizes == expected_opt_sizes, opt_sizes


def main():
    args = parse_args()

    # We reuse the "disable device" environment variable, which prevents the GPU
    # from being initialized at runtime if non-empty.
    use_device = not environ.get("CELER_DISABLE_DEVICE")
    core_geo = environ.get("CELER_CORE_GEO", "ORANGE").lower()

    run_name = Path(args.geometry_file).stem + ("-gpu" if use_device else "-cpu")
    physics_options = get_em_physics_options(core_geo)
    physics_filename = run_geant_export(args.geometry_file, run_name, physics_options)
    cfg = RunConfig(
        geometry_file=args.geometry_file,
        event_file=args.event_file,
        rootout_file=args.rootout_file,
        run_name=run_name,
        use_device=use_device,
        physics_options=physics_options,
        physics_filename=physics_filename,
    )
    inp = build_input(cfg)
    j = run_celer_sim(inp, run_name)
    validate_output(j, inp, use_device)


if __name__ == "__main__":
    main()
