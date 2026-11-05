#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Read-only probe of a Celeritas installation and its runtime environment.

Usage: celer-doctor.py [PREFIX]

Print ``key=value`` facts, then ``warn=``/``note=`` lines for likely
problems, then the raw ``celer-config`` JSON. The probe never modifies
anything and always exits with status 0.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TIMEOUT = 60  # seconds allowed for each probed command

# Environment variables that change Celeritas or Geant4 behavior
ENV_RE = re.compile(
    r"^(CELER_\w+|ORANGE_\w+|G4FORCE\w+|CUDA_(STACK|HEAP)_SIZE|"
    r"(CUDA|HIP)_VISIBLE_DEVICES|VECGEOM_VERBOSE|HEPMC3_VERBOSE|"
    r"G4ORG_OPTIONS|G4VG_COMPARE_VOLUMES|G4LEDATA|G4DATADIR|OMP_NUM_THREADS)$"
)

# Variables that indicate the process was started by an MPI launcher
MPI_LAUNCHER_VARS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_RANK",
    "MPI_LOCALNRANKS",
)


def run(cmd, env=None):
    """Run a command and return its stdout, or None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=TIMEOUT,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


class Report:
    """Accumulate facts, warnings, and notes."""

    def __init__(self):
        self.lines = []

    def fact(self, key, value):
        if value is None or value == "":
            return
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        self.lines.append(f"{key}={value}")

    def warn(self, msg):
        self.lines.append(f"warn={msg}")

    def note(self, msg):
        self.lines.append(f"note={msg}")

    def print(self):
        for line in self.lines:
            print(line)


def candidate_prefixes(argv):
    """Yield (source, prefix) pairs in priority order."""
    if len(argv) > 1:
        yield ("argument", Path(argv[1]))
    for var in ("Celeritas_ROOT", "CELERITAS_ROOT", "CELER_INSTALL_DIR"):
        if os.environ.get(var):
            yield (var, Path(os.environ[var]))
    if os.environ.get("Celeritas_DIR"):
        # <prefix>/lib/cmake/Celeritas
        yield ("Celeritas_DIR", Path(os.environ["Celeritas_DIR"]).parents[2])
    exe = shutil.which("celer-config")
    if exe:
        yield ("PATH", Path(exe).resolve().parent.parent)
    for entry in os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep):
        if entry:
            yield ("CMAKE_PREFIX_PATH", Path(entry))


def spack_prefix():
    """Ask Spack for an installed Celeritas (slow, so used last)."""
    if not shutil.which("spack"):
        return None
    out = run(["spack", "location", "-i", "celeritas"])
    if not out:
        return None
    return Path(out.strip().splitlines()[-1])


def find_install(argv):
    """Return (source, prefix, celer-config path) or None."""
    for source, prefix in candidate_prefixes(argv):
        exe = prefix / "bin" / "celer-config"
        if exe.is_file():
            return (source, prefix, exe)
    prefix = spack_prefix()
    if prefix:
        exe = prefix / "bin" / "celer-config"
        if exe.is_file():
            return ("spack", prefix, exe)
    return None


def find_cmake_config(prefix):
    for path in sorted(prefix.glob("lib*/cmake/Celeritas/CeleritasConfig.cmake")):
        return path
    return None


def query_celer_config(exe):
    """Run celer-config (with the device if possible) and parse its JSON."""
    env = dict(os.environ)
    # Avoid MPI initialization and colored log output while probing
    env["CELER_DISABLE_PARALLEL"] = "1"
    env["CELER_COLOR"] = "0"
    for args in (["--device"], []):
        out = run([str(exe)] + args + ["--indent", "-1"], env=env)
        if out:
            try:
                return json.loads(out)
            except ValueError:
                continue
    return None


def version_tuple(text):
    nums = re.findall(r"\d+", text or "")
    return tuple(int(n) for n in nums[:3])


def check_geant4(report, cfg):
    """Report the Geant4 version in the environment and its EM data."""
    g4_version = None
    out = run(["geant4-config", "--version"]) if shutil.which("geant4-config") else None
    if out:
        g4_version = out.strip()
    report.fact("geant4.version", g4_version)

    built_with = (cfg.get("versions") or {}).get("Geant4") if cfg else None
    report.fact("celeritas.geant4_built_with", built_with)
    if (
        built_with
        and g4_version
        and version_tuple(built_with) != version_tuple(g4_version)
    ):
        report.warn(
            f"Celeritas was built against Geant4 {built_with} but geant4-config "
            f"reports {g4_version}: find_package(Celeritas) requires the same "
            "Geant4, and mixing versions can fail at link or run time"
        )

    effective = version_tuple(g4_version or built_with)
    if effective:
        if effective < (11, 0):
            report.note(
                "Geant4 < 11.0: use UserActionIntegration; tracking manager "
                "offload requires 11.0 and fast simulation offload requires 11.1"
            )
        elif effective < (11, 1):
            report.note("Geant4 11.0: fast simulation offload requires 11.1")

    g4ledata = os.environ.get("G4LEDATA")
    if g4ledata:
        if not Path(g4ledata).is_dir():
            report.warn(f"G4LEDATA={g4ledata} is not a directory")
    elif cfg and "geant4" in (cfg.get("use") or []):
        datasets = run(["geant4-config", "--datasets"]) if g4_version else None
        hint = ""
        for line in (datasets or "").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "G4LEDATA":
                hint = f" (geant4-config suggests {parts[2]})"
        report.warn(
            "G4LEDATA is not set: Celeritas reads Livermore and relaxation "
            f"data through it{hint}"
        )


def detected_gpus():
    """Return a list of GPU description strings from vendor tools."""
    gpus = []
    if shutil.which("nvidia-smi"):
        out = run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total",
                "--format=csv,noheader",
            ]
        )
        gpus.extend(line.strip() for line in (out or "").splitlines() if line.strip())
    if not gpus and shutil.which("rocminfo"):
        out = run(["rocminfo"])
        names = sorted(set(re.findall(r"\bgfx[0-9a-f]+\b", out or "")))
        gpus.extend(names)
    return gpus


def check_device(report, cfg, device):
    use = cfg.get("use") or []
    gpu_api = "cuda" if "cuda" in use else "hip" if "hip" in use else None
    arch = cfg.get("gpu_architectures") or ""
    report.fact("celeritas.gpu_architectures", arch)

    gpus = detected_gpus()
    for i, desc in enumerate(gpus):
        report.fact(f"gpu.{i}", desc)

    if device:
        report.fact("device.name", device.get("name"))
        report.fact("device.capability", device.get("capability"))
        report.fact("device.num_devices", device.get("num_devices"))
        cap = str(device.get("capability", ""))
        if cap and arch and arch != "native" and cap not in arch:
            report.warn(
                f"device compute capability {cap} is not in the compiled GPU "
                f"architectures '{arch}': kernels may fail at run time; "
                "rebuild Celeritas for this GPU"
            )

    if not gpu_api:
        if gpus:
            report.note("GPUs are present but Celeritas was built without CUDA or HIP")
        return
    if os.environ.get("CELER_DISABLE_DEVICE"):
        report.note("CELER_DISABLE_DEVICE is set: Celeritas will run on CPU")
    elif not device:
        report.warn(
            f"Celeritas was built with {gpu_api.upper()} but celer-config could "
            "not activate a GPU (check drivers, CUDA_VISIBLE_DEVICES or "
            "HIP_VISIBLE_DEVICES, and that this node has a GPU)"
        )


def check_config(report, cfg):
    use = cfg.get("use") or []
    if "geant4" not in use:
        report.warn(
            "Celeritas was built without Geant4: Celeritas::G4 and Geant4 "
            "offload are unavailable (standalone apps still work with "
            "exported physics data)"
        )
    debug = cfg.get("debug")
    if debug:
        report.note(
            "debug assertions are enabled: expect much slower execution; "
            "do not use this build for performance measurements"
        )
    if "mpi" in use and not os.environ.get("CELER_DISABLE_PARALLEL"):
        if not any(os.environ.get(v) for v in MPI_LAUNCHER_VARS):
            report.note(
                "Celeritas was built with MPI: if the application does not "
                "run under an MPI launcher, export CELER_DISABLE_PARALLEL=1"
            )
    if cfg.get("core_geo", "").lower() == "vecgeom" and ("cuda" in use):
        report.note(
            "VecGeom on GPU may need a larger CUDA stack: see "
            "SetupOptions::cuda_stack_size or CUDA_STACK_SIZE"
        )


def check_environment(report):
    env = {k: v for k, v in os.environ.items() if ENV_RE.match(k)}
    for key in sorted(env):
        report.fact(f"env.{key}", env[key] or "(empty)")
    if env.get("CELER_DISABLE") and env.get("CELER_KILL_OFFLOAD"):
        report.warn(
            "both CELER_DISABLE and CELER_KILL_OFFLOAD are set: Celeritas "
            "setup will fail; unset one"
        )
    elif env.get("CELER_DISABLE"):
        report.note("CELER_DISABLE is set: Celeritas offload is turned off")
    elif env.get("CELER_KILL_OFFLOAD"):
        report.note(
            "CELER_KILL_OFFLOAD is set: offloadable tracks are killed, not "
            "transported (only for estimating maximum speedup)"
        )


def main(argv):
    report = Report()
    found = find_install(argv)
    raw = None
    if not found:
        report.fact("celeritas.found", "no")
        report.warn(
            "no Celeritas installation found: looked at the argument, "
            "Celeritas_ROOT, CELERITAS_ROOT, CELER_INSTALL_DIR, Celeritas_DIR, "
            "celer-config on PATH, CMAKE_PREFIX_PATH, and 'spack location -i "
            "celeritas'; load the environment (e.g. 'spack load celeritas') "
            "or pass the install prefix"
        )
    else:
        source, prefix, exe = found
        report.fact("celeritas.found", "yes")
        report.fact("celeritas.found_by", source)
        report.fact("celeritas.prefix", prefix)
        report.fact("celeritas.cmake_config", find_cmake_config(prefix))
        raw = query_celer_config(exe)
        if raw is None:
            report.warn(
                f"{exe} failed to run: shared libraries may be missing from "
                "the environment (activate the Spack environment or module "
                "that provides Celeritas and its dependencies)"
            )

    cfg = (raw or {}).get("config") or {}
    if raw:
        report.fact("celeritas.version", raw.get("version"))
        for key in (
            "use",
            "build_type",
            "debug",
            "core_geo",
            "core_rng",
            "real_type",
            "units",
            "openmp",
            "reseed",
        ):
            report.fact(f"celeritas.{key}", cfg.get(key))
        versions = cfg.get("versions") or {}
        for name in sorted(versions):
            report.fact(f"celeritas.dep.{name}", versions[name])
        check_config(report, cfg)
        check_device(report, cfg, raw.get("device"))

    check_geant4(report, cfg)
    check_environment(report)

    report.print()
    if raw:
        print("--- celer-config JSON ---")
        print(json.dumps(raw, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        main(sys.argv)
    except Exception as e:  # Never fail: report and exit cleanly
        print(f"warn=probe error: {e!r}")
    sys.exit(0)
