#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Summarize a Celeritas diagnostic output file such as celeritas.out.json.

Usage: summarize-output.py FILE [--top N]

The file is written by Celeritas applications (celer-sim, celer-g4, ...) and
by Geant4 integrations (SetupOptions::output_file). It is a JSON object, or
JSON lines when output is appended, with top-level categories ``input``,
``result``, ``system``, and ``internal``. This script prints the build
configuration, device, environment variables Celeritas consulted, any
recorded exception, timing, and kernel diagnostics. It only reads the file.
"""

import argparse
import json
import sys

CATEGORIES = ("input", "result", "system", "internal")


def deep_merge(dst, src):
    """Recursively merge ``src`` into ``dst`` (later records win)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def load(path):
    """Load a JSON or JSON-lines file; return (merged dict, record count)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
        return (data, 1)
    except ValueError:
        pass
    merged = {}
    count = 0
    bad_lines = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            bad_lines.append(lineno)
            continue
        if isinstance(record, dict):
            deep_merge(merged, record)
            count += 1
    if bad_lines:
        shown = ", ".join(str(n) for n in bad_lines[:5])
        more = "..." if len(bad_lines) > 5 else ""
        print(f"warning: skipped {len(bad_lines)} unparsable line(s): {shown}{more}")
    return (merged, count)


def section(title):
    print(f"\n== {title} ==")


def fmt_value(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "off"
    return str(value)


def fmt_seconds(value):
    try:
        return f"{float(value):.3f} s"
    except (TypeError, ValueError):
        return str(value)


def summarize_build(build):
    section("Build")
    if not build:
        print("(no system.build entry)")
        return
    cfg = build.get("config", {})
    print(f"version: {build.get('version')}")
    for key in (
        "build_type",
        "debug",
        "core_geo",
        "core_rng",
        "real_type",
        "units",
        "openmp",
        "gpu_architectures",
    ):
        if key in cfg:
            print(f"{key}: {fmt_value(cfg[key])}")
    if cfg.get("use"):
        print("use: " + ", ".join(cfg["use"]))
    versions = cfg.get("versions") or {}
    if versions:
        print(
            "dependencies: "
            + ", ".join(f"{k} {v}" for k, v in sorted(versions.items()))
        )


def summarize_device(device):
    section("Device")
    if not device:
        print("(no GPU in use: CPU execution)")
        return
    for key in (
        "name",
        "platform",
        "capability",
        "num_devices",
        "total_global_mem",
        "debug",
    ):
        if key in device:
            print(f"{key}: {device[key]}")


def summarize_environ(environ):
    section("Environment variables consulted")
    if not environ:
        print("(none recorded)")
        return
    set_vars = {k: v for k, v in environ.items() if v not in ("", None)}
    unset = sorted(k for k in environ if k not in set_vars)
    for key in sorted(set_vars):
        print(f"{key}={set_vars[key]}")
    if unset:
        print("unset: " + ", ".join(unset))


def print_exception(exc, indent=""):
    if not isinstance(exc, dict):
        print(f"{indent}{exc}")
        return
    kind = exc.get("type", "?")
    which = exc.get("which")
    what = exc.get("what") or exc.get("condition")
    where = ""
    if exc.get("file"):
        where = f" at {exc['file']}:{exc.get('line', '?')}"
    label = f"{kind} ({which})" if which else kind
    print(f"{indent}{label}{where}" + (f": {what}" if what else ""))
    extra = {
        k: v
        for k, v in exc.items()
        if k not in ("type", "which", "what", "condition", "file", "line", "context")
    }
    for key, value in extra.items():
        print(f"{indent}  {key}: {json.dumps(value)[:300]}")
    if "context" in exc:
        print(f"{indent}  caused while:")
        print_exception(exc["context"], indent + "    ")


def summarize_exception(exc):
    section("Exception")
    if not exc:
        print("(none)")
        return
    for item in exc if isinstance(exc, list) else [exc]:
        print_exception(item)


def sum_action_times(actions):
    """Combine per-thread action time maps into one map."""
    if isinstance(actions, dict):
        return dict(actions)
    total = {}
    for per_thread in actions or []:
        for name, seconds in (per_thread or {}).items():
            total[name] = total.get(name, 0.0) + float(seconds)
    return total


def print_top_actions(actions, top):
    totals = sum_action_times(actions)
    if not totals:
        print(
            "action times: not recorded (enable SetupOptions::action_times "
            "or the app's action timer option to measure them)"
        )
        return
    grand = sum(totals.values()) or 1.0
    print(f"top {min(top, len(totals))} actions by time:")
    for name, seconds in sorted(totals.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {seconds:10.3f} s  {100 * seconds / grand:5.1f}%  {name}")


def summarize_integration_time(time, top):
    """Timing from Geant4 integration (accel TimeOutput)."""
    section("Timing (Geant4 integration)")
    print(f"setup: {fmt_seconds(time.get('setup'))}")
    print(f"total: {fmt_seconds(time.get('total'))}")
    events = time.get("events") or []
    if events:
        per_thread = [sum(float(t) for t in thread or []) for thread in events]
        num_events = sum(len(thread or []) for thread in events)
        print(
            f"events: {num_events} over {len(events)} thread(s); per-thread event time: "
            + ", ".join(fmt_seconds(t) for t in per_thread)
        )
    print_top_actions(time.get("actions"), top)


def summarize_runner(runner, top):
    """Results from celer-sim."""
    section("celer-sim results")
    for key in ("num_streams",):
        if key in runner:
            print(f"{key}: {runner[key]}")
    totals = {}
    for key in (
        "num_tracks",
        "num_steps",
        "num_step_iterations",
        "num_aborted",
        "num_errored",
        "num_cut",
    ):
        values = runner.get(key)
        if isinstance(values, list) and values:
            try:
                totals[key] = sum(values)
            except TypeError:
                print(f"{key}: {values}")
                continue
            print(f"{key}: total {totals[key]} over {len(values)} event(s)")
    if totals.get("num_aborted") or totals.get("num_errored"):
        print(
            "warning: some tracks were aborted or errored: see the log for killed-track messages"
        )
    time = runner.get("time") or {}
    for key in ("setup", "warmup", "total"):
        if key in time:
            print(f"time.{key}: {fmt_seconds(time[key])}")
    print_top_actions(time.get("actions"), top)


def summarize_kernels(kernels, top):
    if not kernels:
        return
    section("Kernels")
    print(f"{len(kernels)} kernels registered")
    launched = [k for k in kernels if "num_launches" in k]
    if launched:
        print(f"top {min(top, len(launched))} kernels by launches:")
        for k in sorted(launched, key=lambda k: -k["num_launches"])[:top]:
            print(f"  {k['num_launches']:10d}  {k.get('name')}")
    with_occ = [k for k in kernels if isinstance(k.get("occupancy"), (int, float))]
    if with_occ:
        print("lowest occupancy:")
        for k in sorted(with_occ, key=lambda k: k["occupancy"])[: min(top, 5)]:
            print(f"  {k['occupancy']:.2f}  regs={k.get('num_regs')}  {k.get('name')}")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", help="Celeritas JSON or JSON-lines output")
    parser.add_argument("--top", type=int, default=10, help="entries to show per table")
    args = parser.parse_args(argv)

    try:
        data, num_records = load(args.file)
    except OSError as e:
        print(f"error: cannot read {args.file}: {e}")
        return 1
    if not isinstance(data, dict) or not any(c in data for c in CATEGORIES):
        print(
            f"error: {args.file} does not look like Celeritas output (no {'/'.join(CATEGORIES)} keys)"
        )
        return 1

    print(f"file: {args.file} ({num_records} record(s))")
    system = data.get("system", {})
    result = data.get("result", {})

    summarize_build(system.get("build"))
    summarize_device(system.get("device"))
    summarize_environ(system.get("environ"))
    summarize_exception(result.get("exception"))
    if isinstance(result.get("time"), dict):
        summarize_integration_time(result["time"], args.top)
    if isinstance(result.get("runner"), dict):
        summarize_runner(result["runner"], args.top)
    summarize_kernels(system.get("kernels"), args.top)

    section("Other entries")
    for cat in CATEGORIES:
        entries = data.get(cat)
        if isinstance(entries, dict):
            shown = {
                "build",
                "device",
                "environ",
                "kernels",
                "exception",
                "time",
                "runner",
            }
            others = sorted(k for k in entries if k not in shown)
            if others:
                print(f"{cat}: " + ", ".join(others))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
