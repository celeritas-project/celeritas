#!/usr/bin/env python3
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Summarize the field propagation diagnostic from Celeritas JSON output.

The diagnostic is enabled with ``problem.diagnostics.field`` in a celer-sim
input, with ``SetupOptions::field_diagnostic`` in accel, or with the
``CELER_FIELD_DIAGNOSTIC=1`` environment variable (add
``CELER_FIELD_DIAGNOSTIC_ORACLE=1`` to also evaluate the safety distance).

Example::

    field-diagnostic.py celeritas-output.json --top 20
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Columns derived from each row of the energy and volume tallies
COLUMNS = [
    ("steps", lambda t: t["steps"]),
    ("sub/step", lambda t: _div(t["substeps"], t["steps"])),
    ("int/step", lambda t: _div(t["integrations"], t["steps"])),
    ("int/sub", lambda t: _div(t["integrations"], t["substeps"])),
    (
        "first%",
        lambda t: _pct(t["first_substep_integrations"], t["integrations"]),
    ),
    ("isect/step", lambda t: _div(t["intersects"], t["steps"])),
    ("short/bnd", lambda t: _div(t["shortened"], t["boundary"])),
    ("bnd%", lambda t: _pct(t["boundary"], t["steps"])),
    ("loop%", lambda t: _pct(t["looping"], t["steps"])),
    (
        "skip%",
        lambda t: _pct(
            t["skippable_intersects"], t["intersects"] - t["intersect_hits"]
        ),
    ),
]


def _div(num, denom):
    return num / denom if denom else math.nan


def _pct(num, denom):
    return 100 * _div(num, denom)


def find_diagnostic(j):
    """Find the field diagnostic in a Celeritas output file."""
    if "field-diagnostic" in j.get("result", {}):
        return j["result"]["field-diagnostic"]
    if j.get("_label") == "field-diagnostic":
        return j
    sys.exit("error: no 'field-diagnostic' result in the output")


def rows_from_tally(tally):
    """Convert a {quantity: [values]} tally to a list of {quantity: value}."""
    keys = list(tally)
    size = len(tally[keys[0]])
    return [{k: tally[k][i] for k in keys} for i in range(size)]


def sum_rows(rows):
    return {k: sum(r[k] for r in rows) for k in rows[0]}


def format_value(value):
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.3g}"
    return str(value)


def print_table(title, labels, rows, total_integrations):
    """Print derived columns for each row, plus its share of integrations."""
    headers = ["", *(name for name, _ in COLUMNS), "int%"]
    table = []
    for label, row in zip(labels, rows):
        values = [func(row) for _, func in COLUMNS]
        values.append(_pct(row["integrations"], total_integrations))
        table.append([label, *(format_value(v) for v in values)])

    widths = [max(len(r[i]) for r in [headers, *table]) for i in range(len(headers))]
    print(f"\n{title}")
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    for r in table:
        print("  ".join(v.rjust(w) for v, w in zip(r, widths)))


def hist_stats(counts, num_bins):
    """Return (n, mean, p50, p90, p99, max, overflow, tail share) of a histogram.

    The tail share is the fraction of the summed quantity contributed by the
    10% of entries with the largest values. Overflow entries are counted at the
    overflow bin value, so the mean and tail share are lower bounds if there is
    any overflow.
    """
    n = sum(counts)
    if not n:
        return None
    total = sum(i * c for i, c in enumerate(counts))

    def percentile(frac):
        target = frac * n
        acc = 0
        for i, c in enumerate(counts):
            acc += c
            if acc >= target:
                return i
        return len(counts) - 1

    # Sum of the quantity in the top 10% of entries
    remaining = 0.1 * n
    tail = 0.0
    for i in reversed(range(len(counts))):
        take = min(counts[i], remaining)
        tail += take * i
        remaining -= take
        if remaining <= 0:
            break

    overflow = counts[num_bins - 1] if len(counts) >= num_bins else 0
    max_value = max(i for i, c in enumerate(counts) if c)
    return (
        n,
        total / n,
        percentile(0.5),
        percentile(0.9),
        percentile(0.99),
        max_value,
        overflow,
        _div(tail, total),
    )


def print_histograms(diag, kind):
    num_bins = diag["num_bins"]
    print(f"\nPer-{kind} distributions (overflow at {num_bins - 1})")
    headers = [
        "quantity",
        "particle",
        "n",
        "mean",
        "p50",
        "p90",
        "p99",
        "max",
        "overflow",
        "top10%share",
    ]
    print("  ".join(f"{h:>12}" for h in headers))
    for quantity, per_particle in diag[kind].items():
        for name, counts in zip(diag["particles"], per_particle):
            stats = hist_stats(counts, num_bins)
            if stats is None:
                continue
            values = [quantity, name, *(format_value(v) for v in stats)]
            print("  ".join(f"{v:>12}" for v in values))


def print_safety(diag):
    safety = diag.get("safety")
    if not safety:
        return
    log_min = safety["log_min"]
    per_decade = safety["bins_per_decade"]

    def upper_edge(i):
        # Bin 0 is underflow; bin i covers [log_min + (i-1)/n, log_min + i/n)
        return 10 ** (log_min + i / per_decade)

    print("\nSafety at start of step / step length (oracle)")
    thresholds = [0.01, 0.1, 0.5, 1.0]
    print(
        "  ".join(
            f"{h:>12}" for h in ["particle", "n", *(f"<{t:g}" for t in thresholds)]
        )
    )
    for name, counts in zip(diag["particles"], safety["counts"]):
        n = sum(counts)
        if not n:
            continue
        fracs = []
        for t in thresholds:
            below = sum(c for i, c in enumerate(counts) if upper_edge(i) <= t)
            fracs.append(f"{_pct(below, n):.1f}%")
        print("  ".join(f"{v:>12}" for v in [name, str(n), *fracs]))


def print_step_iters(diag, top):
    for stream, series in enumerate(diag.get("step_iter", [])):
        tracks = series["tracks"]
        if not tracks:
            continue
        integ = series["integrations"]
        max_integ = series["max_integrations"]
        print(
            f"\nStream {stream}: {len(tracks)} step iterations recorded"
            + (f" (of {series['_truncated']})" if "_truncated" in series else "")
        )
        ratio = [_div(m * t, i) for m, t, i in zip(max_integ, tracks, integ)]
        finite = [r for r in ratio if not math.isnan(r)]
        if finite:
            print(
                f"  max/mean integrations per launch: median "
                f"{sorted(finite)[len(finite) // 2]:.3g}, "
                f"max {max(finite):.3g}"
            )
        order = sorted(range(len(tracks)), key=lambda i: -max_integ[i])[:top]
        print(f"  Iterations with the largest per-track integrations:")
        print(
            "  "
            + "  ".join(
                f"{h:>12}"
                for h in ["iteration", "tracks", "integ", "max_integ", "max_substeps"]
            )
        )
        for i in order:
            print(
                "  "
                + "  ".join(
                    f"{v:>12}"
                    for v in [
                        i,
                        tracks[i],
                        integ[i],
                        max_integ[i],
                        series["max_substeps"][i],
                    ]
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="Celeritas JSON output")
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="number of volumes and step iterations to show",
    )
    args = parser.parse_args()

    with open(args.output) as f:
        diag = find_diagnostic(json.load(f))

    energy = diag["energy"]
    energy_rows = rows_from_tally(energy["tally"])
    total = sum_rows(energy_rows)
    print_table("Total", ["all"], [total], total["integrations"])

    # Energy bins with any steps
    log_min = energy["log_min"]
    per_decade = energy["bins_per_decade"]

    def energy_label(i):
        if i == 0:
            return f"<{10**log_min:.3g}"
        if i == len(energy_rows) - 1:
            return f">={10 ** (log_min + (i - 1) / per_decade):.3g}"
        return f"{10 ** (log_min + (i - 1) / per_decade):.3g}"

    labels, rows = zip(
        *[(energy_label(i), r) for i, r in enumerate(energy_rows) if r["steps"]]
    )
    print_table(
        "By kinetic energy (lower edge, MeV)", labels, rows, total["integrations"]
    )

    if "volume" in diag:
        vol_rows = rows_from_tally(diag["volume"]["tally"])
        order = sorted(
            range(len(vol_rows)), key=lambda i: -vol_rows[i]["integrations"]
        )[: args.top]
        print_table(
            f"Top {len(order)} volumes by integrations",
            [diag["volume"]["label"][i] for i in order],
            [vol_rows[i] for i in order],
            total["integrations"],
        )

    print_histograms(diag, "step")
    print_histograms(diag, "track")
    print_safety(diag)
    print_step_iters(diag, args.top)


if __name__ == "__main__":
    main()
