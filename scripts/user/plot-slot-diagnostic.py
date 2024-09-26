#!/usr/bin/env python3
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Render a plot for step diagnostics.

TODO: move to celerpy
"""
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from matplotlib.colors import ListedColormap, NoNorm

PARTICLE_COLORS = ListedColormap([
    [1, 0, 0, 0.5], # error
    [0, 0, 0, 0], # inactive
    [0, 0.5, 1.0, 1], # gamma
    [0, 0.5, 0, 1], # e-
    [0.7, 0., 0, 1], # e+
],
    name='particle_colors')

StepDiagnostic = namedtuple('StepDiagnostic', ['steps', 'metadata'])


def load_steps(basename):
    with open(basename + 'metadata.json') as f:
        metadata = json.load(f)

    steps = []
    with open(basename + '0.jsonl') as f:
        for line in f:
            steps.append(json.loads(line))
    return StepDiagnostic(np.array(steps), metadata)


def plot_steps(ax, steps, metadata):
    assert metadata['_label'] == 'particle'
    steps = np.asarray(steps)
    labels = metadata['label']
    assert np.issubdtype(steps.dtype, np.integer)
    assert np.amax(steps) < len(labels)
    assert np.amin(steps) >= -2
    if labels != ["gamma", "e-", "e+"]:
        raise NotImplementedError(
            "This class must be updated to support other particle types")

    steps = 2 + steps
    labels = ["error", "inactive"] + labels

    fig = ax.get_figure()
    ax.set_xlabel("Track slot")
    ax.set_ylabel("Step iteration")
    im = ax.imshow(steps, cmap=PARTICLE_COLORS, norm=NoNorm(), resample=False)
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    cbar = fig.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
    cbar.set_ticks(np.arange(len(PARTICLE_COLORS.colors)),
                   labels=labels,
                   fontsize='x-small')

    return {
        'im': im,
        'cbar': cbar,
    }

def run(basename, outname):
    stepdiag = load_steps(basename)
    (fig, ax) = plt.subplots(figsize=(6,4))
    plot_steps(ax, stepdiag.steps, stepdiag.metadata)
    kwargs = (dict(transparent=True) if outname.endswith('.pdf')
              else dict(dpi=300))
    fig.savefig(outname, bbox_inches='tight', **kwargs)

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prefix",
        help="Slot diagnostic output prefix")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output filename (empty for prefix + plot.png)")
    args = parser.parse_args()

    outfile = args.output or args.prefix + "plot.png"

    run(args.prefix, outfile)

if __name__ == "__main__":
    main()
