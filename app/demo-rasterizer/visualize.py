#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def read_image(input):
    assert input['metadata']['int_size'] == 4
    image = np.fromfile(input['data'], dtype=np.int32)
    return np.reshape(image, input['metadata']['dims'])

def main():
    try:
        (json_input, imgname) = sys.argv[1:]
    except ValueError:
        print("usage: {} [output.json] [image.png]".format(sys.argv[0]))
        sys.exit(1)
    if json_input != '-':
        with open(json_input, 'r') as f:
            input = json.load(f)
    else:
        input = json.load(sys.stdin)

    image = read_image(input)

    (fig, ax) = plt.subplots()
    ax.imshow(image)
    fig.savefig(imgname)

if __name__ == '__main__':
    main()
