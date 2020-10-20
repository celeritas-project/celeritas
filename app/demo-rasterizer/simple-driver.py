#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
"""
import json
from pprint import pprint
import subprocess
from os import environ
from sys import exit

inp = {
    'image': {
        'lower_left': [-10, -10, 0],
        'upper_right': [10, 10, 0],
        'rightward_ax': [1, 0, 0],
        'vertical_pixels': 32
    },
    'input': '../../test/geometry/data/twoBoxes.gdml',
    'output': 'two-boxes.bin'
}

print("Input:")
pprint(inp)

exe = environ.get('CELERITAS_DEMO_EXE', './demo-rasterizer')
print("Running", exe)
with subprocess.Popen([exe, '-'],
                      stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE) as proc:
    (output, _) = proc.communicate(input=json.dumps(inp).encode())

print("Received {} bytes of data".format(len(output)))
out_text = output.decode()
try:
    result = json.loads(out_text)
except json.decoder.JSONDecodeError as e:
    print("error: expected a JSON object but got the following stdout:")
    print(out_text)
    print("fatal:", str(e))
    exit(1)
pprint(result)
