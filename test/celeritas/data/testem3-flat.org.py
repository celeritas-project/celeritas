# -*- coding: utf-8 -*-
# Copyright 2022 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
ORANGE testem3 input definition using python.

Note that this file cannot be run standalone: it is merely an input to
the ``orange2celeritas`` executable. The ``db`` variable is passed into this
script as a global, then it's validated and written as an ORANGE XML input
file. Finally, the ORANGE geometry is built from the XML and writes out the
JSON file field-test.org.json.
"""

comps = "Pb lAr vacuum".split()
gap_width = 0.23
absorber_width = 0.57
radius = 20. # off-axis halfwidth
num_layers = 50
layer_xwidth = gap_width + absorber_width
total_xwidth = num_layers * layer_xwidth
xstart = -total_xwidth / 2

box_base = {
    '_type': 'box',
    'ymin': -radius,
    'ymax': radius,
    'zmin': -radius,
    'zmax': radius,
}

world_lvs = []
shapes = [
    {
        '_type': "box",
        'faces': [xstart, xstart + total_xwidth,
                  -radius, radius,
                  -radius, radius],
        'name': "calorimeter_lv",
    },
    {
        '_type': "box",
        'widths': [48, 48, 48],
        'name': "world_lv",
    }
]
cells = [{
    'name': "world_lv",
    'comp': "vacuum",
    'shapes': ['world_lv', '~calorimeter_lv']
}]

for i in range(num_layers):
    left = xstart + layer_xwidth * i
    mid = left + gap_width
    right = mid + absorber_width

    for (label, mat, xmin, xmax) in [
            ("gap", "lAr", left, mid),
            ("absorber", "Pb", mid, right)
            ]:
        shape_name = f"{label}_lv_{i}"
        box = {
            'name': shape_name,
            'xmin': xmin,
            'xmax': xmax,
        }
        box.update(box_base)
        shapes.append(box)
        cells.append({
            'name': f"{label}_lv_{i}",
            'comp': mat,
            'shapes': [shape_name],
        })

world_univ = {
    '_type': 'unit',
    'name': "world",
    'shape': shapes,
    'cell': cells,
    'interior': "world_lv",
}

db.update({
    'geometry': {
        'global': "world",
        'comp': comps,
        'matid': list(range(len(comps))),
    },
    'universe': [world_univ]
})
