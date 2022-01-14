#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
ORANGE input definition using python.
"""

layer_box = {
    '_type': 'box',
    'widths': [18, 1, 18],
}

world_shapes = []
shapes = []
cells = []
for (i, ymid) in enumerate([-4, -2, 0, 2, 4]):
    box = dict(layer_box)
    box['translate'] = [0, ymid, 0]
    box['name'] = name = f"layerbox{i}"
    shapes.append(box)
    world_shapes.append("~" + name)
    cells.append({
        'name': f"layer{i}",
        'comp': "1",
        'shapes': [name],
    })

shapes.append({
    '_type': "box",
    'widths': [20, 40, 20],
    'name': "worldbox",
})
cells.append({
    'name': "world",
    'comp': "0",
    'shapes': ['worldbox'] + world_shapes,
})

world_univ = {
    '_type': 'unit',
    'name': "world",
    'shape': shapes,
    'cell': cells,
    'interior': "worldbox",
}

db.update({
    'geometry': {
        'global': "world",
    },
    'universe': [world_univ]
})
