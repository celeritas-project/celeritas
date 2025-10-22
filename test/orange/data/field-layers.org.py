# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
ORANGE input definition using python.

Note that this file cannot be run standalone: it is merely an input to
the ``orange2celeritas`` executable. The ``db`` variable is passed into this
script as a global, then it's validated and written as an ORANGE XML input
file. Finally, the ORANGE geometry is built from the XML and writes out the
JSON file field-test.org.json.

The geometry looks like this (with 5 boxes, shortened for clarity)::

        +--------------------+
        | world              |
        |  +--------------+  |
        |  |  layerbox4   |  |
        |  +--------------+  |
        |                    |
    ^   |  +--------------+  |
    |   |  |  layerbox3   |  |
    y   |  +--------------+  |
    |   |   ........         |
        +--------------------+
"""

layer_box = {
    "_type": "box",
    "widths": [18, 1, 18],
}

shapes = []
cells = []
for i, ymid in enumerate([-4, -2, 0, 2, 4]):
    box = dict(layer_box)
    box["translate"] = [0, ymid, 0]
    box["name"] = name = f"layerbox{i}"
    shapes.append(box)
    cells.append(
        {
            "name": f"layer{i}",
            "comp": "1",
            "shapes": [name],
        }
    )

shapes.append(
    {
        "_type": "box",
        "widths": [20, 40, 20],
        "name": "worldbox",
    }
)

world_univ = {
    "_type": "unit",
    "name": "world",
    "shape": shapes,
    "cell": cells,
    "interior": "worldbox",
    "background": "0",  # Fill unassigned space with matid 0
}

db.update(
    {
        "geometry": {
            "global": "world",
        },
        "universe": [world_univ],
    }
)
