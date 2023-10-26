# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Add LLDB wrappers for Celeritas types.

To use::
    (lldb) command script import celeritas/scripts/dev/celerlldb.py --allow-reload
    (lldb) type synthetic add -x "^celeritas::Span<.+>$" --python-class celerlldb.SpanSynthetic

"""

class SpanSynthetic:
    def __init__(self, valobj, *args):
        self.valobj = valobj # type: SBValue

        valtype = valobj.GetType()
        self._size = 0
        self._t = valtype.GetTemplateArgumentType(0)
        self._extent = valtype.GetTemplateArgumentType(1)
        self.sizeof_value = self._t.GetByteSize()

    def update(self):
        if not self.valobj.IsValid():
            self._size = 0
            return False

        storage = self.valobj.GetChildMemberWithName("s_")
        size = storage.GetChildMemberWithName('size')
        assert size.IsValid()
        self._size = size.GetValueAsUnsigned(0)
        self._dataobj = storage.GetChildMemberWithName('data')
        assert self._dataobj.IsValid()
        return False

    def has_children(self):
        return True

    def num_children(self):
        return self._size

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except TypeError as e:
            print(f"Failed to get child index {name}: {e}")
            return None

    def get_child_at_index(self, index):
        if not (0 <= index < self._size):
            print(f"Index {index} is out of bounds")
            # Out of bounds
            return None
        if not self.valobj.IsValid():
            print(f"Value is bad")
            # Value is bad?
            return None
        return self._dataobj.CreateChildAtOffset(
            "[{:d}]".format(index),
            index * self.sizeof_value,
            self._t
        )
