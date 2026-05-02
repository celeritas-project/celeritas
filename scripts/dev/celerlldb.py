# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Add LLDB wrappers for Celeritas types.

To use from inside ``${SOURCE}/build``::
   (lldb) command script import ../scripts/dev/celerlldb.py --allow-reload
   (lldb) type synthetic add -x "^celeritas::Span<.+>$" --python-class celerlldb.SpanSynthetic
   (lldb) type synthetic add -x "^celeritas::ItemRange<.+>$" --python-class celerlldb.ItemRangeSynthetic
   (lldb) type summary add -x "^celeritas::OpaqueId<.+>$" --python-function celerlldb.opaqueid_summary

"""


class SpanSynthetic:
    def __init__(self, valobj, *args):
        self.valobj = valobj  # type: SBValue

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
        size = storage.GetChildMemberWithName("size")
        assert size.IsValid()
        self._size = size.GetValueAsUnsigned(0)
        self._dataobj = storage.GetChildMemberWithName("data")
        assert self._dataobj.IsValid()
        return False

    def has_children(self):
        return True

    def num_children(self):
        return self._size

    def get_child_index(self, name):
        try:
            # See CreateChildAtOffset
            return int(name[1:-1])
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
            "[{:d}]".format(index), index * self.sizeof_value, self._t
        )


class ItemRangeSynthetic:
    def __init__(self, valobj, *args):
        self.valobj = valobj  # type: SBValue

        valtype = valobj.GetType()
        self._value_t = valtype.GetTemplateArgumentType(0)

    def update(self):
        if not self.valobj.IsValid():
            return False

        # Save begin/end names and underlying OpaqueID's data
        self.values_ = []
        for n in ["begin", "end"]:
            iter = self.valobj.GetChildMemberWithName(n + "_")
            self.values_.append((n, iter.GetChildMemberWithName("value_")))

        return False

    def has_children(self):
        return True

    def num_children(self):
        return len(self.values_)

    def get_child_index(self, name):
        # Find the index of the child in the begin/end list
        for i, (n, _) in enumerate(self.values_):
            if n == name:
                return i
        return None

    def get_child_at_index(self, index):
        if not (0 <= index < len(self.values_)):
            print(f"Index {index} is out of bounds")
            # Out of bounds
            return None

        # Print the opaque index
        (name, val) = self.values_[index]
        val_int = val.GetValueAsUnsigned()
        return self.valobj.CreateValueFromExpression(name, f"(unsigned){val_int}")


def opaqueid_summary(valobj, _internal_dict):
    value = valobj.GetChildMemberWithName("value_")
    if not valobj.IsValid() or not value.IsValid():
        return "null"

    raw_value = value.GetValueAsUnsigned(0)

    # OpaqueId null sentinel is max value for the unsigned index type.
    byte_size = value.GetType().GetByteSize()
    if byte_size <= 0:
        return str(raw_value)
    bit_size = 8 * byte_size
    null_value = (1 << min(bit_size, 64)) - 1

    if raw_value == null_value:
        return "null"

    return str(raw_value)
