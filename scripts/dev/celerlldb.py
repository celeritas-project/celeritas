# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Add LLDB wrappers for Celeritas types.

To use from inside ``${SOURCE}/build``::
   (lldb) command script import ../scripts/dev/celerlldb.py --allow-reload
   (lldb) type synthetic add -x "^celeritas::Span<.+>$" --python-class celerlldb.SpanSynthetic
   (lldb) type synthetic add -x "^celeritas::Array<.+>$" --python-class celerlldb.ArraySynthetic
   (lldb) type synthetic add -x "^celeritas::ItemRange<.+>$" --python-class celerlldb.ItemRangeSynthetic
   (lldb) type summary add -x "^celeritas::OpaqueId<.+>$" --python-function celerlldb.opaqueid_summary

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lldb import SBValue


class _ContiguousSyntheticBase:
    def __init__(self, valobj, *args):
        self.valobj = valobj  # type: SBValue
        self._size = 0
        self._t = None
        self._dataobj = None
        self._sizeof_value = 0

    def _set_storage(self, dataobj, size, value_type):
        self._dataobj = dataobj
        self._size = size
        self._t = value_type
        if value_type is None or not value_type.IsValid():
            self._sizeof_value = 0
        else:
            self._sizeof_value = value_type.GetByteSize()

    def has_children(self):
        return True

    def num_children(self):
        return self._size

    def get_child_index(self, name):
        try:
            # See CreateChildAtOffset
            return int(name[1:-1])
        except (TypeError, ValueError) as e:
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
        if self._dataobj is None or not self._dataobj.IsValid():
            print("Container data is invalid")
            return None
        if self._t is None or not self._t.IsValid() or self._sizeof_value <= 0:
            print("Container value type is invalid")
            return None
        return self._dataobj.CreateChildAtOffset(
            "[{:d}]".format(index), index * self._sizeof_value, self._t
        )


class SpanSynthetic(_ContiguousSyntheticBase):
    def __init__(self, valobj, *args):
        super().__init__(valobj, *args)

        valtype = valobj.GetType()
        self._value_t = valtype.GetTemplateArgumentType(0)

    def update(self):
        if not self.valobj.IsValid():
            self._set_storage(None, 0, self._value_t)
            return False

        storage = self.valobj.GetChildMemberWithName("s_")
        assert storage.IsValid()
        size = storage.GetChildMemberWithName("size")
        assert size.IsValid()
        data = storage.GetChildMemberWithName("data")
        assert data.IsValid()
        self._set_storage(data, size.GetValueAsUnsigned(0), self._value_t)
        return False


class ArraySynthetic(_ContiguousSyntheticBase):
    def __init__(self, valobj, *args):
        super().__init__(valobj, *args)

        valtype = valobj.GetType()
        self._value_t = valtype.GetTemplateArgumentType(0)

    def update(self):
        if not self.valobj.IsValid():
            self._set_storage(None, 0, self._value_t)
            return False

        data = self.valobj.GetChildMemberWithName("d_")
        assert data.IsValid()
        data_ptr = data.AddressOf()
        assert data_ptr.IsValid()

        self._set_storage(data_ptr, data.GetNumChildren(), self._value_t)
        return False


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
    size_bytes = value.GetType().GetByteSize()
    if size_bytes <= 0 or size_bytes > 8:
        return str(raw_value)
    num_bits = 8 * size_bytes
    null_value = (1 << num_bits) - 1

    if raw_value == null_value:
        return "null"

    return str(raw_value)
