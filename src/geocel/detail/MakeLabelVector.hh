//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/MakeLabelVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/io/Label.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a vector of labels from a map of vectors of pointers.
 *
 * This implementation detail expects a map of string identifiers to a vector
 * of pointers. The \c get_id function should convert a dereferenced item to an
 * ID.
 */
template<class T, class GetId>
std::vector<Label>
make_label_vector(std::unordered_map<std::string, std::vector<T>> const& names,
                  GetId&& get_id)
{
    std::vector<Label> result;
    auto add_label = [&result](std::size_t id, Label&& label) {
        if (id >= result.size())
        {
            result.resize(id + 1);
        }
        result[id] = std::move(label);
    };

    for (auto&& [name, items] : names)
    {
        CELER_ASSERT(!items.empty());
        if (items.size() == 1)
        {
            // Label is just the name since this item is unique
            CELER_ASSERT(items.front());
            add_label(get_id(items.front()), Label{name});
            continue;
        }

        // Set labels in increasing order
        for (auto idx : range(items.size()))
        {
            CELER_ASSERT(items[idx]);
            add_label(get_id(items[idx]), Label{name, std::to_string(idx)});
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
