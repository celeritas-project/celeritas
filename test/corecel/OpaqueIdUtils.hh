//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/OpaqueIdUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Convert an opaque ID for easier testing.
 */
template<class I, class T>
inline constexpr int id_to_int(OpaqueId<I, T> val) noexcept
{
    if (!val)
        return -1;
    if (val.unchecked_get() > static_cast<T>(std::numeric_limits<int>::max()))
        return -2;
    return static_cast<int>(val.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Convert a vector of opaque IDs for easier testing.
 */
template<class I, class T>
inline auto id_to_int(Span<OpaqueId<I, T> const> vals)
{
    std::vector<int> result;
    result.reserve(vals.size());
    for (auto const& val : vals)
    {
        result.push_back(id_to_int(val));
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
