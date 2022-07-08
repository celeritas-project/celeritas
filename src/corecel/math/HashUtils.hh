//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/HashUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>

#include "corecel/Macros.hh"

#include "detail/HashUtilsImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Combine hashes of the given arguments using a fast hash algorithm.
 *
 * See https://florianjw.de/en/variadic_templates.html for why we constructed
 * this as such. By making the variadic template use function argument
 * expansion rather than recursion, we can unpack the args in a left-to-right
 * order. The `(HASH,0)` construction is to give the unpacked expression a
 * return type; and putting these in an `initializer_list` constructor
 * guarantees the hashes are evaluated from left to right (unlike a typical
 * argument expansion where the orders may be arbitrary).
 */
template<class... Args>
std::size_t hash_combine(Args const&... args)
{
    // Construct a hasher and initialize
    std::size_t             result;
    CELER_MAYBE_UNUSED auto hash_fnv = detail::make_fast_hasher(&result);

    // Hash each one of the arguments sequentially by expanding into an unused
    // initializer list.
    (void)std::initializer_list<int>{(hash_fnv(std::hash<Args>()(args)), 0)...};

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
