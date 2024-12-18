//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/detail/BitsetUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

/*!
 * Clear extra bits in the last word of a bitset.
 */
template<size_t ExtraBits>
struct Sanitize
{
    using word_type = unsigned int;
    static CELER_CONSTEXPR_FUNCTION void sanitize(word_type& word) noexcept
    {
        word &= ~((~static_cast<word_type>(0)) << ExtraBits);
    }
};

template<>
struct Sanitize<0>
{
    using word_type = unsigned int;
    static CELER_CONSTEXPR_FUNCTION void sanitize(word_type) noexcept {}
};

}  // namespace detail
}  // namespace celeritas
