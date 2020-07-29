//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Array.i.hh
//---------------------------------------------------------------------------//

#include "Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Test equality of two arrays.
 */
template<typename T, std::size_t N>
CELER_FUNCTION bool operator==(const array<T, N>& lhs, const array<T, N>& rhs)
{
    for (std::size_t i = 0; i != N; ++i)
    {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Test inequality of two arrays.
 */
template<typename T, std::size_t N>
CELER_FUNCTION bool operator!=(const array<T, N>& lhs, const array<T, N>& rhs)
{
    return !(lhs == rhs);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
