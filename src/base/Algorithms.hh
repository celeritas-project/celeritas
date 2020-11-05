//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Algorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return the lower of two values.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION const T& min(const T& a, const T& b)
{
    return (b < a) ? b : a;
}

//---------------------------------------------------------------------------//
/*!
 * Return the higher of two values.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION const T& max(const T& a, const T& b)
{
    return (b > a) ? b : a;
}

//---------------------------------------------------------------------------//

/*!
 * Return the cube of the input value.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T cube(const T& a)
{
    return a * a * a;
}

//---------------------------------------------------------------------------//
} // namespace celeritas