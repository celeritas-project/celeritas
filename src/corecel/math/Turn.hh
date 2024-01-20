//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Turn.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Types.hh"

#include "Algorithms.hh"
#include "Quantity.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Unit for 2*pi radians
struct TwoPi
{
    static real_type value() { return 2 * static_cast<real_type>(m_pi); }
    //! Text label for output
    static char const* label() { return "tr"; }
};

//---------------------------------------------------------------------------//
//! Unit for pi/2 radians
struct HalfPi
{
    static real_type value() { return static_cast<real_type>(m_pi / 2); }
    //! Text label for output
    static char const* label() { return "qtr"; }
};

//---------------------------------------------------------------------------//
//! Quantity denoting a full turn
using Turn = Quantity<TwoPi, real_type>;

//---------------------------------------------------------------------------//
//! Quantity for an integer number of turns for axis swapping
using QuarterTurn = Quantity<HalfPi, int>;

//---------------------------------------------------------------------------//
//!@{
//! Special overrides for math functions for more precise arithmetic
CELER_FORCEINLINE_FUNCTION real_type sin(Turn r)
{
    return sinpi(r.value() * 2);
}

CELER_FORCEINLINE_FUNCTION real_type cos(Turn r)
{
    return cospi(r.value() * 2);
}

CELER_FORCEINLINE_FUNCTION void sincos(Turn r, real_type* sinv, real_type* cosv)
{
    return sincospi(r.value() * 2, sinv, cosv);
}

CELER_FORCEINLINE_FUNCTION int cos(QuarterTurn r)
{
    constexpr int cosval[] = {1, 0, -1, 0};
    return cosval[std::abs(r.value()) % 4];
}

CELER_FORCEINLINE_FUNCTION int sin(QuarterTurn r)
{
    // Define in terms of the symmetric "cos"
    return cos(QuarterTurn{r.value() - 1});
}

CELER_FORCEINLINE_FUNCTION void sincos(QuarterTurn r, int* sinv, int* cosv)
{
    *sinv = sin(r);
    *cosv = cos(r);
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
