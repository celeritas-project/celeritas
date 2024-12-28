//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Turn.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"

#include "Algorithms.hh"
#include "Quantity.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Unit for 2*pi radians
struct TwoPi
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return 2 * constants::pi;
    }
    //! Text label for output
    static char const* label() { return "tr"; }
};

//---------------------------------------------------------------------------//
//! Unit for pi/2 radians
struct HalfPi
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return constants::pi / 2;
    }
    //! Text label for output
    static char const* label() { return "qtr"; }
};

//---------------------------------------------------------------------------//
/*!
 * Quantity denoting a full turn.
 *
 * Turns are a useful way of representing angles without the historical
 * arbitrariness of degrees or the roundoff errors of radians. See, for
 * example, https://www.computerenhance.com/p/turns-are-better-than-radians .
 *
 * \todo Template on real type and template the functions below.
 */
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

CELER_CONSTEXPR_FUNCTION int cos(QuarterTurn r)
{
    constexpr int cosval[] = {1, 0, -1, 0};
    return cosval[(r.value() > 0 ? r.value() : -r.value()) % 4];
}

CELER_CONSTEXPR_FUNCTION int sin(QuarterTurn r)
{
    // Define in terms of the symmetric "cos"
    return cos(QuarterTurn{r.value() - 1});
}

CELER_CONSTEXPR_FUNCTION void sincos(QuarterTurn r, int* sinv, int* cosv)
{
    *sinv = sin(r);
    *cosv = cos(r);
}
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
