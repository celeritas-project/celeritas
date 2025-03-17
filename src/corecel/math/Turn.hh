//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Turn.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

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
 */
template<class T>
using Turn_t = Quantity<TwoPi, T>;

//! Turn with default precision (DEPRECATEDish)
using Turn = Turn_t<real_type>;
//! Turn with default precision
using RealTurn = Turn_t<real_type>;

//! Create a turn using template deduction
template<class T>
CELER_CONSTEXPR_FUNCTION Turn_t<T> make_turn(T value)
{
    static_assert(std::is_floating_point_v<T>,
                  "turn type must be floating point");
    return Turn_t<T>{value};
}

//---------------------------------------------------------------------------//
//! Quantity for an integer number of turns for axis swapping (DEPRECATEDish)
using QuarterTurn = Quantity<HalfPi, int>;
//! Quantity for an integer number of turns for axis swapping
using IntQuarterTurn = Quantity<HalfPi, int>;

//---------------------------------------------------------------------------//
//!@{
//! Special overrides for math functions for more precise arithmetic
template<class T>
CELER_FORCEINLINE_FUNCTION T sin(Turn_t<T> r)
{
    return sinpi(r.value() * 2);
}

template<class T>
CELER_FORCEINLINE_FUNCTION T cos(Turn_t<T> r)
{
    return cospi(r.value() * 2);
}

template<class T>
CELER_FORCEINLINE_FUNCTION void sincos(Turn_t<T> r, T* sinv, T* cosv)
{
    return sincospi(r.value() * 2, sinv, cosv);
}

CELER_CONSTEXPR_FUNCTION int cos(IntQuarterTurn r)
{
    constexpr int cosval[] = {1, 0, -1, 0};
    return cosval[(r.value() > 0 ? r.value() : -r.value()) % 4];
}

CELER_CONSTEXPR_FUNCTION int sin(IntQuarterTurn r)
{
    // Define in terms of the symmetric "cos"
    return cos(IntQuarterTurn{r.value() - 1});
}

CELER_CONSTEXPR_FUNCTION void sincos(IntQuarterTurn r, int* sinv, int* cosv)
{
    *sinv = sin(r);
    *cosv = cos(r);
}
//!@}

//! Math functions that return turns from types
template<class T>
CELER_FORCEINLINE_FUNCTION Turn_t<T> atan2turn(T y, T x)
{
    // TODO: some OS/lib has this natively; use that instad
    return native_value_to<Turn_t<T>>(std::atan2(y, x));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
