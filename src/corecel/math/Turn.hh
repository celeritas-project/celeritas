//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Turn.hh
//---------------------------------------------------------------------------//
#pragma once

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
//! Quantity denoting a full turn
using Turn = Quantity<TwoPi, real_type>;

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
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
