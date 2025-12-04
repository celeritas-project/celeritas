//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/GridTypes.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Which of two bounding points, faces, energies, etc.
 *
 * Here, lo/hi can correspond to left/right, back/front, bottom/top. It's used
 * for the two points in a bounding box.
 */
enum class Bound : unsigned char
{
    lo,
    hi,
    size_
};

//---------------------------------------------------------------------------//
//! Interpolation type
enum class Interp
{
    linear,
    log,
    size_
};

//---------------------------------------------------------------------------//
//! Interpolation for physics grids
enum class InterpolationType
{
    linear,
    poly_spline,  //!< Piecewise polynomial interpolation
    cubic_spline,  //!< Cubic spline interpolation with \f$ C^2 \f$ continuity
    size_
};

//---------------------------------------------------------------------------//
//! Cubic spline interpolation boundary conditions
enum class SplineBoundaryCondition
{
    natural = 0,
    not_a_knot,
    geant,  //!< Geant4's "not-a-knot"
    size_
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

//! Convert Bound enum value to int
CELER_CONSTEXPR_FUNCTION int to_int(Bound b)
{
    return static_cast<int>(b);
}

// Get a string corresponding to the interpolation method
char const* to_cstring(InterpolationType value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
