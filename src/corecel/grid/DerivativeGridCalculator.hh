//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/DerivativeGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/inp/Grid.hh"
#include "corecel/math/NumericLimits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the derivatives of a given grid.
 *
 * Since grids are piecewise function definitions, the derivative might not be
 * well defined in a region around a grid-point. An epsilon-neighborhood is
 * constructed around each grid-point where the derivative may be interpolated
 * between the two derivative grid-points. Outside of this neighborhood the
 * derivative is well defined since the interpolation function is smooth.
 *
 *
 * \todo Currently only linearly interpolated grids are supported since they
 * are necessary for calculating group velocity from refractive index.
 */
class DerivativeGridCalculator
{
  public:
    // Construct with a reference to mutable host data
    explicit DerivativeGridCalculator(real_type epsilon
                                      = NumericLimits<real_type>::epsilon());

    // Construct the derivative grid of an imported grid
    inp::Grid operator()(inp::Grid const&);

  private:
    real_type epsilon_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
