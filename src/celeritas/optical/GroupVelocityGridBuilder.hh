//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/GroupVelocityGridBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Range.hh"
#include "corecel/grid/DerivativeGridCalculator.hh"
#include "corecel/inp/Grid.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the group velocity grid from a refractive-index grid.
 *
 * The group velocity for refractive index is given by
 * \f[
 * v_g = \frac{c}{n + E \frac{dn}{dE}}
 * \f]
 * where \f[\frac{dn}{dE}\f] is the derivative of the refractive index with
 * respect to energy.
 * Geant4 uses the following formula for the group velocity:
 * \f[
 * v_g = \frac{c}{n + \frac{dn}{d\ln(E)}}
 * \f]
 * which is equivalent to the above formula since \f[\frac{dn}{d\ln(E)} = E
 \frac{dn}{dE}\f].
 * This will give similar results on denser grids, but may differ on coarser
 grids.
 */
class GroupVelocityGridBuilder
{
  public:
    // Construct the group-velocity grid from a refractive-index grid
    explicit GroupVelocityGridBuilder(NonuniformGridCalculator refractive_index)
        : refractive_index_(refractive_index)
    {
    }

    inp::Grid operator()(inp::Grid const& refractive_index) const;

  private:
    NonuniformGridCalculator refractive_index_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
