//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/inp/Grid.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the range from the energy loss.
 *
 * The range of a particle with energy \f$ E_0 \f$ is calculated by integrating
 * the reciprocal of the stopping power over the energy:
 * \f[
   R(E_0) = \int_0^{E_0} - \difd{x}{E} \dif E.
 * \f]
 * Given an energy loss grid for a single particle type and material, this
 * numerically integrates the range.  To keep the range tables as consistent as
 * possible with what we've been importing from Geant4, this performs the same
 * calculation as in Geant4's \c G4LossTableBuilder::BuildRangeTable, which
 * uses the midpoint rule with 100 substeps for improved accuracy.
 *
 * The calculator is constructed with the boundary conditions for cubic spline
 * interpolation. If the default constructor is used, or if the number of grid
 * points is less than 5, linear interpolation will be used instead.
 *
 * \todo support polynomial interpolation as well?
 */
class RangeGridCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using BC = SplineDerivCalculator::BoundaryCondition;
    //!@}

  public:
    // Default constructor
    RangeGridCalculator();

    // Construct with boundary conditions for spline interpolation
    explicit RangeGridCalculator(BC);

    // Calculate the range from the energy loss for a single material
    inp::UniformGrid operator()(inp::UniformGrid const&) const;

  private:
    BC bc_;

    //! Number of substeps in the numerical integration
    static constexpr size_type integration_substeps() { return 100; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
