//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/RangeGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"

#include "SplineXsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the range from the energy loss.
 *
 * The range of a particle with energy \f$ E_0 \f$ is calculated by integrating
 * the reciprocal of the stopping power over the energy:
 * \f[
   R(E_0) = \int_0^{E_0} - \frac{\dif x}{\dif E} \dif E.
 * \f]
 * Given an energy loss grid for a single particle type and material, this
 * numerically integrates the range.  To keep the range tables as consistent as
 * possible with what we've been importing from Geant4, this performs the same
 * calculation as in Geant4's \c G4LossTableBuilder::BuildRangeTable, which
 * uses the midpoint rule with 100 substeps for improved accuracy.
 */
class RangeGridCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Values
        = Collection<real_type, Ownership::const_reference, MemSpace::host>;
    using EnergyLossCalculator = SplineXsCalculator;
    //!@}

  public:
    // Construct with the energy loss grid and spline interpolation order
    RangeGridCalculator(XsGridData const& grid,
                        Values const& reals,
                        size_type spline_order);

    // Calculate the range for a single material
    std::vector<real_type> operator()() const;

  private:
    EnergyLossCalculator calc_dedx_;
    UniformGrid loge_grid_;

    //! Number of substeps in the numerical integration
    static constexpr size_type integration_substeps() { return 100; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
