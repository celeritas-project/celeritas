//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/GammaNuclearMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/GammaNuclearData.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate gamma-nuclear cross sections using the gamma-nuclear data.
 */
class GammaNuclearMicroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<GammaNuclearData>;
    using Energy = units::MevEnergy;
    using BarnXs = units::BarnXs;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    GammaNuclearMicroXsCalculator(ParamsRef const& shared, Energy energy);

    // Compute cross section
    inline CELER_FUNCTION BarnXs operator()(ElementId el_id) const;

  private:
    // Shared constant physics properties
    ParamsRef const& shared_;
    // Incident photon energy
    real_type const inc_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
GammaNuclearMicroXsCalculator::GammaNuclearMicroXsCalculator(
    ParamsRef const& shared, Energy energy)
    : shared_(shared), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute microscopic (element) cross section
 */
CELER_FUNCTION
auto GammaNuclearMicroXsCalculator::operator()(ElementId el_id) const -> BarnXs
{
    CELER_EXPECT(el_id < shared_.micro_xs.size());

    // Get element cross section data
    NonuniformGridRecord grid = shared_.micro_xs[el_id];

    // Calculate micro cross section at the given energy
    NonuniformGridCalculator calc_xs(grid, shared_.reals);
    real_type result = calc_xs(inc_energy_);

    return BarnXs{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
