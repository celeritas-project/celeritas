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
    // Shared cross section data
    ParamsRef const& data_;
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
    ParamsRef const& data, Energy energy)
    : data_(data), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute microscopic gamma-nuclear cross section at the given gamma energy.
 */
CELER_FUNCTION
auto GammaNuclearMicroXsCalculator::operator()(ElementId el_id) const -> BarnXs
{
    NonuniformGridRecord grid;

    // Use G4PARTICLEXS gamma-nuclear cross sections at low energy and CHIPS
    // parameterized cross sections at high energy.

    if (inc_energy_ <= data_.reals[data_.xs_iaea[el_id].grid.back()])
    {
        CELER_EXPECT(el_id < data_.xs_iaea.size());
        grid = data_.xs_iaea[el_id];
    }
    else
    {
        CELER_EXPECT(el_id < data_.xs_chips.size());
        grid = data_.xs_chips[el_id];
    }

    // Calculate micro cross section at the given energy
    NonuniformGridCalculator calc_xs(grid, data_.reals);
    real_type result = calc_xs(inc_energy_);

    return BarnXs{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
