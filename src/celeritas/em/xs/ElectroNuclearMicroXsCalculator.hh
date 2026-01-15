//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/ElectroNuclearMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/ElectroNuclearData.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate electro-nuclear cross sections.
 */
class ElectroNuclearMicroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<ElectroNuclearData>;
    using Energy = units::MevEnergy;
    using BarnXs = units::BarnXs;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    ElectroNuclearMicroXsCalculator(ParamsRef const& shared, Energy energy);

    // Compute cross section
    inline CELER_FUNCTION BarnXs operator()(ElementId el_id) const;

  private:
    // Shared cross section data
    ParamsRef const& data_;
    // Incident particle energy
    real_type const inc_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
ElectroNuclearMicroXsCalculator::ElectroNuclearMicroXsCalculator(
    ParamsRef const& data, Energy energy)
    : data_(data), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute microscopic electro-nuclear cross section at the given particle
 * energy.
 */
CELER_FUNCTION
auto ElectroNuclearMicroXsCalculator::operator()(ElementId el_id) const
    -> BarnXs
{
    NonuniformGridRecord grid;

    // Use tabulated electro-nuclear micro cross sections
    CELER_EXPECT(el_id < data_.micro_xs.size());
    grid = data_.micro_xs[el_id];

    // Calculate micro cross section at the given energy
    NonuniformGridCalculator calc_micro_xs(grid, data_.reals);
    real_type result = calc_micro_xs(inc_energy_);

    return BarnXs{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
