//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/xs/NeutronElasticMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/neutron/data/NeutronElasticData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate neutron elastic cross sections from NeutronElasticXsData
 */
class NeutronElasticMicroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NeutronElasticRef;
    using Energy = units::MevEnergy;
    using BarnXs = units::BarnXs;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    NeutronElasticMicroXsCalculator(ParamsRef const& shared, Energy energy);

    // Compute cross section
    inline CELER_FUNCTION BarnXs operator()(ElementId el_id) const;

  private:
    // Shared constant physics properties
    NeutronElasticRef const& shared_;
    // Incident neutron energy
    real_type const inc_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION NeutronElasticMicroXsCalculator::NeutronElasticMicroXsCalculator(
    ParamsRef const& shared, Energy energy)
    : shared_(shared), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute microscopic (element) cross section
 */
CELER_FUNCTION
auto NeutronElasticMicroXsCalculator::operator()(ElementId el_id) const -> BarnXs
{
    CELER_EXPECT(el_id < shared_.micro_xs.size());

    // Calculate micro cross section at the given energy
    NonuniformGridCalculator calc_xs(shared_.micro_xs[el_id], shared_.reals);
    real_type result = calc_xs(inc_energy_);

    return BarnXs{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
