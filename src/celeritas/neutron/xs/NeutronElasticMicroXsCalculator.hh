//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/grid/GenericCalculator.hh"
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
    using Energy = units::MevEnergy;
    using XsUnits = units::Native;  // [len^2]
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    NeutronElasticMicroXsCalculator(NeutronElasticRef const& shared,
                                    Energy energy);

    // Compute cross section
    inline CELER_FUNCTION real_type operator()(ElementId el_id) const;

  private:
    // Shared constant physics properties
    NeutronElasticRef const& shared_;
    // Incident neutron energy
    Energy const inc_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION NeutronElasticMicroXsCalculator::NeutronElasticMicroXsCalculator(
    NeutronElasticRef const& shared, Energy energy)
    : shared_(shared), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute microscopic (element) cross section
 */
CELER_FUNCTION
real_type NeutronElasticMicroXsCalculator::operator()(ElementId el_id) const
{
    CELER_EXPECT(el_id);

    // Get element cross section data
    GenericGridData grid = shared_.micro_xs[el_id];

    // Calculate micro cross section at the given energy
    GenericCalculator calc_xs(grid, shared_.reals);
    real_type result = calc_xs(inc_energy_.value());

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
