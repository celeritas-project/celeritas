//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/NeutronMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * XsData_traits defined in the derived class.
 */
template<typename T>
struct XsData_traits;

//---------------------------------------------------------------------------//
/*!
 * Calculate micro cross sections from XsData. This is a static base class of
 * neutron micro cross section calculators.
 *
 * \tparam T derived microscopic cross section calculator
 */
template<class T>
class NeutronMicroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = typename XsData_traits<T>::ParamsRef;
    using Energy = units::MevEnergy;
    using BarnXs = units::BarnXs;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    NeutronMicroXsCalculator(ParamsRef const& shared, Energy energy);

    // Compute cross section
    inline CELER_FUNCTION BarnXs operator()(ElementId el_id) const;

  protected:
    // Shared constant physics properties
    ParamsRef const& shared_;
    // Incident neutron energy
    real_type const inc_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
template<class T>
CELER_FUNCTION
NeutronMicroXsCalculator<T>::NeutronMicroXsCalculator(ParamsRef const& shared,
                                                      Energy energy)
    : shared_(shared), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute microscopic (element) cross section.
 */
template<typename T>
CELER_FUNCTION auto
NeutronMicroXsCalculator<T>::operator()(ElementId el_id) const -> BarnXs
{
    CELER_EXPECT(el_id < shared_.micro_xs.size());

    // Get element cross section data
    GenericGridData grid = shared_.micro_xs[el_id];

    // Calculate micro cross section at the given energy
    GenericCalculator calc_xs(grid, shared_.reals);
    real_type result = calc_xs(inc_energy_);

    return BarnXs{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
