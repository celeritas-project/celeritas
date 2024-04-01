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
#include "celeritas/neutron/data/NeutronElasticData.hh"

#include "NeutronMicroXsCalculator.hh"

namespace celeritas
{
class NeutronElasticMicroXsCalculator;

//---------------------------------------------------------------------------//
/*!
 * XsData_traits for NeutronElasticRef.
 */
template<>
struct XsData_traits<NeutronElasticMicroXsCalculator>
{
    using ParamsRef = NeutronElasticRef;
};

//---------------------------------------------------------------------------//
/*!
 * Calculate neutron elastic cross sections from NeutronElasticData.
 */
class NeutronElasticMicroXsCalculator
    : public NeutronMicroXsCalculator<NeutronElasticMicroXsCalculator>
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef =
        typename XsData_traits<NeutronElasticMicroXsCalculator>::ParamsRef;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    NeutronElasticMicroXsCalculator(ParamsRef const& shared, Energy energy)
        : NeutronMicroXsCalculator<NeutronElasticMicroXsCalculator>(shared,
                                                                    energy)
    {
    }

  private:
    friend class NeutronMicroXsCalculator<NeutronElasticMicroXsCalculator>;
};

//---------------------------------------------------------------------------//
<<<<<<< HEAD
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
auto NeutronElasticMicroXsCalculator::operator()(ElementId el_id) const
    -> BarnXs
{
    CELER_EXPECT(el_id < shared_.micro_xs.size());

    // Calculate micro cross section at the given energy
    GenericCalculator calc_xs(shared_.micro_xs[el_id], shared_.reals);
    real_type result = calc_xs(inc_energy_);

    return BarnXs{result};
}

//---------------------------------------------------------------------------//
=======
>>>>>>> 415eef003 (Add NeutronMicroXsCalculator)
}  // namespace celeritas
