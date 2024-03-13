//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/xs/NeutronInelasticMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/neutron/data/NeutronCaptureData.hh"

#include "NeutronMicroXsCalculator.hh"

namespace celeritas
{
class NeutronInelasticMicroXsCalculator;

//---------------------------------------------------------------------------//
/*!
 * XsData_traits for NeutronInelasticRef.
 */
template<>
struct XsData_traits<NeutronInelasticMicroXsCalculator>
{
    using ParamsRef = NeutronInelasticRef;
};

//---------------------------------------------------------------------------//
/*!
 * Calculate neutron inelastic cross sections from NeutronInelasticData.
 */
class NeutronInelasticMicroXsCalculator
    : public NeutronMicroXsCalculator<NeutronInelasticMicroXsCalculator>
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef =
        typename XsData_traits<NeutronInelasticMicroXsCalculator>::ParamsRef;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    NeutronInelasticMicroXsCalculator(ParamsRef const& shared, Energy energy)
        : NeutronMicroXsCalculator<NeutronInelasticMicroXsCalculator>(shared,
                                                                      energy)
    {
    }

  private:
    friend class NeutronMicroXsCalculator<NeutronInelasticMicroXsCalculator>;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
