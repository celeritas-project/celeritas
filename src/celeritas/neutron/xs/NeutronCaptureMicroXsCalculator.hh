//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/xs/NeutronCaptureMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/neutron/data/NeutronCaptureData.hh"

#include "NeutronMicroXsCalculator.hh"

namespace celeritas
{
class NeutronCaptureMicroXsCalculator;

//---------------------------------------------------------------------------//
/*!
 * XsData_traits for NeutronCaptureRef.
 */
template<>
struct XsData_traits<NeutronCaptureMicroXsCalculator>
{
    using ParamsRef = NeutronCaptureRef;
};

//---------------------------------------------------------------------------//
/*!
 * Calculate neutron capture cross sections from NeutronCaptureData.
 */
class NeutronCaptureMicroXsCalculator
    : public NeutronMicroXsCalculator<NeutronCaptureMicroXsCalculator>
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef =
        typename XsData_traits<NeutronCaptureMicroXsCalculator>::ParamsRef;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    NeutronCaptureMicroXsCalculator(ParamsRef const& shared, Energy energy)
        : NeutronMicroXsCalculator<NeutronCaptureMicroXsCalculator>(shared,
                                                                    energy)
    {
    }

  private:
    friend class NeutronMicroXsCalculator<NeutronCaptureMicroXsCalculator>;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
