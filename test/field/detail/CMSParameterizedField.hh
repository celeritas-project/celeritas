//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSParameterizedField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the value of magnetic field based on a parameterized function in
 * the tracker region of the CMS detector
 *
 * This code is experted from CMSSW and only used for the purpose of the
 * magnetic field tests.
 */
class CMSParameterizedField
{
    //!@{
    //! Type aliases
    using Real4 = celeritas::Array<real_type, 4>;
    //!@}

  public:
    CELER_FUNCTION
    inline CMSParameterizedField() {}

    // Return the magnetic field for the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 pos);

  private:
    // Evaluate the magnetic field for the given r and z
    CELER_FUNCTION
    inline Real3 evaluate_field(real_type r, real_type z);

    // Evaluate the parameterized function and its derivatives
    CELER_FUNCTION
    inline Real4 evaluate_parameters(real_type x);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "CMSParameterizedField.i.hh"
