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
 * Evaluate the magnetic field based on a parameterized function
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

    // Return the magnetic field at the given position
    CELER_FUNCTION
    inline Real3 operator()(Real3 pos);

  private:
    CELER_FUNCTION
    inline Real3 evaluate_field(real_type r, real_type z);

    CELER_FUNCTION
    inline Real4 evaluate_parameters(real_type u);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "CMSParameterizedField.i.hh"
