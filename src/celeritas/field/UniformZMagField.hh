//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformZMagField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A uniform magnetic field along Z axis
 */
class UniformZMagField
{
  public:
    // Construct with a scalar magnetic field value
    CELER_FUNCTION
    explicit UniformZMagField(real_type value) : value_({0, 0, value}) {}

    // Return a constant magnetic field value along Z axis
    CELER_FUNCTION
    Real3 operator()(CELER_MAYBE_UNUSED const Real3& pos) const
    {
        return value_;
    }

  private:
    Real3 value_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
