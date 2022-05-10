//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformMagField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A uniform magnetic field
 */
class UniformMagField
{
  public:
    // Construct with a uniform magnetic field vector
    CELER_FUNCTION
    explicit UniformMagField(Real3 value) : value_(value) {}

    // Return a const magnetic field value
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
