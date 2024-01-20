//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformZField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A uniform field along the Z axis.
 */
class UniformZField
{
  public:
    //! Construct with a scalar magnetic field value
    CELER_FUNCTION
    explicit UniformZField(real_type value) : value_(value) {}

    //! Return the field at the given position
    CELER_FUNCTION
    Real3 operator()(Real3 const&) const { return {0, 0, value_}; }

  private:
    real_type value_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
