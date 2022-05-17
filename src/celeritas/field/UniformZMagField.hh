//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformZMagField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A uniform magnetic field along the Z axis.
 */
class UniformZMagField
{
  public:
    //! Construct with a scalar magnetic field value
    CELER_FUNCTION
    explicit UniformZMagField(real_type value) : value_(value) {}

    //! Return the magnetic field at the given position
    CELER_FUNCTION
    Real3 operator()(const Real3&) const { return {0, 0, value_}; }

  private:
    real_type value_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
