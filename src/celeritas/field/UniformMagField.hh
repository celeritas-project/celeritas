//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformMagField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A uniform magnetic field.
 */
class UniformMagField
{
  public:
    //! Construct with a uniform magnetic field vector
    explicit CELER_FUNCTION UniformMagField(Real3 value) : value_(value) {}

    //! Return the magnetic field at the given position
    CELER_FUNCTION Real3 operator()(const Real3&) const { return value_; }

  private:
    Real3 value_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
