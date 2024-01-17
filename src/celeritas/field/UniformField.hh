//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A uniform field.
 *
 * The values of the field should be in native units. For magnetic fields, this
 * unit is gauss.
 */
class UniformField
{
  public:
    //! Construct with a field vector
    explicit CELER_FUNCTION UniformField(Real3 const& value) : value_(value) {}

    //! Return the field at the given position
    CELER_FUNCTION Real3 const& operator()(Real3 const&) const
    {
        return value_;
    }

  private:
    Real3 value_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
