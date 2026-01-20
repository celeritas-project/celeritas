//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/UniformFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A uniform field.
 *
 * The values of the field should be in native units. For magnetic fields, this
 * unit is gauss for the CGS system.
 */
class UniformField
{
  public:
    using ParamsRef = NativeCRef<UniformFieldParamsData>;

    //! Construct with a field vector
    explicit CELER_FUNCTION UniformField(Real3 const& value) : value_(value) {}

    //! Construct with field params
    explicit CELER_FUNCTION UniformField(ParamsRef const& params)
        : value_(params.field)
    {
    }

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
