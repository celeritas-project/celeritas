//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The MagField evaluates the magnetic field value at a given position.
 */
class MagField
{
  public:
    //@{
    //! Type aliases
    using field_value = Real3;
    using field_type  = bool;
    //@}

  public:
    // Construct from a uniform field
    CELER_FUNCTION MagField(const field_value& value);

    // Return a magnetic field value at a given position
    CELER_FUNCTION field_value operator()(const Real3& position) const;

    // Interface for a position-dependent magnetic field value
    CELER_FUNCTION field_value get_field(const Real3& position) const;

  private:
    //! Shared/persistent field data
    field_value value_;
    field_type  is_uniform_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MagField.i.hh"
