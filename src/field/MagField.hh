//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagField.cuh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The MagField evaluates the magnetic field value at a given position.
 * Support: 1) a constant magnetic field
 *          2) a user defined field (e.g.: field map or parameterized field)
 *             along with MagFieldPointers and MagFieldStore: XXX TODO
 */
class MagField
{
  public:
    //@{
    //! Type aliases
    using field_value = Real3;
    //@}

  public:
    // Construct from a uniform field
    CELER_FUNCTION MagField(const field_value& value);

    // return a uniform magnetic field value
    CELER_FUNCTION field_value operator()();

    // return a magnetic field value at a given position
    CELER_FUNCTION field_value operator()(const Real3 position);

  private:
    //! Shared/persistent field data
    field_value value_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MagField.i.hh"
