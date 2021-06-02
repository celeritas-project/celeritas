//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformMagField.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate magnetic field based on a parameterized function
 */
class UniformMagField
{
  public:
    // Construct the reader and locate the data using the environment variable
    CELER_FUNCTION
    explicit UniformMagField(Real3 value) : value_(value) {}

    // Return a const magnetic field value
    CELER_FUNCTION
    Real3 operator()(CELER_MAYBE_UNUSED Real3 pos) const { return value_; }

  private:
    Real3 value_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
