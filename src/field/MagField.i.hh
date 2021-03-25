//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagField.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a uniform magnetic field
 */
CELER_FUNCTION
MagField::MagField(const Real3& value) : value_(value) {}

//---------------------------------------------------------------------------//
/*!
 * Return a magnetic field value at a given position
 */
CELER_FUNCTION Real3 MagField::operator()() const
{
    return value_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
