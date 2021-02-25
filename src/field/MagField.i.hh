//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagField.i.cuh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a uniform magnetic field
 */
CELER_FUNCTION
MagField::MagField(const field_value& value) : value_(value) {}

//---------------------------------------------------------------------------//
/*!
 * Return a uniform magnefic value
 */
CELER_FUNCTION auto MagField::operator()() -> field_value
{
    return value_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
