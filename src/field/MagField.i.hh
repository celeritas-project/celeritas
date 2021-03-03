//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagField.i.hh
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
/*!
 * Return a magnefic value at a given position
 */
CELER_FUNCTION auto MagField::operator()(const Real3& position) -> field_value
{
    // XXX: Not implemented yet, but a place holder for the next extension
    CELER_ENSURE(position[0] && position[1] && position[2]);
    return value_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
