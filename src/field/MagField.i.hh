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
MagField::MagField(const Real3& value) : value_(value), uniform_(true) {}

//---------------------------------------------------------------------------//
/*!
 * Return a magnetic field value at a given position
 */
CELER_FUNCTION Real3 MagField::operator()(const Real3& position) const
{
    return (uniform_) ? value_ : get_field(position);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate a magnetic field value at a given position - dummy for now
 * XXX TODO: 1) add ctors to support FieldMap or ParameterizedField
 *           2) evaluate the position dependent field value
 */
CELER_FUNCTION Real3 MagField::get_field(const Real3& position) const
{
    CELER_ENSURE(position.size() == 3);
    return value_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
