//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSMapField.i.hh
//---------------------------------------------------------------------------//
#include "CMSMapField.hh"

#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/Units.hh"

#include <math.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the volume-based magnetic field map excerpted from CMSSW.
 *
 * Values of the CMS magnetic field were stored at the r-z grid point in the
 * unit length (i.e., units::cm) in the r-z
 * [-offset_z:offset_z][0]
 * detail::FieldMapInput from
 *
 */
CELER_FUNCTION
CMSMapField::CMSMapField(const FieldMapRef& shared) : shared_(shared) {}

//---------------------------------------------------------------------------//
/*!
 * Return a magnetic field value at a given position.
 */
CELER_FUNCTION Real3 CMSMapField::operator()(Real3 pos) const
{
    CELER_ENSURE(shared_);

    Real3 value{0, 0, 0};

    real_type r = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
    real_type z = pos[2];

    size_type ir = (size_type)(r);
    size_type iz = (size_type)(z + shared_.params.offset_z);

    real_type dr = r - real_type(ir);
    real_type dz = z + shared_.params.offset_z - real_type(iz);

    if (!shared_.valid(iz, ir))
        return value;

    // z component
    real_type low  = shared_.fieldmap[shared_.id(iz, ir)].value_z;
    real_type high = shared_.fieldmap[shared_.id(iz + 1, ir)].value_z;

    value[2] = units::tesla * (low + (high - low) * dz);

    // x and y components
    low  = shared_.fieldmap[shared_.id(iz, ir)].value_r;
    high = shared_.fieldmap[shared_.id(iz, ir + 1)].value_r;

    real_type tmp = (r != 0) ? (low + (high - low) * dr) / r : low;
    value[0]      = units::tesla * tmp * pos[0];
    value[1]      = units::tesla * tmp * pos[1];

    return value;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
