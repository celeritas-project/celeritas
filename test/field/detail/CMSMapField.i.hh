//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSMapField.i.hh
//---------------------------------------------------------------------------//
#include "CMSMapField.hh"

#include "base/Algorithms.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/Units.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the shared magnetic field map data (FieldMapGroup).
 */
CELER_FUNCTION
CMSMapField::CMSMapField(const FieldMapRef& shared) : shared_(shared) {}

//---------------------------------------------------------------------------//
/*!
 * Retrieve the magnetic field value for the given position.
 */
CELER_FUNCTION Real3 CMSMapField::operator()(const Real3& pos) const
{
    CELER_ENSURE(shared_);

    Real3 value{0, 0, 0};

    real_type r = std::sqrt(ipow<2>(pos[0]) + ipow<2>(pos[1]));
    real_type z = pos[2];

    real_type scale = 1 / shared_.params.delta_grid;

    size_type ir = static_cast<size_type>(r * scale);
    size_type iz
        = static_cast<size_type>((z + shared_.params.offset_z) * scale);

    real_type dr = r - static_cast<real_type>(ir) * shared_.params.delta_grid;
    real_type dz = z + shared_.params.offset_z
                   - static_cast<real_type>(iz) * shared_.params.delta_grid;

    if (!shared_.valid(iz, ir))
        return value;

    // z component
    real_type low  = shared_.fieldmap[shared_.id(iz, ir)].value_z;
    real_type high = shared_.fieldmap[shared_.id(iz + 1, ir)].value_z;

    value[2] = units::tesla * (low + (high - low) * dz * scale);

    // x and y components
    low  = shared_.fieldmap[shared_.id(iz, ir)].value_r;
    high = shared_.fieldmap[shared_.id(iz, ir + 1)].value_r;

    real_type tmp = (r != 0) ? (low + (high - low) * dr * scale) / r : low;
    value[0]      = units::tesla * tmp * pos[0];
    value[1]      = units::tesla * tmp * pos[1];

    return value;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
