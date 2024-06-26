//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/IntersectSurfaceState.cc
//---------------------------------------------------------------------------//
#include "IntersectSurfaceState.hh"

#include "orange/transform/VariantTransform.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Use the local and global bounding zones to create a better zone.
 *
 * This requires that the local and global zones have been set, and the
 * transform be present.
 */
BoundingZone calc_merged_bzone(IntersectSurfaceState const& iss)
{
    CELER_EXPECT(iss.transform);
    CELER_EXPECT(
        (!iss.local_bzone.exterior && !iss.local_bzone.interior)
        || encloses(iss.local_bzone.exterior, iss.local_bzone.interior));
    CELER_EXPECT(!iss.local_bzone.negated);
    CELER_EXPECT(!iss.global_bzone.negated);

    BoundingZone transformed_local;
    if (iss.local_bzone.interior)
    {
        transformed_local.interior
            = apply_transform(*iss.transform, iss.local_bzone.interior);
    }
    transformed_local.exterior
        = apply_transform(*iss.transform, iss.local_bzone.exterior);
    transformed_local.negated = false;
    return calc_intersection(transformed_local, iss.global_bzone);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
