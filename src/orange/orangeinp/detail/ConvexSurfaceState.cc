//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ConvexSurfaceState.cc
//---------------------------------------------------------------------------//
#include "ConvexSurfaceState.hh"

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
BoundingZone calc_merged_bzone(ConvexSurfaceState const& css)
{
    CELER_EXPECT(css.transform);
    CELER_EXPECT(encloses(css.local_bzone.exterior, css.local_bzone.interior));
    CELER_EXPECT(!css.local_bzone.negated);
    CELER_EXPECT(!css.global_bzone.negated);

    BoundingZone transformed_local;
    transformed_local.interior
        = apply_transform(*css.transform, css.local_bzone.interior);
    transformed_local.exterior
        = apply_transform(*css.transform, css.local_bzone.exterior);
    transformed_local.negated = false;
    return calc_intersection(transformed_local, css.global_bzone);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
