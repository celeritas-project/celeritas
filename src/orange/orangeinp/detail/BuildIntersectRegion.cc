//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildIntersectRegion.cc
//---------------------------------------------------------------------------//
#include "BuildIntersectRegion.hh"

#include "orange/orangeinp/IntersectRegion.hh"
#include "orange/orangeinp/IntersectSurfaceBuilder.hh"
#include "orange/surf/FaceNamer.hh"

#include "CsgUnitBuilder.hh"
#include "IntersectSurfaceState.hh"
#include "VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build an intersect region.
 */
NodeId build_intersect_region(VolumeBuilder& vb,
                              std::string&& label,
                              std::string&& face_prefix,
                              IntersectRegionInterface const& region)
{
    // Set input attributes for surface state
    IntersectSurfaceState css;
    css.transform = &vb.local_transform();
    css.object_name = std::move(label);
    css.make_face_name = FaceNamer{std::string{face_prefix}};

    // Construct surfaces
    auto sb = IntersectSurfaceBuilder(&vb.unit_builder(), &css);
    region.build(sb);

    // Intersect the given surfaces to create a new CSG node
    return vb.insert_region(
        Label{std::move(css.object_name), std::move(face_prefix)},
        Joined{op_and, std::move(css.nodes)},
        calc_merged_bzone(css));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
