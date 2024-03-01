//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildConvexRegion.cc
//---------------------------------------------------------------------------//
#include "BuildConvexRegion.hh"

#include "orange/orangeinp/ConvexRegion.hh"
#include "orange/orangeinp/ConvexSurfaceBuilder.hh"
#include "orange/surf/FaceNamer.hh"

#include "ConvexSurfaceState.hh"
#include "CsgUnitBuilder.hh"
#include "VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build a convex region.
 */
NodeId build_convex_region(VolumeBuilder& vb,
                           std::string&& label,
                           std::string&& face_prefix,
                           ConvexRegionInterface const& region)
{
    // Set input attributes for surface state
    ConvexSurfaceState css;
    css.transform = &vb.local_transform();
    css.object_name = std::move(label);
    css.make_face_name = FaceNamer{std::move(face_prefix)};

    // Construct surfaces
    auto sb = ConvexSurfaceBuilder(&vb.unit_builder(), &css);
    region.build(sb);

    // Intersect the given surfaces to create a new CSG node
    return vb.insert_region(Label{std::move(css.object_name)},
                            Joined{op_and, std::move(css.nodes)},
                            calc_merged_bzone(css));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
