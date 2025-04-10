//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildIntersectRegion.cc
//---------------------------------------------------------------------------//
#include "BuildIntersectRegion.hh"

#include "orange/surf/FaceNamer.hh"

#include "CsgUnitBuilder.hh"
#include "IntersectSurfaceState.hh"
#include "VolumeBuilder.hh"
#include "../IntersectRegion.hh"
#include "../IntersectSurfaceBuilder.hh"

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
    IntersectSurfaceState iss;
    iss.transform = &vb.local_transform();
    iss.object_name = std::move(label);
    iss.make_face_name = FaceNamer{std::string{face_prefix}};

    // Construct surfaces
    auto sb = IntersectSurfaceBuilder(&vb.unit_builder(), &iss);
    region.build(sb);

    // Intersect the given surfaces to create a new CSG node
    return vb.insert_region(
        Label{std::move(iss.object_name), std::move(face_prefix)},
        Joined{op_and, std::move(iss.nodes)},
        calc_merged_bzone(iss));
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
