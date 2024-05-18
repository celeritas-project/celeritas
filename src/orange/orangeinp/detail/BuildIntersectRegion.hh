//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildIntersectRegion.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class IntersectRegionInterface;
namespace detail
{
class VolumeBuilder;
//---------------------------------------------------------------------------//

// Build a intersect region
NodeId build_intersect_region(VolumeBuilder& vb,
                              std::string&& label,
                              std::string&& face_prefix,
                              IntersectRegionInterface const& region);

//! Build a intersect region with no face prefix
inline NodeId build_intersect_region(VolumeBuilder& vb,
                                     std::string&& label,
                                     IntersectRegionInterface const& region)
{
    return build_intersect_region(vb, std::move(label), {}, region);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
