//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildIntersectRegion.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <string_view>

#include "../CsgTypes.hh"

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
[[nodiscard]] NodeId
build_intersect_region(VolumeBuilder& vb,
                       std::string&& label,
                       std::string&& face_prefix,
                       IntersectRegionInterface const& region);

//! Build a intersect region with no face prefix
[[nodiscard]] inline NodeId
build_intersect_region(VolumeBuilder& vb,
                       std::string&& label,
                       IntersectRegionInterface const& region)
{
    return build_intersect_region(vb, std::move(label), {}, region);
}

//! Build a intersect region using a string view
[[nodiscard]] inline NodeId
build_intersect_region(VolumeBuilder& vb,
                       std::string_view label,
                       std::string&& face_prefix,
                       IntersectRegionInterface const& region)
{
    return build_intersect_region(
        vb, std::string{label}, std::move(face_prefix), region);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
