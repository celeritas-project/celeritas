//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BuildConvexRegion.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class ConvexRegionInterface;
namespace detail
{
class VolumeBuilder;
//---------------------------------------------------------------------------//

// Build a convex region
NodeId build_convex_region(VolumeBuilder& vb,
                           std::string&& label,
                           std::string&& face_prefix,
                           ConvexRegionInterface const& region);

//! Build a convex region with no face prefix
inline NodeId build_convex_region(VolumeBuilder& vb,
                                  std::string&& label,
                                  ConvexRegionInterface const& region)
{
    return build_convex_region(vb, std::move(label), {}, region);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
