//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/VolumeBuilder.cc
//---------------------------------------------------------------------------//
#include "VolumeBuilder.hh"

#include "corecel/Assert.hh"

#include "BoundingZone.hh"
#include "CsgUnitBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with unit builder.
 */
VolumeBuilder::VolumeBuilder(CsgUnitBuilder* ub) : ub_{ub}
{
    CELER_EXPECT(ub_);
    ub_->insert_transform(NoTransformation{});
}

//---------------------------------------------------------------------------//
/*!
 * Add a region to the CSG tree.
 */
NodeId
VolumeBuilder::insert_region(Metadata&& md, Joined&& j, BoundingZone&& bzone)
{
    auto node_id = ub_->insert_csg(std::move(j)).first;
    ub_->insert_region(node_id, std::move(bzone), TransformId{0});

    // Always add metadata
    ub_->insert_md(node_id, std::move(md));

    return node_id;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
