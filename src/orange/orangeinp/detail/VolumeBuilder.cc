//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/VolumeBuilder.cc
//---------------------------------------------------------------------------//
#include "VolumeBuilder.hh"

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
 * Add a region to the CSG tree.
 */
NodeId
VolumeBuilder::insert_region(Metadata&& md, Joined&& j, BoundingZone&& bzone)
{
    CELER_DISCARD(md);
    CELER_DISCARD(j);
    CELER_DISCARD(bzone);
    CELER_NOT_IMPLEMENTED("insert_region");
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
