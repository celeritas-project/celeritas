//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LocalSurfaceInserter.cc
//---------------------------------------------------------------------------//
#include "LocalSurfaceInserter.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
LocalSurfaceInserter::LocalSurfaceInserter(VecSurface* v,
                                           Tolerance<> const& tol)
    : surfaces_{v}, soft_surface_equal_{tol}
{
    CELER_EXPECT(surfaces_);
    CELER_EXPECT(surfaces_->empty());
}

//---------------------------------------------------------------------------//
/*!
 * Look for duplicate surfaces and store equivalency relationship.
 */
LocalSurfaceId
LocalSurfaceInserter::merge_impl(LocalSurfaceId source, LocalSurfaceId target)
{
    CELER_EXPECT(source < surfaces_->size());
    CELER_EXPECT(target < source);

    if (auto iter = merged_.find(target); iter != merged_.end())
    {
        // Surface was already merged with another: chain them
        target = iter->second;
        CELER_ENSURE(target < source);
    }

    merged_.emplace(source, target);
    return target;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
