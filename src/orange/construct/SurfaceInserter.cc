//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceInserter.cc
//---------------------------------------------------------------------------//
#include "SurfaceInserter.hh"

#include "base/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to empty surfaces.
 */
SurfaceInserter::SurfaceInserter(Data* surfaces) : surface_data_(surfaces)
{
    CELER_EXPECT(surface_data_ && surface_data_->types.empty()
                 && surface_data_->offsets.empty()
                 && surface_data_->reals.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Insert a generic surface.
 */
SurfaceId SurfaceInserter::operator()(GenericSurfaceRef generic_surf)
{
    CELER_EXPECT(generic_surf);

    // TODO: surface deduplication goes here

    auto types   = make_builder(&surface_data_->types);
    auto offsets = make_builder(&surface_data_->offsets);
    auto reals   = make_builder(&surface_data_->reals);

    SurfaceId::size_type new_id = types.size();
    types.push_back(generic_surf.type);
    offsets.push_back(OpaqueId<real_type>(reals.size()));
    reals.insert_back(generic_surf.data.begin(), generic_surf.data.end());

    CELER_ENSURE(types.size() == offsets.size());
    return SurfaceId{new_id};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
