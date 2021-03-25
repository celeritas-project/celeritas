//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoInterface.cc
//---------------------------------------------------------------------------//
#include "GeoInterface.hh"

#include "base/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize particle states in host code.
 */
template<MemSpace M>
void resize(
    GeoStateData<Ownership::value, M>*                               data,
    const GeoParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                        size)
{
    CELER_EXPECT(data);
    CELER_EXPECT(size > 0);

    make_builder(&data->pos).resize(size);
    make_builder(&data->dir).resize(size);
    make_builder(&data->next_step).resize(size);
    data->vgstate.resize(params.max_depth, size);
    data->vgnext.resize(params.max_depth, size);

    CELER_ENSURE(data);
}

// Explicitly instantiate
template void
resize(GeoStateData<Ownership::value, MemSpace::host>*,
       const GeoParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type);
template void
resize(GeoStateData<Ownership::value, MemSpace::device>*,
       const GeoParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type);

//---------------------------------------------------------------------------//
} // namespace celeritas
