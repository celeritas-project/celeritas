//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleCaloData.cc
//---------------------------------------------------------------------------//
#include "SimpleCaloData.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize based on the number of detectors.
 */
template<MemSpace M>
void resize(SimpleCaloStateData<Ownership::value, M>* state,
            HostCRef<SimpleCaloParamsData> const& params,
            StreamId,
            size_type num_track_slots)
{
    CELER_EXPECT(params);
    resize(&state->energy_deposition, params.num_detectors);
    fill(real_type(0), &state->energy_deposition);
    state->num_track_slots = num_track_slots;
    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//

template void resize(SimpleCaloStateData<Ownership::value, MemSpace::host>*,
                     HostCRef<SimpleCaloParamsData> const&,
                     StreamId,
                     size_type);
template void resize(SimpleCaloStateData<Ownership::value, MemSpace::device>*,
                     HostCRef<SimpleCaloParamsData> const&,
                     StreamId,
                     size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
