//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreState.cc
//---------------------------------------------------------------------------//
#include "CoreState.hh"

#include "CoreParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from CoreParams.
 */
template<MemSpace M>
CoreState<M>::CoreState(CoreParams const& params,
                        StreamId stream_id,
                        size_type num_track_slots)
{
    CELER_VALIDATE(stream_id < params.max_streams(),
                   << "stream ID " << stream_id.unchecked_get()
                   << " is out of range: max streams is "
                   << params.max_streams());
    CELER_VALIDATE(num_track_slots > 0, << "number of track slots is not set");

    states_ = CollectionStateStore<CoreStateData, M>(
        params.host_ref(), stream_id, num_track_slots);
    CELER_ENSURE(states_);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template class CoreState<MemSpace::host>;
template class CoreState<MemSpace::device>;
//---------------------------------------------------------------------------//
}  // namespace celeritas
