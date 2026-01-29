//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorData.cc
//---------------------------------------------------------------------------//
#include "DetectorData.hh"

#include <iostream>

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Copy hits from host state data to pinned memory.
 *
 * Because both buffer reside host-side, this is just a trivial copy between
 * the buffers.
 */
template<>
void copy_hits<MemSpace::host>(
    DetectorHitOutput* output,
    DetectorStateData<Ownership::reference, MemSpace::host> const& state,
    StreamId /* unused */)
{
    // Trivial copy to pinned memory
    output->hits.resize(state.all_track_hits.size());

    for (auto tid : range(TrackSlotId{state.all_track_hits.size()}))
    {
        output->hits[tid.unchecked_get()] = state.all_track_hits[tid];
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
