//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorData.cc
//---------------------------------------------------------------------------//
#include "DetectorData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

template<>
void copy_hits(
    DetectorHitOutput* output,
    DetectorStateData<Ownership::reference, MemSpace::host> const& state)
{
    // Trivial copy to pinned memory
    output->hits.reserve(state.all_track_hits.size());

    for (auto tid : range(TrackSlotId{state.all_track_hits.size()}))
    {
        output->hits[tid.unchecked_get()]
            = state.all_track_hits[tid.unchecked_get()];
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
