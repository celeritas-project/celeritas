//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/MiniStepTestBase.cc
//---------------------------------------------------------------------------//
#include "MiniStepTestBase.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"
#include "celeritas/track/TrackInitData.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/TrackInitUtils.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void MiniStepTestBase::init(const Input& inp, size_type num_tracks)
{
    CELER_EXPECT(inp);
    CELER_EXPECT(num_tracks > 0);

    // Create states
    states_ = CollectionStateStore<CoreStateData, MemSpace::host>{
        this->core()->host_ref(), num_tracks};
    core_ref_.params = this->core()->host_ref();
    core_ref_.states = states_.ref();
    CELER_ASSERT(core_ref_);

    {
        // Create primary from input (separate event IDs)
        Primary p;
        p.particle_id = inp.particle_id;
        p.energy      = inp.energy;
        p.position    = inp.position;
        p.direction   = inp.direction;
        p.time        = inp.time;

        TrackInitParams::Input inp;
        inp.primaries.assign(num_tracks, p);
        inp.capacity = num_tracks;
        for (auto i : range(num_tracks))
        {
            inp.primaries[i].event_id = EventId{i};
            inp.primaries[i].track_id = TrackId{i};
        }

        // Primary -> track initializer -> track
        TrackInitParams init_params{std::move(inp)};
        TrackInitStateData<Ownership::value, MemSpace::host> init_states;
        resize(&init_states, init_params.host_ref(), num_tracks);
        extend_from_primaries(init_params.host_ref(), &init_states);
        initialize_tracks(core_ref_, &init_states);
    }

    // Set remaining MFP
    for (auto tid : range(ThreadId{num_tracks}))
    {
        CoreTrackView track{core_ref_.params, core_ref_.states, tid};
        auto          phys = track.make_physics_view();
        phys.interaction_mfp(inp.phys_mfp);
    }
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
