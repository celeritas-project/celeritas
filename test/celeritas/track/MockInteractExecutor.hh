//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "MockInteractData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct MockInteractExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<MockInteractData> data;
};

//---------------------------------------------------------------------------//
/*!
 * Apply a mock interaction to the current track.
 */
CELER_FUNCTION void MockInteractExecutor::operator()(CoreTrackView const& track)
{
    auto sim = track.sim();
    CELER_ASSERT(sim.status() == TrackStatus::alive);
    if (!data.alive[track.track_slot_id()])
    {
        // Kill the particle
        sim.status(TrackStatus::killed);
    }

    // Create secondaries
    auto phys_step = track.physics_step();
    size_type num_secondaries = data.num_secondaries[track.track_slot_id()];
    if (num_secondaries > 0)
    {
        auto allocate_secondaries = phys_step.make_secondary_allocator();
        Secondary* allocated = allocate_secondaries(num_secondaries);
        CELER_ASSERT(allocated);

        Span<Secondary> secondaries{allocated, num_secondaries};
        for (auto& secondary : secondaries)
        {
            secondary.particle_id = ParticleId(0);
            secondary.energy = units::MevEnergy(5.);
            secondary.direction = {1., 0., 0.};
        }

        // Save secondaries
        phys_step.secondaries(secondaries);
    }
    else
    {
        // Clear secondaries
        phys_step.secondaries({});
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
