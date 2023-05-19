//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/ProcessPrimariesLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/TrackInitData.hh"

#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
struct ProcessPrimariesLauncher
{
    //// TYPES ////

    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    StatePtr state_;
    Span<Primary const> primaries_;
    CoreStateCounters counters_;

    //// FUNCTIONS ////

    // Create track initializers from primaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primaries.
 */
CELER_FUNCTION void ProcessPrimariesLauncher::operator()(ThreadId tid) const
{
#if CELER_DEVICE_COMPILE
    CELER_EXPECT(tid);
    if (!(tid < primaries_.size()))
    {
        return;
    }
#else
    CELER_EXPECT(tid < primaries_.size());
#endif
    CELER_ASSERT(primaries_.size() <= counters_.num_initializers + tid.get());

    ItemId<TrackInitializer> idx{
        index_after(counters_.num_initializers - primaries_.size(), tid)};
    TrackInitializer& ti = state_->init.initializers[idx];
    Primary const& primary = primaries_[tid.unchecked_get()];

    // Construct a track initializer from a primary particle
    ti.sim.track_id = primary.track_id;
    ti.sim.parent_id = TrackId{};
    ti.sim.event_id = primary.event_id;
    ti.sim.time = primary.time;
    ti.sim.status = TrackStatus::alive;
    ti.geo.pos = primary.position;
    ti.geo.dir = primary.direction;
    ti.particle.particle_id = primary.particle_id;
    ti.particle.energy = primary.energy;

    // Update per-event counter of number of tracks created
    CELER_ASSERT(ti.sim.event_id < state_->init.track_counters.size());
    atomic_add(&state_->init.track_counters[ti.sim.event_id], size_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
