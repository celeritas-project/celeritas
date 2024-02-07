//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/ProcessPrimariesExecutor.hh
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

#include "Utils.hh"
#include "../SimData.hh"
#include "../TrackInitData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
struct ProcessPrimariesExecutor
{
    //// TYPES ////

    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    StatePtr state;
    Span<Primary const> primaries;
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Create track initializers from primaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primaries.
 */
CELER_FUNCTION void ProcessPrimariesExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(tid < primaries.size());
    CELER_EXPECT(primaries.size() <= counters.num_initializers + tid.get());

    ItemId<TrackInitializer> idx{
        index_after(counters.num_initializers - primaries.size(), tid)};
    TrackInitializer& ti = state->init.initializers[idx];
    Primary const& primary = primaries[tid.unchecked_get()];

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
    CELER_ASSERT(ti.sim.event_id < state->init.track_counters.size());
    atomic_add(&state->init.track_counters[ti.sim.event_id], size_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
