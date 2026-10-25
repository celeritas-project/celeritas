//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/ProcessPrimariesExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/random/engine/InitializeRngState.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/Primary.hh"

#include "../SimData.hh"
#include "../TrackInitData.hh"
#include "../Utils.hh"

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

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    ParamsPtr params;
    StatePtr state;

    Span<Primary const> primaries;

    //// FUNCTIONS ////

    // Create track initializers from primaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primaries.
 *
 * When reseeding each track, the RNG state is initialized from the primary ID
 * and the primary's unique event ID, falling back to its local event ID if the
 * unique ID is unset.
 *
 * \todo Revisit event IDs and seeding together with the supported
 * threading/tasking models: the local event ID could be combined with the
 * stream ID, and the RNG could be seeded from the unique event ID and the
 * Geant4 track ID (e.g., hashed into 64 bits or used as a 128-bit seed) rather
 * than from event IDs read from input files.
 */
CELER_FUNCTION void ProcessPrimariesExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(tid < primaries.size() || tid < state->size());

    size_type num_initializers
        = state->init.counters.data().get()->num_initializers;

    // Only state.size() threads participate in this grid-stride loop.
    // Additional threads may be launched to process a larger set of primaries:
    // allowing them into the loop would cause overlapping writes to the same
    // initializers.
    if (tid < state->size())
    {
        for (size_type i = tid.get(); i < num_initializers; i += state->size())
        {
            // New primaries can delay existing initializers, so invalidate
            // geometry parent IDs that assume initialization in the next step
            state->init.initializers[ItemId<TrackInitializer>(i)].geo.parent
                = {};
        }
    }

    if (!(tid < primaries.size()))
    {
        return;
    }

    Primary const& primary = primaries[tid.unchecked_get()];

    // Construct a track initializer from a primary particle
    TrackInitializer ti;
    ti.sim.track_id
        = make_track_id(params->init, state->init, primary.event_id);
    ti.sim.primary_id = primary.primary_id;
    ti.sim.event_id = primary.event_id;
    ti.sim.time = primary.time;
    ti.sim.weight = primary.weight;
    ti.geo.pos = primary.position;
    ti.geo.dir = primary.direction;
    ti.particle.particle_id = primary.particle_id;
    ti.particle.energy = primary.energy;

    // Set the RNG state initializer appropriately dispatched on RNG type
    if constexpr (CELERITAS_RESEED == CELERITAS_RESEED_TRACK)
    {
        std::uint64_t event_key = primary.unique_event_id
                                      ? primary.unique_event_id.get()
                                      : primary.event_id.get();
        celeritas::initialize_rng_state(
            params->rng.seed, event_key, ti.sim.primary_id.get(), ti.rng);
    }

    // Store the initializer
    size_type idx = num_initializers + tid.get();
    state->init.initializers[ItemId<TrackInitializer>(idx)] = ti;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
