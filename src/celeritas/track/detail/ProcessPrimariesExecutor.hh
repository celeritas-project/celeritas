//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/ProcessPrimariesExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/random/engine/RanluxppRngEngine.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "corecel/random/engine/SplitMix64.hh"
#include "corecel/random/engine/XorwowRngEngine.hh"
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

  private:
    // Fill the RNG state initializer for the Xorwow engine
    inline CELER_FUNCTION void fillRngStateInitializer(
        unsigned int seed,
        unsigned int event_id,
        unsigned int geant_track_id,
        unsigned int geant_step_id,
        XorwowRngEngine::RngStateInitializer_t& rng_init) const;

    // Fill the RNG state initializer for the Ranluxpp engine
    void fillRngStateInitializer(
        unsigned int seed,
        unsigned int event_id,
        unsigned int geant_track_id,
        unsigned int geant_step_id,
        RanluxppRngEngine::RngStateInitializer_t& rng_init) const;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primaries.
 */
CELER_FUNCTION void ProcessPrimariesExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(tid < primaries.size());
    auto counters = state->init.counters.data().get();
    CELER_EXPECT(primaries.size() <= counters->num_initializers + tid.get());

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
#if CELERITAS_RESEED == CELERITAS_RESEED_TRACK
    this->fillRngStateInitializer(params->rng.get_seed(),
                                  ti.sim.event_id.get(),
                                  primary.geant_track_id,
                                  primary.geant_step_count,
                                  ti.rng);
#endif

    // Store the initializer
    size_type idx = counters->num_initializers - primaries.size() + tid.get();
    state->init.initializers[ItemId<TrackInitializer>(idx)] = ti;
}

//---------------------------------------------------------------------------//
/*!
 * Fill a XorwowRngEngine state initializer
 */
CELER_FUNCTION void ProcessPrimariesExecutor::fillRngStateInitializer(
    unsigned int seed,
    unsigned int event_id,
    unsigned int geant_track_id,
    unsigned int geant_step_id,
    XorwowRngEngine::RngStateInitializer_t& rng_init) const
{
    // Initialize SplitMix64 with the seed XORed with the track id
    SplitMix64 rng(seed ^ geant_track_id);

    // Fill first two state values
    std::uint64_t val = rng();
    rng_init.xorstate[0] = static_cast<XorwowUInt>(val);
    rng_init.xorstate[1] = static_cast<XorwowUInt>(val >> 32);

    // XOR with event id
    rng.xor_state(event_id);
    val = rng();
    rng_init.xorstate[2] = static_cast<XorwowUInt>(val);
    rng_init.xorstate[3] = static_cast<XorwowUInt>(val >> 32);

    // XOR with step id
    rng.xor_state(geant_step_id);
    val = rng();
    rng_init.xorstate[4] = static_cast<XorwowUInt>(val);
    rng_init.weylstate = static_cast<XorwowUInt>(val >> 32);
}

//---------------------------------------------------------------------------//
/*!
 * Fill a Ranluxpp state initializer
 */
CELER_FUNCTION
void ProcessPrimariesExecutor::fillRngStateInitializer(
    unsigned int seed,
    unsigned int event_id,
    unsigned int geant_track_id,
    unsigned int geant_step_id,
    RanluxppRngEngine::RngStateInitializer_t& rng_init) const
{
    // Initialize SplitMix64 with the seed XORed with the track id
    SplitMix64 rng(seed ^ geant_track_id);

    // Fill first three state values
    rng_init.value.number[0] = rng();
    rng_init.value.number[1] = rng();
    rng_init.value.number[2] = rng();

    // XOR with event id and fill next three values
    rng.xor_state(event_id);
    rng_init.value.number[3] = rng();
    rng_init.value.number[4] = rng();
    rng_init.value.number[5] = rng();

    // XOR with step id and fill next three values
    rng.xor_state(geant_step_id);
    rng_init.value.number[6] = rng();
    rng_init.value.number[7] = rng();
    rng_init.value.number[8] = rng();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
