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

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
template<MemSpace M>
class ProcessPrimariesLauncher
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = CoreParamsData<Ownership::const_reference, M>;
    using StateRef = CoreStateData<Ownership::reference, M>;
    using TrackInitStateRef = TrackInitStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION ProcessPrimariesLauncher(ParamsRef const&,
                                            StateRef const& states,
                                            Span<Primary const> primaries)
        : data_(states.init), primaries_(primaries)
    {
        CELER_EXPECT(data_);
    }

    // Create track initializers from primaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    TrackInitStateRef const& data_;
    Span<Primary const> primaries_;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primaries.
 */
template<MemSpace M>
CELER_FUNCTION void ProcessPrimariesLauncher<M>::operator()(ThreadId tid) const
{
    Primary const& primary = primaries_[tid.get()];

    CELER_ASSERT(primaries_.size() <= data_.initializers.size() + tid.get());
    TrackInitializer& ti = data_.initializers[data_.initializers.size()
                                              - primaries_.size() + tid.get()];

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
    CELER_ASSERT(ti.sim.event_id < data_.track_counters.size());
    atomic_add(&data_.track_counters[ti.sim.event_id], size_type{1});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
