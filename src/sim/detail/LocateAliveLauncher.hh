//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LocateAliveLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Atomics.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "sim/TrackData.hh"
#include "sim/TrackInitData.hh"
#include "sim/SimTrackView.hh"
#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Determine which tracks are alive and count secondaries.
 *
 * This finds empty slots in the track vector and counts the number of
 * secondaries created in each interaction. If the track was killed and
 * produced secondaries, the empty track slot is filled with the first
 * secondary.
 */
template<MemSpace M>
class LocateAliveLauncher
{
  public:
    //!@{
    //! Type aliases
    using ParamsDataRef         = ParamsData<Ownership::const_reference, M>;
    using StateDataRef          = StateData<Ownership::reference, M>;
    using TrackInitStateDataRef = TrackInitStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION LocateAliveLauncher(const ParamsDataRef&         params,
                                       const StateDataRef&          states,
                                       const TrackInitStateDataRef& data)
        : params_(params), states_(states), data_(data)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
        CELER_EXPECT(data_);
    }

    // Determine which tracks are alive and count secondaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsDataRef&         params_;
    const StateDataRef&          states_;
    const TrackInitStateDataRef& data_;
};

//---------------------------------------------------------------------------//
/*!
 * Determine which tracks are alive and count secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void LocateAliveLauncher<M>::operator()(ThreadId tid) const
{
    // Count how many secondaries survived cutoffs for each track
    data_.secondary_counts[tid] = 0;
    for (const auto& secondary : states_.interactions[tid].secondaries)
    {
        if (secondary)
        {
            ++data_.secondary_counts[tid];
        }
    }

    SimTrackView sim(states_.sim, tid);
    if (sim.alive())
    {
        // The track is alive: mark this track slot as occupied
        data_.vacancies[tid] = flag_id();
    }
    else if (data_.secondary_counts[tid] > 0)
    {
        // The track was killed and produced secondaries: in this case, the
        // empty track slot will be filled with the first secondary. Mark this
        // slot as occupied even though the secondary has not been initialized
        // in it yet, and don't include the first secondary in the count
        data_.vacancies[tid] = flag_id();
        --data_.secondary_counts[tid];
    }
    else
    {
        // The track was killed and did not produce secondaries: store the
        // index so it can be used later to initialize a new track
        data_.vacancies[tid] = tid.get();
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
