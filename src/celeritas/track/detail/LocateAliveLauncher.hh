//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/LocateAliveLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/Secondary.hh"

#include "../SimTrackView.hh"
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
    using ParamsRef = CoreParamsData<Ownership::const_reference, M>;
    using StateRef  = CoreStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION LocateAliveLauncher(const CoreRef<M>& core_data)
        : params_(core_data.params), states_(core_data.states)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
    }

    // Determine which tracks are alive and count secondaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsRef& params_;
    const StateRef&  states_;
};

//---------------------------------------------------------------------------//
/*!
 * Determine which tracks are alive and count secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void LocateAliveLauncher<M>::operator()(ThreadId tid) const
{
    // Count the number of secondaries produced by each track
    size_type    num_secondaries{0};
    SimTrackView sim(states_.sim, tid);

    if (sim.status() != TrackStatus::inactive)
    {
        PhysicsStepView phys(params_.physics, states_.physics, tid);
        for (const auto& secondary : phys.secondaries())
        {
            if (secondary)
            {
                ++num_secondaries;
            }
        }
    }

    if (sim.status() == TrackStatus::alive)
    {
        // The track is alive: mark this track slot as occupied
        states_.init.vacancies[tid] = occupied();
    }
    else if (num_secondaries > 0)
    {
        // The track was killed and produced secondaries: in this case, the
        // empty track slot will be filled with the first secondary. Mark this
        // slot as occupied even though the secondary has not been initialized
        // in it yet, and don't include the first secondary in the count
        states_.init.vacancies[tid] = occupied();
        --num_secondaries;
    }
    else
    {
        // The track is inactive/killed and did not produce secondaries: store
        // the index so it can be used later to initialize a new track
        states_.init.vacancies[tid] = tid.unchecked_get();
    }
    states_.init.secondary_counts[tid] = num_secondaries;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
