//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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
    //! \name Type aliases
    using ParamsRef = CoreParamsData<Ownership::const_reference, M>;
    using StateRef = CoreStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION
    LocateAliveLauncher(ParamsRef const& params, StateRef const& states)
        : params_(params), states_(states)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
        // Vacancies should have been resized to be the number of track slots
        CELER_EXPECT(states_.init.vacancies.size() == states_.size());
    }

    // Determine which tracks are alive and count secondaries
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        // The grid size should be equal to the state size and no thread/slot
        // remapping should be performed
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }

  private:
    ParamsRef const& params_;
    StateRef const& states_;
};

//---------------------------------------------------------------------------//
/*!
 * Determine which tracks are alive and count secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void LocateAliveLauncher<M>::operator()(TrackSlotId tid) const
{
    // Count the number of secondaries produced by each track
    size_type num_secondaries{0};
    SimTrackView sim(params_.sim, states_.sim, tid);

    if (sim.status() != TrackStatus::inactive)
    {
        PhysicsStepView phys(params_.physics, states_.physics, tid);
        for (auto const& secondary : phys.secondaries())
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
        states_.init.vacancies[tid.unchecked_get()] = occupied();
    }
    else if (num_secondaries > 0)
    {
        // The track was killed and produced secondaries: in this case, the
        // empty track slot will be filled with the first secondary. Mark this
        // slot as occupied even though the secondary has not been initialized
        // in it yet, and don't include the first secondary in the count
        states_.init.vacancies[tid.unchecked_get()] = occupied();
        --num_secondaries;
    }
    else
    {
        // The track is inactive/killed and did not produce secondaries: store
        // the index so it can be used later to initialize a new track
        states_.init.vacancies[tid.unchecked_get()] = tid;
    }

    states_.init.secondary_counts[tid] = num_secondaries;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
