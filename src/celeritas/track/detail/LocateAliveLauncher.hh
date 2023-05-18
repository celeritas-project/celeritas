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
struct LocateAliveLauncher
{
    //// TYPES ////

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    ParamsPtr params_;
    StatePtr state_;

    //// FUNCTIONS ////

    // Determine which tracks are alive and count secondaries
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        // The grid size should be equal to the state size and no thread/slot
        // remapping should be performed
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
/*!
 * Determine which tracks are alive and count secondaries.
 */
CELER_FUNCTION void LocateAliveLauncher::operator()(TrackSlotId tid) const
{
#if CELER_DEVICE_COMPILE
    CELER_EXPECT(tid);
    if (!(tid < state_->size()))
    {
        return;
    }
#else
    CELER_EXPECT(tid < state_->size());
#endif

    // Count the number of secondaries produced by each track
    size_type num_secondaries{0};
    SimTrackView sim(params_->sim, state_->sim, tid);

    if (sim.status() != TrackStatus::inactive)
    {
        PhysicsStepView phys(params_->physics, state_->physics, tid);
        for (auto const& secondary : phys.secondaries())
        {
            if (secondary)
            {
                ++num_secondaries;
            }
        }
    }

    state_->init.vacancies[tid] = [&] {
        if (sim.status() == TrackStatus::alive)
        {
            // The track is alive: mark this track slot as occupied
            return occupied();
        }
        else if (num_secondaries > 0)
        {
            // The track was killed and produced secondaries: in this case, the
            // empty track slot will be filled with the first secondary. Mark
            // this slot as occupied even though the secondary has not been
            // initialized in it yet, and don't include the first secondary in
            // the count
            --num_secondaries;
            return occupied();
        }
        else
        {
            // The track is inactive/killed and did not produce secondaries:
            // store the index so it can be used later to initialize a new
            // track
            return tid;
        }
    }();
    state_->init.secondary_counts[tid] = num_secondaries;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
