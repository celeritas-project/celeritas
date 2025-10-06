//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/PrimaryGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/track/CoreStateCounters.hh"
#include "celeritas/track/Utils.hh"

#include "../GeneratorData.hh"
#include "../PrimaryGenerator.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate primary optical photons from distributions.
 */
struct PrimaryGeneratorExecutor
{
    //// DATA ////

    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    PrimaryDistributionData data;
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Generate optical photons
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 */
CELER_FUNCTION void PrimaryGeneratorExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(data);

    CoreTrackView track(*params, *state, tid);

    // Create the view to the new track to be initialized
    CoreTrackView vacancy{*params, *state, [&] {
                              // Get the vacancy from the back in case there
                              // are more vacancies than photons to generate
                              TrackSlotId idx{index_before(
                                  counters.num_vacancies, ThreadId(tid.get()))};
                              return state->init.vacancies[idx];
                          }()};

    // Generate one primary from the distribution
    auto rng = track.rng();
    vacancy = PrimaryGenerator(data)(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
