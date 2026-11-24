//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/PrimaryGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/phys/GeneratorCounters.hh"
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
    NativeCRef<DistributionParamsData> distributions;
    ObserverPtr<GeneratorStateCounters const, MemSpace::native> gen_counters;

    //// FUNCTIONS ////

    // Generate optical photons
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 */
CELER_FUNCTION void PrimaryGeneratorExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(data);
    CELER_EXPECT(distributions);
    CELER_EXPECT(gen_counters);

    auto* counters = state->init.counters.data().get();

    // The kernel is launched over all track slots to avoid copying the
    // counters to the host: skip threads beyond the number of vacancies or
    // the number of photons left to generate
    if (!(tid
          < min(counters->num_vacancies, gen_counters->current.num_pending)))
    {
        return;
    }
    // Create the view to the new track to be initialized
    CoreTrackView vacancy{*params, *state, [&] {
                              // Get the vacancy from the back in case there
                              // are more vacancies than photons to generate
                              TrackSlotId idx{
                                  index_before(counters->num_vacancies, tid)};
                              return state->init.vacancies[idx];
                          }()};

    // Generate one primary from the distribution
    CoreTrackView track(*params, *state, TrackSlotId{tid.get()});
    auto rng = track.rng();
    vacancy = PrimaryGenerator(distributions, data)(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
