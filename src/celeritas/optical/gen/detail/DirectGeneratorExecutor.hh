//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/DirectGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/phys/GeneratorCounters.hh"
#include "celeritas/track/CoreStateCounters.hh"
#include "celeritas/track/Utils.hh"

#include "../DirectGeneratorData.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Directly initialize photons.
 */
struct DirectGeneratorExecutor
{
    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    NativeRef<DirectGeneratorStateData> const data;
    ObserverPtr<GeneratorStateCounters const, MemSpace::native> gen_counters;

    // Initialize optical photons
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize optical photons.
 */
CELER_FUNCTION void DirectGeneratorExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(gen_counters);

    auto counters = state->init.counters.data();
    size_type const num_pending = gen_counters->current.num_pending;

    // The kernel is launched over all track slots to avoid copying the
    // counters to the host: skip threads beyond the number of vacancies or
    // the number of photons left to generate
    if (!(tid < min(counters->num_vacancies, num_pending)))
    {
        return;
    }

    // Create view to new track to be initialized
    CoreTrackView vacancy(*params, *state, [&] {
        TrackSlotId idx{index_before(counters->num_vacancies, tid)};
        return state->init.vacancies[idx];
    }());

    // Get initializer from the back
    TrackInitializer const& init = data.initializers[ItemId<TrackInitializer>(
        index_before(num_pending, tid))];

    // Initialize track
    vacancy = init;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
