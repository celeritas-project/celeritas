//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/DirectGeneratorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"
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
    CoreStateCounters counters;

    // Initialize optical photons
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
 * Initialize optical photons.
 */
CELER_FUNCTION void DirectGeneratorExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    // Create view to new track to be initialized
    CoreTrackView vacancy(*params, *state, [&] {
        TrackSlotId idx{
            index_before(counters.num_vacancies, ThreadId(tid.get()))};
        return state->init.vacancies[idx];
    }());

    // Get initializer from the back
    TrackInitializer const& init = data.initializers[ItemId<TrackInitializer>(
        index_before(counters.num_pending, ThreadId(tid.get())))];

    // Initialize track
    vacancy = init;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
