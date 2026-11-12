//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/UpdateSumExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/phys/GeneratorCounters.hh"
#include "celeritas/track/CoreStateCounters.hh"

#include "../GeneratorData.hh"

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
 * Subtract the number of tracks generated in the step from the cumulative sum.
 */
struct UpdateSumExecutor
{
    //// DATA ////

    RefPtr<CoreStateData, MemSpace::native> state;
    NativeRef<GeneratorStateData> const offload;
    ObserverPtr<GeneratorStateCounters const, MemSpace::native> gen_counters;

    //// FUNCTIONS ////

    // Update the cumulative sum of the number of photons per distribution
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
 * Update the cumulative sum of the number of photons per distribution.
 */
CELER_FUNCTION void UpdateSumExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(state);
    CELER_EXPECT(offload);
    CELER_EXPECT(gen_counters);
    CELER_EXPECT(tid < offload.offsets.size());

    auto const& current = gen_counters->current;
    if (!(tid.get() < current.buffer_size))
    {
        // Launched over an upper bound of the buffer size
        return;
    }

    // The number of photons generated in this step was capped at the number
    // of vacancies in the GeneratorExecutor functor
    auto counters = state->init.counters.data();
    size_type num_generated = min(current.num_pending, counters->num_vacancies);

    auto& offset = offload.offsets[ItemId<size_type>(tid.get())];
    if (offset < num_generated)
    {
        offset = 0;
    }
    else
    {
        offset -= num_generated;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
