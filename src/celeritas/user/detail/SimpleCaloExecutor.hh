//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SimpleCaloExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Types.hh"
#include "corecel/math/Atomics.hh"

#include "../SimpleCaloData.hh"
#include "../StepData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Help gather detector hits in parallel.
 *
 * We do not remap any threads, so the track slot ID should be the thread ID.
 */
struct SimpleCaloExecutor
{
    NativeRef<StepStateData> const& step;
    NativeRef<SimpleCaloStateData>& calo;

    inline CELER_FUNCTION void operator()(TrackSlotId tid);
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid)
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Accumulate detector hits on each thread.
 */
CELER_FUNCTION void SimpleCaloExecutor::operator()(TrackSlotId tid)
{
    CELER_EXPECT(tid < step.data.detector.size());
    CELER_EXPECT(!step.data.energy_deposition.empty());

    DetectorId det = step.data.detector[tid];
    if (!det)
    {
        // No energy deposition or inactive track
        return;
    }

    static_assert(
        std::is_same_v<NativeRef<StepStateDataImpl>::Energy::unit_type,
                       NativeRef<SimpleCaloStateData>::EnergyUnits>);
    real_type edep = step.data.energy_deposition[tid].value();
    CELER_ASSERT(edep > 0);
    CELER_ASSERT(det < calo.energy_deposition.size());
    atomic_add(&calo.energy_deposition[det], edep);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
