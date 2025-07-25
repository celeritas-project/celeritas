//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/UpdateSumExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

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

    NativeRef<GeneratorStateData> const offload;
    size_type num_gen{};

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
    CELER_EXPECT(offload);
    CELER_EXPECT(num_gen > 0);
    CELER_EXPECT(tid < offload.offsets.size());

    auto& offset = offload.offsets[ItemId<size_type>(tid.get())];
    if (offset < num_gen)
    {
        offset = 0;
    }
    else
    {
        offset -= num_gen;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
