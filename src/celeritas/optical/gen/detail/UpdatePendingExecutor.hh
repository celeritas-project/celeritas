//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/UpdatePendingExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/optical/CoreTrackView.hh"

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
 * Update the num_pending counter based on the generated photons from buffered
 * optical distribution data.
 */
template<typename CounterType>
struct UpdatePendingExecutor
{
    //// DATA ////

    CounterType num_photons;

    //// FUNCTIONS ////

    // Update number of primaries waiting to be generated
    CELER_FORCEINLINE_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update number of primaries to be generated to include the buffered optical
 * photons.
 */
template<typename CounterType>
CELER_FORCEINLINE_FUNCTION void UpdatePendingExecutor<CounterType>::operator()(
    CoreTrackView& track)
{
    CELER_EXPECT(track.thread_id() == ThreadId{0});  // single thread kernel

    // This executor is called with two possible template values -- a size_type
    // (the typical case) and a pointer to a size_type value stored on device
    // that is produced after running a CUB/hipCUB function
    if constexpr (std::is_pointer_v<CounterType>)
    {
        track.counters().num_pending += static_cast<size_type>(*num_photons);
    }
    else
    {
        track.counters().num_pending += static_cast<size_type>(num_photons);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
