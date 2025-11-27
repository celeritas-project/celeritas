//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/detail/RanluxppRngStateInit.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/random/data/RanluxppRngData.hh"
#include "corecel/random/engine/RanluxppRngEngine.hh"

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Initialize the given track slot.
 */
struct RanluxppRngSeedExecutor
{
    NativeCRef<RanluxppRngParamsData> const params;
    NativeRef<RanluxppRngStateData> const state;

    //! Initialize the given track slot.
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const
    {
        CELER_EXPECT(tid < state.size());
        RanluxppRngEngine rng(params, state, tid);

        // Hard-code offset to 0
        RanluxppUInt offset = 0;
        rng = {params.seed, tid.unchecked_get(), offset};
    }

    //! Initialize from the given thread.
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
// Initialize the RNG state on host/device
void ranlux_state_init(DeviceCRef<RanluxppRngParamsData> const& params,
                       DeviceRef<RanluxppRngStateData> const& state,
                       StreamId stream);

void ranlux_state_init(HostCRef<RanluxppRngParamsData> const& params,
                       HostRef<RanluxppRngStateData> const& state,
                       StreamId);

#if !CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device.
 */
inline void ranlux_state_init(DeviceCRef<RanluxppRngParamsData> const&,
                              DeviceRef<RanluxppRngStateData> const&,
                              StreamId)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
