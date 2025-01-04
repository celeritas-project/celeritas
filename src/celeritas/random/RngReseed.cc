//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngReseed.cc
//---------------------------------------------------------------------------//
#include "RngReseed.hh"

#include "corecel/sys/KernelLauncher.hh"

#include "detail/RngReseedExecutor.hh"

namespace celeritas
{
#if !defined(__DOXYGEN__) || __DOXYGEN__ > 0x010908
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on host at the start of an event.
 *
 * Each thread's state is initialized using same seed and skipped ahead a
 * different number of subsequences so the sequences on different threads will
 * not have statistically correlated values.
 */
void reseed_rng(HostCRef<RngParamsData> const& params,
                HostRef<RngStateData> const& state,
                StreamId,
                UniqueEventId event_id)
{
    launch_kernel(state.size(),
                  detail::RngReseedExecutor{params, state, event_id});
}

//---------------------------------------------------------------------------//
#endif
}  // namespace celeritas
