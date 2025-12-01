//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/detail/RanluxppRngStateInit.cc
//---------------------------------------------------------------------------//
#include "RanluxppRngStateInit.hh"

#include "corecel/sys/KernelLauncher.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states from user defined seed on host.
 */
void ranlux_state_init(HostCRef<RanluxppRngParamsData> const& params,
                       HostRef<RanluxppRngStateData> const& state,
                       StreamId)
{
    launch_kernel(state.size(), RanluxppRngSeedExecutor{params, state});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
