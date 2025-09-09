//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/detail/CuHipRngStateInit.cc
//---------------------------------------------------------------------------//
#include "CuHipRngStateInit.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/KernelLauncher.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states from seeds randomly generated on host.
 */
void rng_state_init(HostCRef<CuHipRngParamsData> const& params,
                    HostRef<CuHipRngStateData> const& state,
                    HostCRef<CuHipRngInitData> const& seeds,
                    StreamId)
{
    CELER_EXPECT(state.size() == seeds.size());
    launch_kernel(state.size(), RngSeedExecutor{params, state, seeds});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
