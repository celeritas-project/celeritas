//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/detail/CuHipRngStateInit.cc
//---------------------------------------------------------------------------//
#include "CuHipRngStateInit.hh"

#include "corecel/cont/Range.hh"
#include "corecel/sys/ThreadId.hh"

#include "../CuHipRngData.hh"
#include "../CuHipRngEngine.hh"

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
                    HostCRef<CuHipRngInitData> const& seeds)
{
    for (auto tid : range(TrackSlotId{seeds.size()}))
    {
        CuHipRngInitializer init;
        init.seed = seeds.seeds[tid];
        CuHipRngEngine engine(params, state, tid);
        engine = init;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
