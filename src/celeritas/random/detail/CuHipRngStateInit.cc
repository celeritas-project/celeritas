//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CuHipRngStateInit.cc
//---------------------------------------------------------------------------//
#include "random/detail/CuHipRngStateInit.hh"

#include "base/Span.hh"
#include "random/CuHipRngData.hh"
#include "random/CuHipRngEngine.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states from seeds randomly generated on host.
 */
void rng_state_init(
    const CuHipRngStateData<Ownership::reference, MemSpace::host>&      rng,
    const CuHipRngInitData<Ownership::const_reference, MemSpace::host>& seeds)
{
    for (auto tid : range(ThreadId{seeds.size()}))
    {
        CuHipRngEngine engine(rng, tid);
        engine = seeds.seeds[tid];
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
