//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateInit.cc
//---------------------------------------------------------------------------//
#include "random/detail/RngStateInit.hh"

#include "base/Span.hh"
#include "random/RngData.hh"
#include "random/RngEngine.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states from seeds randomly generated on host.
 */
void rng_state_init(
    const RngStateData<Ownership::reference, MemSpace::host>&      rng,
    const RngInitData<Ownership::const_reference, MemSpace::host>& seeds)
{
    for (auto tid : range(ThreadId{seeds.size()}))
    {
        RngEngine engine(rng, tid);
        engine = seeds.seeds[tid];
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
