//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngData.cc
//---------------------------------------------------------------------------//
#include "XorwowRngData.hh"

#include <random>

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize and initialize with the seed stored in params.
 */
template<MemSpace M>
void resize(XorwowRngStateData<Ownership::value, M>*          state,
            const ::celeritas::HostCRef<XorwowRngParamsData>& params,
            size_type                                         size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params);

    using uint_t = XorwowState::uint_t;

    // Seed sequence to generate well-distributed seed numbers
    std::seed_seq seeds(params.seed.begin(), params.seed.end());
    // 32-bit generator to fill initial states
    std::mt19937                          rng(seeds);
    std::uniform_int_distribution<uint_t> sample_uniform_int;

    // Create seeds for device in host memory
    ::celeritas::HostVal<XorwowRngStateData> host_state;
    resize(&host_state.state, size);

    // Fill all seeds with random data. The xorstate is never all
    // zeros, with probability 2^{-320}.
    for (XorwowState& init : host_state.state[AllItems<XorwowState>{}])
    {
        for (uint_t& u : init.xorstate)
        {
            u = sample_uniform_int(rng);
        }
        init.weylstate = sample_uniform_int(rng);
    }

    // Move or copy to input
    if (M == MemSpace::host)
    {
        state->state = std::move(host_state.state);
    }
    else
    {
        *state = host_state;
    }

    CELER_ENSURE(*state);
    CELER_ENSURE(state->size() == size);
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void resize(::celeritas::HostVal<XorwowRngStateData>*,
                     const ::celeritas::HostCRef<XorwowRngParamsData>&,
                     size_type);

template void resize(XorwowRngStateData<Ownership::value, MemSpace::device>*,
                     const ::celeritas::HostCRef<XorwowRngParamsData>&,
                     size_type);

//---------------------------------------------------------------------------//
} // namespace celeritas
