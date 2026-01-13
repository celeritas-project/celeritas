//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/XorwowRngData.cc
//---------------------------------------------------------------------------//
#include "XorwowRngData.hh"

#include <random>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ScopedProfiling.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Initialize XORWOW states with well-distributed random data.
 *
 * cuRAND and hipRAND implement functions that permute the original (but
 * seemingly arbitrary) seeds given in Marsaglia with 64 bits of seed data. We
 * simply generate pseudorandom, independent starting states for all data in
 * all threads using a 32-bit MT19937 engine.
 */
void initialize_xorwow(Span<XorwowState> state,
                       XorwowSeed const& seed,
                       StreamId stream)
{
    // Seed sequence to generate well-distributed seed numbers, including
    // stream ID to give strings different starting contributions
    std::vector<std::seed_seq::result_type> host_seeds(seed.begin(),
                                                       seed.end());
    if (stream != StreamId{0})
    {
        // For backward compatibility with prior RNG seed, don't modify the
        // seed for the first stream
        host_seeds.push_back(stream.get());
    }
    std::seed_seq seed_seq(host_seeds.begin(), host_seeds.end());

    // 32-bit generator to fill initial states
    std::mt19937 rng(seed_seq);
    std::uniform_int_distribution<XorwowUInt> sample_uniform_int;

    // Fill all seeds with random data. The xorstate is never all
    // zeros, with probability 2^{-320}.
    for (XorwowState& init : state)
    {
        for (XorwowUInt& u : init.xorstate)
        {
            u = sample_uniform_int(rng);
        }
        init.weylstate = sample_uniform_int(rng);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Resize and seed the RNG states.
 */
template<MemSpace M>
void resize(XorwowRngStateData<Ownership::value, M>* state,
            HostCRef<XorwowRngParamsData> const& params,
            StreamId stream,
            size_type size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params);

    ScopedProfiling profile_this{"init-rng"};

    // Create seeds for device in host memory
    HostVal<XorwowRngStateData> host_state;
    resize(&host_state.state, size);

    initialize_xorwow(
        host_state.state[AllItems<XorwowState>{}], params.seed, stream);

    // Move or copy to input
    if constexpr (M == MemSpace::host)
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
template void resize(HostVal<XorwowRngStateData>*,
                     HostCRef<XorwowRngParamsData> const&,
                     StreamId,
                     size_type);

template void resize(XorwowRngStateData<Ownership::value, MemSpace::device>*,
                     HostCRef<XorwowRngParamsData> const&,
                     StreamId,
                     size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
