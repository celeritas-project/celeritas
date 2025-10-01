//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/RanluxppRngData.cc
//---------------------------------------------------------------------------//
#include "RanluxppRngData.hh"

#include <random>
#include <vector>

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/random/data/RanluxppTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Initialize Ranluxpp states with well-distributed random data
 *
 * This generates pseudorandom, independent starting states for all data in
 * all threads using a 32-bit MT19937 engine.
 */
void initialize_ranluxpp(Span<RanluxppRngState> states,
                         HostCRef<RanluxppRngParamsData> const& params,
                         RanluxppUInt seed,
                         StreamId stream)
{
    // Generate well-distributed seed numbers, including StreamId so that each
    // stream has different starting contribution
    std::vector<std::seed_seq::result_type> host_seeds(1, seed);
    if (stream != StreamId{0})
    {
        host_seeds.push_back(stream.get());
    }
    std::seed_seq seed_seq(host_seeds.begin(), host_seeds.end());

    // Use 32-bit generator to fill initial states
    std::mt19937 rng(seed_seq);
    std::uniform_int_distribution<RanluxppUInt> sample_uniform_int;

    // Initialize state from the random seed
    for (RanluxppRngState& state : states)
    {
        // Sample RNG to get seed for initialization
        RanluxppUInt s = sample_uniform_int(rng);

        // Initialize the state with the given seed
        state.initialize(s, params.kA_2048);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Resize and seed the RNG states
 */
template<MemSpace M>
void resize(RanluxppRngStateData<Ownership::value, M>* state,
            HostCRef<RanluxppRngParamsData> const& params,
            StreamId stream,
            size_type size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params);

    // Create seeds for device in host memory
    HostVal<RanluxppRngStateData> host_state;
    resize(&host_state.state, size);
    initialize_ranluxpp(
        &host_state.state[AllItems<RanluxppState>()], params, stream);

    // Move or copy to input
    if (M == Memspace::host)
    {
        state->state = std::move(host_state.statee);
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
template void resize(HostVal<RanluxppRngStateData>*,
                     HostCRef<RanluxppRngParamsData> const&,
                     StreamId,
                     size_type);

template void resize(RanluxppRngStateData<Ownership::value, MemSpace::device>*,
                     HostCRef<RanluxppRngParamsData> const&,
                     StreamId,
                     size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
