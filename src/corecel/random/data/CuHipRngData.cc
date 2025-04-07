//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/CuHipRngData.cc
//---------------------------------------------------------------------------//
#include "CuHipRngData.hh"

#include <random>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"

#include "detail/CuHipRngStateInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize and initialize with the seed stored in params.
 */
template<MemSpace M>
void resize(CuHipRngStateData<Ownership::value, M>* state,
            HostCRef<CuHipRngParamsData> const& params,
            StreamId stream,
            size_type size)
{
    CELER_EXPECT(stream);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(M == MemSpace::host || celeritas::device());

    // Host-side RNG for creating seeds
    std::mt19937 host_rng(params.seed + stream.get());
    std::uniform_int_distribution<ull_int> sample_uniform_int;

    // Create seeds for device in host memory
    StateCollection<ull_int, Ownership::value, MemSpace::host> host_seeds;
    resize(&host_seeds, size);
    for (auto& seed : host_seeds[AllItems<ull_int>{}])
    {
        seed = sample_uniform_int(host_rng);
    }

    // Set up params on device to initialize the engine
    HostVal<CuHipRngParamsData> host_data;
    host_data.seed = params.seed;
    CuHipRngParamsData<Ownership::value, M> data;
    data = host_data;

    // Resize state data and assign
    resize(&state->rng, size);
    detail::CuHipRngInitData<Ownership::value, M> init_data;
    init_data.seeds = host_seeds;
    detail::rng_state_init(make_const_ref(data),
                           make_ref(*state),
                           make_const_ref(init_data),
                           stream);
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void resize(HostVal<CuHipRngStateData>*,
                     HostCRef<CuHipRngParamsData> const&,
                     StreamId,
                     size_type);

template void resize(CuHipRngStateData<Ownership::value, MemSpace::device>*,
                     HostCRef<CuHipRngParamsData> const&,
                     StreamId,
                     size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
