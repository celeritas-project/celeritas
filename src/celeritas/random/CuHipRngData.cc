//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/CuHipRngData.cc
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

    using RngInit = CuHipRngInitializer;

    // Host-side RNG for creating seeds
    std::mt19937 host_rng(params.seed + stream.get());
    std::uniform_int_distribution<ull_int> sample_uniform_int;

    // Create seeds for device in host memory
    StateCollection<RngInit, Ownership::value, MemSpace::host> host_seeds;
    resize(&host_seeds, size);
    for (RngInit& init : host_seeds[AllItems<RngInit>{}])
    {
        init.seed = sample_uniform_int(host_rng);
    }

    // Resize state data and assign
    resize(&state->rng, size);
    detail::CuHipRngInitData<Ownership::value, M> init_data;
    init_data.seeds = host_seeds;
    detail::rng_state_init(make_ref(*state), make_const_ref(init_data));
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
