//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/CuHipRngData.cc
//---------------------------------------------------------------------------//
#include "CuHipRngData.hh"

#include <random>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"

#include "detail/CuHipRngStateInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize and initialize with the seed stored in params.
 */
template<MemSpace M>
void resize(
    CuHipRngStateData<Ownership::value, M>* state,
    const CuHipRngParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                             size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(M == MemSpace::host || celeritas::device());

    using RngInit = CuHipRngInitializer;

    // Host-side RNG for creating seeds
    std::mt19937                           host_rng(params.seed);
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
template void
resize(CuHipRngStateData<Ownership::value, MemSpace::host>*,
       const CuHipRngParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type);

template void
resize(CuHipRngStateData<Ownership::value, MemSpace::device>*,
       const CuHipRngParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type);

//---------------------------------------------------------------------------//
} // namespace celeritas
