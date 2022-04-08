//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngData.cc
//---------------------------------------------------------------------------//
#include "RngData.hh"

#include <random>
#include "random/detail/RngStateInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize and initialize with the seed stored in params.
 */
template<MemSpace M>
void
resize(RngStateData<Ownership::value, M>*                               state,
       const RngParamsData<Ownership::const_reference, MemSpace::host>& params,
       size_type                                                        size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(M == MemSpace::host || celeritas::device());

    using RngInit = RngInitializer<M>;

    // Host-side RNG for creating seeds
    std::mt19937                           host_rng(params.seed);
    std::uniform_int_distribution<ull_int> sample_uniform_int;

    // Create seeds for device in host memory
    StateCollection<RngInit, Ownership::value, MemSpace::host> host_seeds;
    make_builder(&host_seeds).resize(size);
    for (RngInit& init : host_seeds[AllItems<RngInit>{}])
    {
        init.seed = sample_uniform_int(host_rng);
    }

    // Resize state data and assign
    make_builder(&state->rng).resize(size);
    detail::RngInitData<Ownership::value, M> init_data;
    init_data.seeds = host_seeds;
    detail::rng_state_init(make_ref(*state), make_const_ref(init_data));
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void
resize(RngStateData<Ownership::value, MemSpace::host>*,
       const RngParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type);

template void
resize(RngStateData<Ownership::value, MemSpace::device>*,
       const RngParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type);

//---------------------------------------------------------------------------//
} // namespace celeritas
