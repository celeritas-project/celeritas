//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngInterface.cc
//---------------------------------------------------------------------------//
#include "RngInterface.hh"

#include <random>
#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"
#include "comm/Device.hh"
#include "detail/RngStateInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize and initialize with the seed stored in params.
 */
void resize(
    RngStateData<Ownership::value, MemSpace::device>*                state,
    const RngParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                        size)
{
    CELER_EXPECT(size > 0);
    CELER_EXPECT(celeritas::device());

    using RngInit = RngInitializer<MemSpace::device>;

    // Host-side RNG for seeding device RNG
    std::mt19937                           host_rng(params.seed);
    std::uniform_int_distribution<ull_int> sample_uniform_int;

    // Create seeds for device in host memory
    StateCollection<RngInit, Ownership::value, MemSpace::host> host_seeds;
    make_builder(&host_seeds).resize(size);
    for (RngInit& init : host_seeds[AllItems<RngInit>{}])
    {
        init.seed = sample_uniform_int(host_rng);
    }

    // Resize device data and assign
    make_builder(&state->rng).resize(size);
    detail::RngInitData<Ownership::value, MemSpace::device> inits_device;
    inits_device.seeds = host_seeds;
    detail::rng_state_init(make_ref(*state), make_const_ref(inits_device));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
