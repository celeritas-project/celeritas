//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateStore.cu
//---------------------------------------------------------------------------//
#include "RngStateStore.hh"

#include <random>
#include <vector>
#include "base/Assert.hh"
#include "RngStateInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// METHODS
//---------------------------------------------------------------------------//
/*!
 * Construct with the number of RNG states.
 */
RngStateStore::RngStateStore(size_type size, unsigned long host_seed)
    : data_(size)
{
    REQUIRE(size > 0);

    // Host-side RNG for seeding device RNG
    using seed_type = RngSeed::value_type;
    std::mt19937                             host_rng(host_seed);
    std::uniform_int_distribution<seed_type> sample_uniform_int;

    // Create seeds on host
    std::vector<seed_type> host_seeds(size);
    for (auto& seed : host_seeds)
        seed = sample_uniform_int(host_rng);

    DeviceVector<seed_type> device_seeds(size);
    device_seeds.copy_to_device(make_span(host_seeds));
    rng_state_init_device(this->device_pointers(),
                          device_seeds.device_pointers());
}

//---------------------------------------------------------------------------//
/*!
 * Return a view to on-device memory
 */
RngStatePointers RngStateStore::device_pointers()
{
    REQUIRE(!data_.empty());

    RngStatePointers result;
    result.rng = data_.device_pointers();

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
