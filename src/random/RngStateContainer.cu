//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateContainer.cu
//---------------------------------------------------------------------------//
#include "random/RngStateContainer.cuh"

#include <thrust/host_vector.h>

#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
__global__ void
initialize_states(RngStateView view, RngStateContainer::seed_type* seeds)
{
    int local_thread_id = celeritas::KernelParamCalculator::thread_id();
    if (local_thread_id < view.size())
    {
        auto rng = view[local_thread_id];
        rng.initialize_state(seeds[local_thread_id]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct with no states.
 */
RngStateContainer::RngStateContainer() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct with the number of RNG states.
 */
RngStateContainer::RngStateContainer(size_type count, seed_type host_seed)
    : host_rng_(host_seed)
{
    this->resize(count);
}

//---------------------------------------------------------------------------//
/*!
 * Resize the RNG state vector, initializing new states if the number requested
 * is larger than the current size.
 */
void RngStateContainer::resize(size_type count)
{
    using thrust::raw_pointer_cast;

    int current_states = this->size();
    int new_states     = count - current_states;

    rng_.resize(count);

    if (new_states > 0)
    {
        // Create seeds on host
        thrust::host_vector<seed_type> host_seeds(new_states);
        for (auto& seed : host_seeds)
            seed = sample_uniform_int_(host_rng_);
 
        // Copy seeds to device
        thrust::device_vector<seed_type> seeds = host_seeds;
 
        // Create a view of new states to initialize
        RngStateView::Params view_params;
        view_params.size  = new_states;
        view_params.rng   = raw_pointer_cast(rng_.data()) + current_states;
        RngStateView view = RngStateView(view_params);

        // Launch kernel to build RNG states on device
        celeritas::KernelParamCalculator calc_launch_params;
        auto params = calc_launch_params(new_states);
        initialize_states<<<params.grid_size, params.block_size>>>(
            view, raw_pointer_cast(seeds.data()));
    }
    ENSURE(rng_.size() == count);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
