//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateContainer.cu
//---------------------------------------------------------------------------//
#include "RngStateContainer.hh"

#include <vector>
#include <thrust/device_vector.h>
#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "RngEngine.cuh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RngStateContainerPimpl
{
    thrust::device_vector<RngState> rng;
};

//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
__global__ void initialize_states(RngStateView view, seed_type* seeds)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < view.size)
    {
        RngEngine rng(view, tid);
        rng.initialize_state(seeds[tid.get()]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct with the number of RNG states.
 */
RngStateContainer::RngStateContainer(ssize_type size, seed_type host_seed)
    : host_rng_(host_seed)
{
    this->resize(size);
}

//---------------------------------------------------------------------------//
// Default constructor/destructor/move
RngStateContainer::RngStateContainer()                    = default;
RngStateContainer::~RngStateContainer()                   = default;
RngStateContainer::RngStateContainer(RngStateContainer&&) = default;
RngStateContainer& RngStateContainer::operator=(RngStateContainer&&) = default;

//---------------------------------------------------------------------------//
/*!
 * Resize the RNG state vector, initializing new states if the number requested
 * is larger than the current size.
 */
void RngStateContainer::resize(ssize_type size)
{
    int num_states     = this->size();
    int num_new_states = size - num_states;

    // Allocate and copy data to device
    if (this->size() == 0)
    {
        data_ = std::make_unique<RngStateContainerPimpl>();
    }
    data_->rng.resize(size);
    size_ = size;

    if (num_new_states > 0)
    {
        // Create seeds on host
        std::vector<seed_type> host_seeds(num_new_states);
        for (auto& seed : host_seeds)
            seed = sample_uniform_int_(host_rng_);

        // Copy seeds to device
        thrust::device_vector<seed_type> seeds = host_seeds;

        // Create a view of new states to initialize
        RngStateView view;
        view.size = num_new_states;
        view.rng  = thrust::raw_pointer_cast(data_->rng.data()) + num_states;

        // Launch kernel to build RNG states on device
        celeritas::KernelParamCalculator calc_launch_params;
        auto params = calc_launch_params(num_new_states);
        initialize_states<<<params.grid_size, params.block_size>>>(
            view, thrust::raw_pointer_cast(seeds.data()));
    }
    ENSURE(data_->rng.size() == size);
}

//---------------------------------------------------------------------------//
/*!
 * Return a view to on-device memory
 */
RngStateView RngStateContainer::device_view() const
{
    REQUIRE(data_);

    RngStateView view;
    view.size = data_->rng.size();
    view.rng  = thrust::raw_pointer_cast(data_->rng.data());

    return view;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
