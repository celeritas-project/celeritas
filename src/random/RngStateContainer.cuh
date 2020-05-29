//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateContainer.cuh
//---------------------------------------------------------------------------//
#ifndef random_RngStateContainer_cuh
#define random_RngStateContainer_cuh

#include "random/RngEngine.cuh"
#include "random/RngStateView.cuh"

#include <random>

#include <thrust/device_vector.h>

#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage ownership of on-device random number generator.
 */
class RngStateContainer
{
  public:
    //@{
    //! Type aliases
    using size_type = celeritas::ssize_type;
    using seed_type = RngEngine::seed_type;
    using RngState  = RngStateView::RngState;
    //@}

  public:
    // Default constructor
    RngStateContainer();

    // Construct with the number of RNG states
    RngStateContainer(size_type count, seed_type host_seed = 12345);

    // Emit a view to on-device memory
    inline RngStateView device_view();

    //! Number of states
    size_type size() const { return rng_.size(); }

    //! Resize the RNG state vector, initializing new states if necessary
    void resize(size_type count);

  private:
    thrust::device_vector<RngState> rng_;

    // Host-side RNG for seeding device RNG
    std::mt19937                             host_rng_;
    std::uniform_int_distribution<seed_type> sample_uniform_int_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "RngStateContainer.i.cuh"

#endif // random_RngStateContainer_cuh
