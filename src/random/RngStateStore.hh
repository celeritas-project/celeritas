//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>
#include "base/Types.hh"
#include "Types.hh"

namespace celeritas
{
struct RngStateContainerPimpl;
struct RngStatePointers;
//---------------------------------------------------------------------------//
/*!
 * Manage ownership of on-device random number generator.
 */
class RngStateStore
{
  public:
    // Construct with the number of RNG states
    RngStateStore(ssize_type size, seed_type host_seed = 12345);

    //@{
    //! Defaults that cause thrust to launch kernels
    RngStateStore();
    ~RngStateStore();
    RngStateStore(RngStateStore&&);
    RngStateStore& operator=(RngStateStore&&);
    //@}

    //! Number of states
    ssize_type size() const { return size_; }

    //! Resize the RNG state vector, initializing new states if necessary
    void resize(ssize_type size);

    // Emit a view to on-device memory
    RngStatePointers device_pointers() const;

  private:
    // Host-side RNG for seeding device RNG
    std::mt19937                             host_rng_;
    std::uniform_int_distribution<seed_type> sample_uniform_int_;

    // Number of states
    ssize_type size_ = 0;

    // Stored RNG states on device
    std::unique_ptr<RngStateContainerPimpl> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
