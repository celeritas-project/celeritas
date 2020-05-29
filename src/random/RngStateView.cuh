//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngStateView.cuh
//---------------------------------------------------------------------------//
#ifndef random_RngStateView_cuh
#define random_RngStateView_cuh

#include "random/RngEngine.cuh"

#include "base/Assert.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device view to a vector of CUDA random number generator states.
 *
 * This "view" is expected to be an argument to a kernel launch.
 */
class RngStateView
{
  public:
    //@{
    //! Type aliases
    using size_type = celeritas::ssize_type;
    using RngState  = RngEngine::RngState;
    //@}

    //! Construction parameters
    struct Params
    {
        size_type size = 0;
        RngState* rng  = nullptr;
    };

  public:
    //! Construct on host with invariant parameters
    explicit RngStateView(const Params& params) : data_(params)
    {
        REQUIRE(data_.size > 0);
        assert(data_.rng);
    }

    //! Number of states
    __device__ size_type size() const { return data_.size; }

    //! Get a reference to the local state for a thread
    __device__ RngEngine operator[](size_type idx) const
    {
        REQUIRE(idx < this->size());
        return RngEngine(data_.rng + idx);
    }

  private:
    Params data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // random_RngStateView_cuh
