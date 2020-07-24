//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file IsotropicDistribution.test.cu
//---------------------------------------------------------------------------//
#include "random/distributions/IsotropicDistribution.hh"
#include "random/cuda/RngStateStore.hh"
#include "random/cuda/RngEngine.cuh"
#include <random>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "base/Range.hh"
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "base/KernelParamCalculator.cuda.hh"

using celeritas::IsotropicDistribution;
using celeritas::RngEngine;
using celeritas::RngStatePointers;
using celeritas::RngStateStore;

//---------------------------------------------------------------------------//
// CUDA KERNELS
//---------------------------------------------------------------------------//

__global__ void sample(RngStatePointers                      view,
                       IsotropicDistribution<>::result_type* samples,
                       IsotropicDistribution<>               sample_isotropic)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < view.size)
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = sample_isotropic(rng);
    }
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class IsotropicDistributionTestCu : public celeritas::Test
{
  protected:
    void SetUp() override {}

    thrust::device_vector<IsotropicDistribution<>::result_type> samples;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(IsotropicDistributionTestCu, bin)
{
    int num_samples = 10000;

    IsotropicDistribution<> sample_isotropic;

    // Allocate device memory for results
    samples.resize(num_samples);

    // Initialize the RNG states on device
    RngStateStore container(num_samples);

    celeritas::KernelParamCalculator calc_launch_params;
    auto                             params = calc_launch_params(num_samples);
    sample<<<params.grid_size, params.block_size>>>(
        container.device_pointers(),
        thrust::raw_pointer_cast(samples.data()),
        sample_isotropic);

    cudaDeviceSynchronize();

    // Copy data back to host
    thrust::host_vector<IsotropicDistribution<>::result_type> host_samples
        = this->samples;

    std::vector<int> octant_tally(8, 0);
    for (auto u : host_samples)
    {
        // Make sure sampled point is on the surface of the unit sphere
        double r = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        ASSERT_DOUBLE_EQ(r, 1.0);

        // Tally octant
        int tally_bin = 1 * (u[0] >= 0) + 2 * (u[1] >= 0) + 4 * (u[2] >= 0);
        ASSERT_GE(tally_bin, 0);
        ASSERT_LE(tally_bin, octant_tally.size() - 1);
        ++octant_tally[tally_bin];
    }

    for (int count : octant_tally)
    {
        double octant = static_cast<double>(count) / num_samples;
        EXPECT_NEAR(octant, 1. / 8, 0.01);
        cout << octant << ' ';
    }
    cout << endl;
}
