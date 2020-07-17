//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformRealDistribution.test.cu
//---------------------------------------------------------------------------//
#include "random/UniformRealDistribution.hh"
#include "random/RngStateStore.hh"
#include "random/RngEngine.cuh"
#include <random>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "base/Range.hh"
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "base/KernelParamCalculator.cuda.hh"

using celeritas::RngEngine;
using celeritas::RngStatePointers;
using celeritas::RngStateStore;
using celeritas::UniformRealDistribution;

//---------------------------------------------------------------------------//
// CUDA KERNELS
//---------------------------------------------------------------------------//

__global__ void sample(RngStatePointers          view,
                       double*                   samples,
                       UniformRealDistribution<> sample_uniform)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < view.size)
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = sample_uniform(rng);
    }
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UniformRealDistributionTestCu : public celeritas::Test
{
  protected:
    void SetUp() override {}

    thrust::device_vector<double> samples;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UniformRealDistributionTestCu, bin)
{
    int num_samples = 1000;

    double                    min = 0.0;
    double                    max = 5.0;
    UniformRealDistribution<> sample_uniform{min, max};

    // Allocate device memory for results
    samples.resize(num_samples);

    // Initialize the RNG states on device
    RngStateStore container(num_samples);

    celeritas::KernelParamCalculator calc_launch_params;
    auto                             params = calc_launch_params(num_samples);
    sample<<<params.grid_size, params.block_size>>>(
        container.device_pointers(),
        thrust::raw_pointer_cast(samples.data()),
        sample_uniform);

    cudaDeviceSynchronize();

    // Copy data back to host
    thrust::host_vector<double> host_samples = this->samples;

    // Bin the data
    std::vector<int> counters(5);
    for (double sample : host_samples)
    {
        EXPECT_GE(sample, min);
        EXPECT_LE(sample, max);
        counters[int(sample)] += 1;
    }

    for (int count : counters)
    {
        cout << count << ' ';
    }
    cout << endl;
}
