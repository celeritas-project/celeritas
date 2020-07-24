//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.test.cu
//---------------------------------------------------------------------------//
#include "random/distributions/RadialDistribution.hh"
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

using celeritas::RadialDistribution;
using celeritas::RngEngine;
using celeritas::RngStatePointers;
using celeritas::RngStateStore;

//---------------------------------------------------------------------------//
// CUDA KERNELS
//---------------------------------------------------------------------------//

__global__ void sample(RngStatePointers     view,
                       double*              samples,
                       RadialDistribution<> sample_radial)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < view.size)
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = sample_radial(rng);
    }
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RadialDistributionTestCu : public celeritas::Test
{
  protected:
    void SetUp() override {}

    thrust::device_vector<double> samples;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RadialDistributionTestCu, bin)
{
    int num_samples = 1000;

    double               radius = 5.0;
    RadialDistribution<> sample_radial{radius};

    // Allocate device memory for results
    samples.resize(num_samples);

    // Initialize the RNG states on device
    RngStateStore container(num_samples);

    celeritas::KernelParamCalculator calc_launch_params;
    auto                             params = calc_launch_params(num_samples);
    sample<<<params.grid_size, params.block_size>>>(
        container.device_pointers(),
        thrust::raw_pointer_cast(samples.data()),
        sample_radial);

    cudaDeviceSynchronize();

    // Copy data back to host
    thrust::host_vector<double> host_samples = this->samples;

    // Bin the data
    std::vector<int> counters(5);
    for (double sample : host_samples)
    {
        EXPECT_GE(sample, 0.0);
        EXPECT_LE(sample, radius);
        counters[int(sample)] += 1;
    }

    for (int count : counters)
    {
        cout << count << ' ';
    }
    cout << endl;
}
