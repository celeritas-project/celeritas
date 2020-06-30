//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.test.cc
//---------------------------------------------------------------------------//
#include "random/RadialDistribution.hh"

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

class StupidRngEngine
{
    using result_type = unsigned int;
    __device__ result_type operator()() { return 1234567; }
};

struct RngStateView
{
    size_t               num_samples;
    StupidRngEngine      engine;
    RadialDistribution<> sample_radial{1.0};
    double*              result;
};

//---------------------------------------------------------------------------//
// CUDA KERNELS
//---------------------------------------------------------------------------//

__global__ void sample(RngStateView view)
{
    int local_thread_id = celeritas::KernelParamCalculator::thread_id();
    if (local_thread_id < view.num_samples)
    {
        view.result[local_thread_id] = view.sample_radial(view.engine);
    }
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RadialDistributionTest : public celeritas::Test
{
  protected:
    void SetUp() override {}

    thrust::device_vector<double> samples;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RadialDistributionTest, bin)
{
    // Allocate device memory
    samples.resize(10);

    // Construct view to device data (or POD that can be copied through kernel
    // launch arguments)
    RngStateView view;
    view.num_samples   = samples.size();
    view.sample_radial = RadialDistribution<>{4.5};
    view.result        = thrust::raw_pointer_cast(samples.data());

    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(samples.size());
    sample<<<params.grid_size, params.block_size>>>(view);

    // TODO: add error checking from CUDA
    // cudaDeviceSynchronize();

    // Copy data back to host
    thrust::host_vector<double> host_samples(this->samples.begin(),
                                             this->samples.end());

    cout << "First sample: " << host_samples[0] << endl;

    // Bin the data
    std::vector<int> counters(5);
    for (double sample : host_samples)
    {
        EXPECT_GE(sample, 0.0);
        EXPECT_LE(sample, 5.0);
        counters[int(sample)] += 1;
    }

    for (int count : counters)
    {
        cout << count << ' ';
    }
    cout << endl;
}
