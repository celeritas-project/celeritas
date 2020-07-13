//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.test.cu
//---------------------------------------------------------------------------//
#include "random/UniformRealDistribution.hh"
#include "random/RngStateContainer.hh"
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

using celeritas::RngStateContainer;
using celeritas::RngStateView;
using celeritas::RngEngine;
using celeritas::UniformRealDistribution;

//---------------------------------------------------------------------------//
// CUDA KERNELS
//---------------------------------------------------------------------------//

__global__ void
sample(RngStateView view, double* samples, UniformRealDistribution<>
       sample_uniform)
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

class RngEngineTestCu : public celeritas::Test
{
  protected:
    void SetUp() override {}

    thrust::device_vector<double> samples;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RngEngineTestCu, container)
{
    int num_samples = 100;

    UniformRealDistribution<> sample_uniform;

    // Allocate device memory for results
    samples.resize(num_samples);

    // Initialize the RNG states on device
    RngStateContainer container(num_samples);
    EXPECT_EQ(container.size(), num_samples);

    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(num_samples);
    sample<<<params.grid_size, params.block_size>>>(
        container.device_view(),
        thrust::raw_pointer_cast(samples.data()),
        sample_uniform);

    // Copy data back to host
    thrust::host_vector<double> host_samples = this->samples;
    EXPECT_EQ(host_samples.size(), num_samples);
    for (double sample : host_samples)
    {
        EXPECT_GE(sample, 0.0);
        EXPECT_LE(sample, 1.0);
    }

    // Increase the number of threads
    num_samples = 1000;
    samples.resize(num_samples);
    container.resize(num_samples);
    EXPECT_EQ(container.size(), num_samples);
    
    params = calc_launch_params(num_samples);
    sample<<<params.grid_size, params.block_size>>>(
        container.device_view(),
        thrust::raw_pointer_cast(samples.data()),
        sample_uniform);

    // Copy data back to host
    host_samples = this->samples;
    EXPECT_EQ(host_samples.size(), num_samples);
    for (double sample : host_samples)
    {
        EXPECT_GE(sample, 0.0);
        EXPECT_LE(sample, 1.0);
    }

    // Decrease the number of threads
    num_samples = 50;
    samples.resize(num_samples);
    container.resize(num_samples);
    EXPECT_EQ(container.size(), num_samples);
    
    params = calc_launch_params(num_samples);
    sample<<<params.grid_size, params.block_size>>>(
        container.device_view(),
        thrust::raw_pointer_cast(samples.data()),
        sample_uniform);

    // Copy data back to host
    host_samples = this->samples;
    EXPECT_EQ(host_samples.size(), num_samples);
    for (double sample : host_samples)
    {
        EXPECT_GE(sample, 0.0);
        EXPECT_LE(sample, 1.0);
    }
}
