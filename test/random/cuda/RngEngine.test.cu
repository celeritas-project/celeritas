//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.test.cu
//---------------------------------------------------------------------------//
#include "random/cuda/RngStateStore.hh"
#include "random/cuda/RngEngine.cuh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "base/Range.hh"
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "base/KernelParamCalculator.cuda.hh"

using celeritas::generate_canonical;
using celeritas::RngEngine;
using celeritas::RngState;
using celeritas::RngStatePointers;
using celeritas::RngStateStore;

//---------------------------------------------------------------------------//
// CUDA KERNELS
//---------------------------------------------------------------------------//

__global__ void sample_native(int                     num_samples,
                              RngStatePointers        view,
                              RngEngine::result_type* samples)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < num_samples)
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = rng();
    }
}

template<class RealType>
__global__ void
sample_real(int num_samples, RngStatePointers view, RealType* samples)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() < num_samples)
    {
        RngEngine rng(view, tid);
        samples[tid.get()] = generate_canonical<RealType>(rng);
    }
}

//---------------------------------------------------------------------------//
// INT TEST
//---------------------------------------------------------------------------//

TEST(RngEngineIntTest, regression)
{
    using value_type = RngEngine::result_type;

    int num_samples = 1024;

    // Allocate device memory for results
    thrust::device_vector<value_type> samples(num_samples);

    // Initialize the RNG states on device
    RngStateStore container(num_samples);
    EXPECT_EQ(container.size(), num_samples);

    celeritas::KernelParamCalculator calc_launch_params;
    auto                             params = calc_launch_params(num_samples);
    sample_native<<<params.grid_size, params.block_size>>>(
        num_samples,
        container.device_pointers(),
        thrust::raw_pointer_cast(samples.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy data back to host
    std::vector<value_type> host_samples(num_samples);
    thrust::copy(samples.begin(), samples.end(), host_samples.begin());

    // Print a subset of the values
    std::vector<value_type> test_values;
    for (int i = 0; i < num_samples; i += 127)
    {
        test_values.push_back(host_samples[i]);
    }

    // PRINT_EXPECTED(test_values);
    static const unsigned int expected_test_values[] = {165860337u,
                                                        3006138920u,
                                                        2161337536u,
                                                        390101068u,
                                                        2347834113u,
                                                        100129048u,
                                                        4122784086u,
                                                        473544901u,
                                                        2822849608u};
    EXPECT_VEC_EQ(test_values, expected_test_values);
}

//---------------------------------------------------------------------------//
// FLOAT TEST
//---------------------------------------------------------------------------//

template<typename T>
class RngEngineFloatTest : public celeritas::Test
{
};

void check_expected_float_samples(const thrust::host_vector<float>& v)
{
    EXPECT_FLOAT_EQ(0.038617369, v[0]);
    EXPECT_FLOAT_EQ(0.411269426, v[1]);
}
void check_expected_float_samples(const thrust::host_vector<double>& v)
{
    EXPECT_DOUBLE_EQ(0.283318433931184, v[0]);
    EXPECT_DOUBLE_EQ(0.653335242131673, v[1]);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RngEngineFloatTest, FloatTypes, );

TYPED_TEST(RngEngineFloatTest, generate_canonical)
{
    using real_type = TypeParam;
    int num_samples = 100;

    // Allocate device memory for results
    thrust::device_vector<real_type> samples(num_samples);

    // Initialize the RNG states on device
    RngStateStore container(num_samples);
    EXPECT_EQ(container.size(), num_samples);

    celeritas::KernelParamCalculator calc_launch_params;
    auto                             params = calc_launch_params(num_samples);
    sample_real<<<params.grid_size, params.block_size>>>(
        num_samples,
        container.device_pointers(),
        thrust::raw_pointer_cast(samples.data()));
    CELER_CUDA_CALL(cudaDeviceSynchronize());

    // Copy data back to host
    thrust::host_vector<real_type> host_samples = samples;
    EXPECT_EQ(host_samples.size(), num_samples);
    for (real_type sample : host_samples)
    {
        EXPECT_GE(sample, real_type(0));
        EXPECT_LE(sample, real_type(1));
    }

    check_expected_float_samples(host_samples);
}

//---------------------------------------------------------------------------//
// TEST on CPU
//---------------------------------------------------------------------------//
TEST(RngEngineCPUTest, generate_on_cpu)
{
    int           num_samples = 1024 * 1000;
    unsigned long seed        = 12345u;

    RngState         host_state[1];
    RngStatePointers host_pointers{celeritas::make_span(host_state)};
    RngEngine        rng(host_pointers, celeritas::ThreadId{0});
    rng = RngEngine::Initializer_t{seed};

    double mean = 0;
    for (int i = 0; i < num_samples; ++i)
    {
        mean += generate_canonical<double>(rng);
    }
    mean /= num_samples;
    EXPECT_NEAR(0.5, mean, 0.0001);
}
