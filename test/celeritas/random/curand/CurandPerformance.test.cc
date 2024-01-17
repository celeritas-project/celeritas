//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/curand/CurandPerformance.test.cc
//---------------------------------------------------------------------------//
#include "CurandPerformance.test.hh"

#if CELERITAS_USE_CUDA
#    include <curand_kernel.h>
#endif

#include "corecel/cont/Range.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CurandTest : public Test
{
  protected:
    void SetUp() override
    {
        // Test parameters on the host
        test_params.nsamples = 1.e+7;
        test_params.nblocks = 1;
        test_params.nthreads = 1;
        test_params.seed = 12345u;
        test_params.tolerance = 1.0e-3;
    }

    template<typename T>
    void check_mean_host()
    {
        T devStates;
        curand_init(test_params.seed, 0, 0, &devStates);

        double sum = 0;
        double sum2 = 0;
        for ([[maybe_unused]] auto i : range(test_params.nsamples))
        {
            double u01 = curand_uniform(&devStates);
            sum += u01;
            sum2 += u01 * u01;
        }

        double mean = sum / test_params.nsamples;
        double variance = sum2 / test_params.nsamples - mean * mean;
        EXPECT_SOFT_NEAR(mean, 0.5, test_params.tolerance);
        EXPECT_SOFT_NEAR(variance, 1 / 12., test_params.tolerance);
    }

  protected:
    // Test parameters
    TestParams test_params;
};

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//
#if CELERITAS_USE_CUDA
TEST_F(CurandTest, curand_xorwow_host)
{
    // XORWOW (default) generator
    this->check_mean_host<curandState>();
}

TEST_F(CurandTest, curand_mrg32k3a_host)
{
    // MRG32k3a generator
    this->check_mean_host<curandStateMRG32k3a>();
}

TEST_F(CurandTest, curand_philox4_32_10_host)
{
    // Philox4_32_10 generator
    this->check_mean_host<curandStatePhilox4_32_10_t>();
}
#endif

TEST_F(CurandTest, std_mt19937_host)
{
    // Mersenne Twister generator
    auto rng = DiagnosticRngEngine<std::mt19937>();

    double sum = 0;
    double sum2 = 0;
    for ([[maybe_unused]] auto i : range(test_params.nsamples))
    {
        double u01 = generate_canonical<double>(rng);
        sum += u01;
        sum2 += u01 * u01;
    }

    double mean = sum / test_params.nsamples;
    double variance = sum2 / test_params.nsamples - mean * mean;
    EXPECT_SOFT_NEAR(mean, 0.5, test_params.tolerance);
    EXPECT_SOFT_NEAR(variance, 1 / 12., test_params.tolerance);
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

#if CELERITAS_USE_CUDA
class CurandDeviceTest : public CurandTest
{
    void SetUp() override
    {
        // Test parameters on the device (100 * host nsamples)
        test_params.nsamples = 1.e+9;
        test_params.nblocks = 64;
        test_params.nthreads = 256;
        test_params.seed = 12345u;
        test_params.tolerance = 1.0e-3;
    }

  public:
    void check_mean_device(TestOutput result)
    {
        double sum_total = 0;
        double sum2_total = 0;
        for (auto i : range(test_params.nblocks * test_params.nthreads))
        {
            sum_total += result.sum[i];
            sum2_total += result.sum2[i];
        }
        double mean = sum_total / test_params.nsamples;
        double variance = sum2_total / test_params.nsamples - mean * mean;
        EXPECT_SOFT_NEAR(mean, 0.5, test_params.tolerance);
        EXPECT_SOFT_NEAR(variance, 1 / 12., test_params.tolerance);
    }
};

TEST_F(CurandDeviceTest, curand_xorwow_device)
{
    // XORWOW (default) generator
    auto output = curand_test<curandState>(test_params);
    this->check_mean_device(output);
}

TEST_F(CurandDeviceTest, curand_mrg32k3a_device)
{
    // MRG32k3a generator
    auto output = curand_test<curandStateMRG32k3a>(test_params);
    this->check_mean_device(output);
}

TEST_F(CurandDeviceTest, curand_philox4_32_10_t_device)
{
    // Philox4_32_10 generator
    auto output = curand_test<curandStatePhilox4_32_10_t>(test_params);
    this->check_mean_device(output);
}

TEST_F(CurandDeviceTest, curand_mtgp32_device)
{
    // MTGP32-11213 (Mersenne Twister RNG for the GPU)
    auto output = curand_test<curandStateMtgp32>(test_params);
    this->check_mean_device(output);
}
#endif
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
