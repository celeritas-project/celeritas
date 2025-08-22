//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/RoughnessCalculator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/HistogramSampler.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"
#include "celeritas/optical/surface/calc/GaussianRoughnessCalculator.hh"
#include "celeritas/optical/surface/calc/SmearRoughnessCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

class RoughnessCalculatorTest : public ::celeritas::test::Test
{
  public:
};

//---------------------------------------------------------------------------//
// Test whether the surface vs  normal rejection sampler
TEST_F(RoughnessCalculatorTest, entering_surface)
{
    constexpr size_type num_samples = 4000;
    HistogramSampler calc_histogram(4, {-1, 1}, num_samples);

    std::vector<SampledHistogram> actual;

    // Test over range of incident directions
    std::vector<Real3> incident_directions
        = {make_unit_vector(Real3{0, 0, -1}),
           make_unit_vector(Real3{1, 0, -1}),
           make_unit_vector(Real3{0, 1, -1}),
           make_unit_vector(Real3{1, 1, 1}),
           make_unit_vector(Real3{-1, 0, -1})};

    for (Real3 const& incident_dir : incident_directions)
    {
        EnteringSurfaceNormalSampler sample_normal{incident_dir,
                                                   IsotropicDistribution{}};

        auto transform = [&incident_dir](Real3 const& sampled_normal) {
            return dot_product(incident_dir, sampled_normal);
        };

        actual.push_back(calc_histogram(transform, sample_normal));
    }

    // All sampled normals should satisfy entering surface condition
    // which means dot_product(sampled_normal, incident_dir) < 0
    static SampledHistogram const expected[] = {
        {{0.9595, 1.0405, 0, 0}, 7.987},
        {{0.998, 1.002, 0, 0}, 8.09},
        {{0.982, 1.018, 0, 0}, 8.026},
        {{1.0155, 0.9845, 0, 0}, 7.91},
        {{0.9925, 1.0075, 0, 0}, 8.016},
    };
    EXPECT_REF_EQ(expected, actual);
}

//---------------------------------------------------------------------------//
// Test smear roughness model distribution
TEST_F(RoughnessCalculatorTest, smear)
{
    constexpr size_type num_samples = 10000;
    HistogramSampler calc_histogram(5, {0, 1}, num_samples);

    Real3 normal = make_unit_vector(Real3{1, 0, -1});
    std::vector<SampledHistogram> actual;

    // Test over range of roughness values
    for (real_type roughness : {0.0, 0.1, 0.5, 0.7, 0.9, 1.0})
    {
        SmearRoughnessCalculator sample_normal{roughness, normal};

        auto transform = [&normal](Real3 const& sampled_normal) {
            return dot_product(normal, sampled_normal);
        };

        actual.push_back(calc_histogram(transform, sample_normal));
    }

    static SampledHistogram const expected[] = {
        {{0, 0, 0, 0, 5}, 6},
        {{0, 0, 0, 0, 5}, 6},
        {{0, 0, 0, 0, 5}, 6},
        {{0, 0, 0, 0.552, 4.448}, 6},
        {{0, 0, 0.289, 1.385, 3.326}, 6},
        {{0.0065, 0.131, 0.498, 1.411, 2.9535}, 6},
    };

    EXPECT_REF_EQ(expected, actual);
}

//---------------------------------------------------------------------------//
// Test Gaussian roughness model distribution
TEST_F(RoughnessCalculatorTest, gaussian)
{
    constexpr size_type num_samples = 5000;
    HistogramSampler calc_histogram(5, {0, 1}, num_samples);

    Real3 normal = make_unit_vector(Real3{1, 0, -1});
    std::vector<SampledHistogram> actual;

    // Test over range of sigma_alpha values
    for (real_type sigma_alpha : {0.1, 0.7, 1.5, 3.0})
    {
        GaussianRoughnessCalculator sample_normal{sigma_alpha, normal};

        auto transform = [&normal](Real3 const& sampled_normal) {
            return dot_product(normal, sampled_normal);
        };

        actual.push_back(calc_histogram(transform, sample_normal));
    }

    static SampledHistogram const expected[] = {
        {{0, 0, 0, 0, 5}, 22.0336},
        {{0.303, 0.495, 0.792, 1.34, 2.07}, 10.5816},
        {{0.788, 0.9, 1.021, 1.125, 1.166}, 10.1508},
        {{0.993, 1.012, 0.997, 1.013, 0.985}, 11.2816},
    };
    EXPECT_REF_EQ(expected, actual);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
