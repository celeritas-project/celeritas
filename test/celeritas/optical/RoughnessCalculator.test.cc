//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/RoughnessSampler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/random/HistogramSampler.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "celeritas/optical/surface/GaussianRoughnessSampler.hh"
#include "celeritas/optical/surface/SmearRoughnessSampler.hh"
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

//! Mock sampler that chooses a random
class IsotropicSampler
{
  public:
    // Take but ignore a normal
    explicit IsotropicSampler(Real3 const&) {}

    template<class Engine>
    Real3 operator()(Engine& rng)
    {
        return sample_iso_(rng);
    }

  private:
    IsotropicDistribution<> sample_iso_;
};

//---------------------------------------------------------------------------//
// Test harness
class RoughnessSamplerTest : public ::celeritas::test::Test
{
  public:
};

//---------------------------------------------------------------------------//
// Test the surface vs normal rejection sampler
TEST_F(RoughnessSamplerTest, entering_surface)
{
    constexpr size_type num_samples = 4000;
    HistogramSampler calc_histogram(4, {-1, 1}, num_samples);

    std::vector<SampledHistogram> actual;

    // Test over range of incident directions
    static Real3 incident_directions[]
        = {{0, 0, -1}, {1, 0, -1}, {0, 1, -1}, {-1, 0, -1}};

    constexpr Real3 global_normal = {0, 0, 1};
    for (Real3 incident_dir : incident_directions)
    {
        incident_dir = make_unit_vector(incident_dir);
        EnteringSurfaceNormalSampler<IsotropicSampler> sample_normal{
            incident_dir, global_normal};

        auto to_cos_normal = [&incident_dir](Real3 const& sampled_normal) {
            return dot_product(incident_dir, sampled_normal);
        };

        actual.push_back(calc_histogram(to_cos_normal, sample_normal));
    }

    // All sampled normals should satisfy entering surface condition
    // which means dot_product(sampled_normal, incident_dir) < 0
    static SampledHistogram const expected[] = {
        {{0.9595, 1.0405, 0, 0}, 7.987},
        {{0.998, 1.002, 0, 0}, 8.09},
        {{0.982, 1.018, 0, 0}, 8.026},
        {{1.019, 0.981, 0, 0}, 8.041},
    };
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
// Test smear roughness model distribution
TEST_F(RoughnessSamplerTest, smear)
{
    constexpr size_type num_samples = 10000;
    HistogramSampler calc_histogram(5, {0, 1}, num_samples);

    Real3 normal = make_unit_vector(Real3{1, 0, -1});
    std::vector<SampledHistogram> actual;

    // Test over range of roughness values
    for (real_type roughness : {0.0, 0.1, 0.5, 0.7, 0.9, 1.0})
    {
        SmearRoughnessSampler sample_normal{normal, roughness};

        auto to_cos_normal = [&normal](Real3 const& sampled_normal) {
            return dot_product(normal, sampled_normal);
        };

        actual.push_back(calc_histogram(to_cos_normal, sample_normal));
    }

    static SampledHistogram const expected[] = {
        {{0, 0, 0, 0, 5}, 6},
        {{0, 0, 0, 0, 5}, 6},
        {{0, 0, 0, 0, 5}, 6},
        {{0, 0, 0, 0.552, 4.448}, 6},
        {{0, 0, 0.289, 1.385, 3.326}, 6},
        {{0.0065, 0.131, 0.498, 1.411, 2.9535}, 6},
    };

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
// Test Gaussian roughness model distribution
TEST_F(RoughnessSamplerTest, gaussian)
{
    constexpr size_type num_samples = 10000;
    HistogramSampler calc_histogram(5, {0.0, 1.0}, num_samples);

    Real3 normal = make_unit_vector(Real3{1, 0, -1});
    std::vector<SampledHistogram> actual;

    // Test over range of sigma_alpha (stdev in radians) values
    // A "very rough" crystal in the UNIFIED paper has sigma_alpha of 0.2053
    // (note that the paper gives the value in degrees), having at most a
    // deflection angle cosine of ~0.76 (40 degrees)
    for (real_type sigma_alpha : {0.01, 0.05, 0.1, 0.2053, 0.4, 0.6})
    {
        GaussianRoughnessSampler sample_normal{normal, sigma_alpha};

        auto to_cos_normal = [&normal](Real3 const& sampled_normal) {
            return dot_product(normal, sampled_normal);
        };

        actual.push_back(calc_histogram(to_cos_normal, sample_normal));
    }

    static SampledHistogram const expected[] = {
        {{0, 0, 0, 0, 5}, 21.8074},
        {{0, 0, 0, 0, 5}, 22.0256},
        {{0, 0, 0, 0, 5}, 22.4858},
        {{0, 0, 0.0005, 0.037, 4.9625}, 22.3334},
        {{0.011, 0.051, 0.2305, 0.967, 3.7405}, 15.1088},
        {{0.174, 0.366, 0.7145, 1.298, 2.4475}, 11.5844},
    };
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
