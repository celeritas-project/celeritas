//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/UnifiedReflectionSampler.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/model/UnifiedReflectionSampler.hh"

#include "corecel/random/HistogramSampler.hh"
#include "celeritas/optical/surface/model/LambertianDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

class UnifiedReflectionSamplerTest : public ::celeritas::test::Test
{
};

//---------------------------------------------------------------------------//
// Test Lambertian distribution
TEST_F(UnifiedReflectionSamplerTest, lambertian)
{
    constexpr size_type num_samples = 10000;
    HistogramSampler calc_histogram(10, {0, 1}, num_samples);

    Real3 normal = make_unit_vector(Real3{2, -1, 3});
    auto to_cos_normal
        = [&normal](Real3 const& refl) { return dot_product(normal, refl); };

    auto actual = calc_histogram(to_cos_normal, LambertianDistribution{normal});

    SampledHistogram expected;
    expected.distribution = {
        0.095,
        0.299,
        0.487,
        0.72,
        0.926,
        1.066,
        1.321,
        1.587,
        1.643,
        1.856,
    };
    expected.rng_count = 4;

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
// Test specular spike, specular lobe, and back-scattering modes
TEST_F(UnifiedReflectionSamplerTest, modes)
{
    auto global_normal = make_unit_vector(Real3{-1, 3, 2});
    auto facet_normal = make_unit_vector(Real3{-1, 4, 2});

    PhotonPhasor photon;
    photon.direction = make_unit_vector(Real3{1, -1, -2});
    photon.polarization = make_unit_vector(Real3{2, 0, 1});

    UnifiedReflectionSampler calc_reflection{
        {0.3, 0.3, 0.4, 0}, photon, global_normal, facet_normal};

    // Specular spike
    {
        auto result = calc_reflection.calc_specular_spike();

        PhotonPhasor expected{
            {-0.0583211843519805, 0.991460133983668, 0.116642368703961},
            {-0.894427190999916, 0, -0.447213595499958}};

        EXPECT_VEC_SOFT_EQ(expected.direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected.polarization, result.polarization);
    }
    // Specular lobe
    {
        auto result = calc_reflection.calc_specular_lobe();

        PhotonPhasor expected{
            {0.0583211843519804, 0.991460133983668, -0.116642368703961},
            {-0.894427190999916, 0, -0.447213595499958}};

        EXPECT_VEC_SOFT_EQ(expected.direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected.polarization, result.polarization);
    }
    // Back scattering
    {
        auto result = calc_reflection.calc_back_scattering();

        PhotonPhasor expected{-photon.direction, -photon.polarization};

        EXPECT_VEC_SOFT_EQ(expected.direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected.polarization, result.polarization);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
