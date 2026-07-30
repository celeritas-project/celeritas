//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ReflectionFormSampler.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/model/ReflectionFormSampler.hh"

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

class ReflectionFormSamplerTest : public ::celeritas::test::Test
{
};

//---------------------------------------------------------------------------//
// Test Lambertian distribution
TEST_F(ReflectionFormSamplerTest, lambertian)
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
TEST_F(ReflectionFormSamplerTest, modes)
{
    auto global_normal = make_unit_vector(Real3{-1, 3, 2});
    auto facet_normal = make_unit_vector(Real3{-1, 4, 2});

    auto direction = make_unit_vector(Real3{1, -1, -2});
    auto polarization = make_unit_vector(Real3{2, 0, 1});

    ReflectionFormCalculator calc_reflection{
        direction, polarization, global_normal, facet_normal};

    // Specular spike
    {
        auto result = calc_reflection.calc_specular_spike();

        static Real3 const expected_direction{
            -0.0583211843519805, 0.991460133983668, 0.116642368703961};
        static Real3 const expected_polarization{
            -0.894427190999916, 0, -0.447213595499958};

        EXPECT_VEC_SOFT_EQ(expected_direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected_polarization, result.polarization);
    }
    // Specular lobe
    {
        auto result = calc_reflection.calc_specular_lobe();

        static Real3 const expected_direction{
            0.0583211843519804, 0.991460133983668, -0.116642368703961};
        static Real3 const expected_polarization{
            -0.894427190999916, 0, -0.447213595499958};

        EXPECT_VEC_SOFT_EQ(expected_direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected_polarization, result.polarization);
    }
    // Back scattering
    {
        auto result = calc_reflection.calc_backscatter();

        EXPECT_VEC_SOFT_EQ(-direction, result.direction);
        EXPECT_VEC_SOFT_EQ(-polarization, result.polarization);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
