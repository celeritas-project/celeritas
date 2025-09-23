//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/UnifiedReflectionCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/model/UnifiedReflectionCalculator.hh"

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

class UnifiedReflectionCalculatorTest : public ::celeritas::test::Test
{
};

//---------------------------------------------------------------------------//
// Test Lambertian distribution
TEST_F(UnifiedReflectionCalculatorTest, lambertian)
{
    constexpr size_type num_samples = 10000;
    HistogramSampler calc_histogram(20, {0.0, 1.0}, num_samples);

    Real3 normal = make_unit_vector(Real3{2, -1, 3});
    auto to_cos_normal
        = [&normal](Real3 const& refl) { return dot_product(normal, refl); };

    auto actual = calc_histogram(to_cos_normal, LambertianDistribution{normal});

    // Approximate values following cos(x) PDF over 20 bins in [0,1]
    // Replace with PRINT_EXPECTED results after checking they're close
    static SampledHistogram const expected{
        {
            500, 498, 493, 486, 475, 461, 445, 426, 404, 380,
            353, 324, 293, 261, 226, 191, 154, 116, 78,  39,
        },
        2};

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_REF_EQ(expected, actual);
    }
}

//---------------------------------------------------------------------------//
// Test specular spike, specular lobe, and back-scattering modes
TEST_F(UnifiedReflectionCalculatorTest, modes)
{
    auto global_normal = make_unit_vector(Real3{-1, 3, 2});
    auto facet_normal = make_unit_vector(Real3{-1, 4, 2});

    PhotonPhasor photon;
    photon.direction = make_unit_vector(Real3{1, -1, -2});
    photon.polarization = make_unit_vector(Real3{2, 0, 1});

    UnifiedReflectionCalculator calc_reflection{
        {0.3, 0.3, 0.4, 0}, photon, global_normal, facet_normal};

    // Specular spike
    {
        auto result = calc_reflection.specular_spike();

        PhotonPhasor expected{{-0.05832118, 0.99146013, 0.11664237},
                              {-0.89442719, 0, -0.4472136}};

        EXPECT_VEC_SOFT_EQ(expected.direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected.polarization, result.polarization);
    }
    // Specular lobe
    {
        auto result = calc_reflection.specular_lobe();

        PhotonPhasor expected{{0.05832118, 0.99146013, -0.11664237},
                              {-0.89442719, -0., -0.4472136}};

        EXPECT_VEC_SOFT_EQ(expected.direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected.polarization, result.polarization);
    }
    // Back scattering
    {
        auto result = calc_reflection.back_scattering();

        PhotonPhasor expected{-photon.direction, -photon.polarization};

        EXPECT_VEC_SOFT_EQ(expected.direction, result.direction);
        EXPECT_VEC_SOFT_EQ(expected.polarization, result.polarization);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
