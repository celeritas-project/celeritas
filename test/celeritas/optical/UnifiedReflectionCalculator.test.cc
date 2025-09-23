//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/UnifiedReflectionCalculator.test.cc
//---------------------------------------------------------------------------//
// #include "celeritas/optical/UnifiedReflectionCalculator.hh"

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
}  // namespace test
}  // namespace optical
}  // namespace celeritas
