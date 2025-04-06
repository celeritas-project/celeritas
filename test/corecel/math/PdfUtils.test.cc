//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/PdfUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/PdfUtils.hh"

#include <cmath>

#include "corecel/grid/VectorUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class PdfUtilsTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(PdfUtilsTest, segment_integrators)
{
    using Arr2 = Array<double, 2>;
    {
        PostRectangleSegmentIntegrator integrate{};
        EXPECT_SOFT_EQ(3.0, integrate(Arr2{-1, 0.5}, Arr2{5, 12345}));
        EXPECT_SOFT_EQ(-1, integrate.x_eval(Arr2{-1, 0.5}, Arr2{5, 12345}));
    }
    {
        TrapezoidSegmentIntegrator integrate{};
        EXPECT_SOFT_EQ(2.0, integrate(Arr2{1, 0.5}, Arr2{3, 1.5}));
        EXPECT_SOFT_EQ(2.0, integrate.x_eval(Arr2{1, 0.5}, Arr2{3, 1.5}));
    }
}

TEST_F(PdfUtilsTest, integrate_segments)
{
    static double const x[] = {-1, 0, 1, 3, 6};
    static double const f[] = {1, 0, 2, 1, 0};
    std::vector<double> dst(std::size(x));

    {
        SegmentIntegrator integrate_segments{PostRectangleSegmentIntegrator{}};
        integrate_segments(make_span(x), make_span(f), make_span(dst));

        static double const expected_dst[] = {0, 1, 1, 5, 8};
        EXPECT_VEC_SOFT_EQ(expected_dst, dst);

        integrate_segments(make_span(x), make_span(f), make_span(dst), 1.0);
        static double const expected_dst2[] = {1, 2, 2, 6, 9};
        EXPECT_VEC_SOFT_EQ(expected_dst2, dst);
    }

    {
        SegmentIntegrator integrate_segments{TrapezoidSegmentIntegrator{}};
        integrate_segments(make_span(x), make_span(f), make_span(dst));
        static double const expected_dst[] = {0, 0.5, 1.5, 4.5, 6};
        EXPECT_VEC_SOFT_EQ(expected_dst, dst);
    }
}

TEST_F(PdfUtilsTest, calc_moments)
{
    // Uniform distribution with (a, b) = (3, 7): mean = (a + b) / 2 = 5,
    // variance = (b - a)^2 / 12 = 4/3
    std::vector<double> const x_coarse{3, 3.5, 4.25, 5, 6.75, 7};
    std::vector<double> const f_coarse{1, 1, 1, 1, 1, 1};

    std::vector<double> const x_fine = linspace(3, 7, 1000);
    std::vector<double> const f_fine(1000, 1);

    {
        MomentCalculator calc_moments{PostRectangleSegmentIntegrator{}};

        auto result = calc_moments(make_span(x_coarse), make_span(f_coarse));
        EXPECT_SOFT_EQ(4.4375, result.mean);
        EXPECT_SOFT_EQ(0.90234375, result.variance);

        result = calc_moments(make_span(x_fine), make_span(f_fine));
        EXPECT_SOFT_EQ(4.9979979979980023, result.mean);
        EXPECT_SOFT_EQ(1.333331997329303, result.variance);
    }
    {
        MomentCalculator calc_moments{TrapezoidSegmentIntegrator{}};

        auto result = calc_moments(make_span(x_coarse), make_span(f_coarse));
        EXPECT_SOFT_EQ(5, result.mean);
        EXPECT_SOFT_EQ(1.201171875, result.variance);

        result = calc_moments(make_span(x_fine), make_span(f_fine));
        EXPECT_SOFT_EQ(5, result.mean);
        EXPECT_SOFT_EQ(1.3333319973293456, result.variance);
    }
}

TEST_F(PdfUtilsTest, normalize_cdf)
{
    std::vector<double> cdf = {1, 2, 4, 4, 8};

    normalize_cdf(make_span(cdf));
    static double const expected_cdf[] = {0.125, 0.25, 0.5, 0.5, 1};
    EXPECT_VEC_SOFT_EQ(expected_cdf, cdf);

    if (CELERITAS_DEBUG)
    {
        // Empty
        cdf.clear();
        EXPECT_THROW(normalize_cdf(make_span(cdf)), DebugError);

        // One and two zeros
        cdf = {0.0};
        EXPECT_THROW(normalize_cdf(make_span(cdf)), DebugError);
        cdf = {0.0, 0.0};
        EXPECT_THROW(normalize_cdf(make_span(cdf)), DebugError);

        // Nonmonotonic
        cdf = {0, 1, 2, 1.5, 3};
        EXPECT_THROW(normalize_cdf(make_span(cdf)), DebugError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
