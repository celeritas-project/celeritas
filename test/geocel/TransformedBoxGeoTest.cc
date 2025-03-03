//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/TransformedBoxGeoTest.cc
//---------------------------------------------------------------------------//
#include "TransformedBoxGeoTest.hh"

#include "corecel/math/ArrayOperators.hh"

#include "GenericGeoTestInterface.hh"
#include "TestMacros.hh"
#include "UnitUtils.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct with tracking test interface.
 */
TransformedBoxGeoTest::TransformedBoxGeoTest(GenericGeoTestInterface* geo_test)
    : test_{geo_test}
{
    CELER_EXPECT(test_);
}

//---------------------------------------------------------------------------//
//! Test geometry accessors
void TransformedBoxGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();
    if (test_->geometry_type() != "ORANGE")
    {
        EXPECT_EQ(3, geo.max_depth());
    }

    Real3 expected_lo{-50., -50., -50.};
    Real3 expected_hi{50., 50., 50.};
    if (test_->geometry_type() == "VecGeom")
    {
        // VecGeom expands its bboxes
        expected_lo -= .001;
        expected_hi += .001;
    }
    auto const& bbox = geo.bbox();
    EXPECT_VEC_SOFT_EQ(expected_lo, to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ(expected_hi, to_cm(bbox.upper()));
}

//---------------------------------------------------------------------------//
void TransformedBoxGeoTest::test_trace() const
{
    // Surface VecGeom needs lower safety tolerance, and this test needs even
    // lower
    real_type const safety_tol = real_type{10} * test_->safety_tol();

    {
        auto result = test_->track({0, 0, -25}, {0, 0, 1});
        static char const* const expected_volumes[] = {
            "world",
            "simple",
            "world",
            "enclosing",
            "tiny",
            "enclosing",
            "world",
            "simple",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            13,
            4,
            6,
            1.75,
            0.5,
            1.75,
            6,
            4,
            38,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        if (test_->geometry_type() != "ORANGE")
        {
            static real_type const expected_hw_safety[] = {
                5.3612159321677,
                1,
                2.3301270189222,
                0.875,
                0.25,
                0.875,
                3,
                1,
                19,
            };
            EXPECT_VEC_NEAR(
                expected_hw_safety, result.halfway_safeties, safety_tol);
        }
    }
    {
        auto result = test_->track({0.25, 0, -25}, {0., 0, 1});
        static char const* const expected_volumes[] = {
            "world",
            "simple",
            "world",
            "enclosing",
            "tiny",
            "enclosing",
            "world",
            "simple",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            12.834936490539,
            3.7320508075689,
            6.4330127018922,
            1.75,
            0.5,
            1.75,
            6,
            4,
            38,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        if (test_->geometry_type() != "ORANGE")
        {
            static real_type const expected_hw_safety[] = {
                5.5576905283833,
                0.93301270189222,
                2.0176270189222,
                0.75,
                0.25,
                0.75,
                3,
                0.75,
                19,
            };
            EXPECT_VEC_NEAR(
                expected_hw_safety, result.halfway_safeties, safety_tol);
        }
    }
    {
        auto result = test_->track({0, 0.25, -25}, {0, 0., 1});
        static char const* const expected_volumes[] = {
            "world",
            "simple",
            "world",
            "enclosing",
            "tiny",
            "enclosing",
            "world",
            "simple",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            13,
            4,
            6,
            1.75,
            0.5,
            1.75,
            6,
            4,
            38,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        if (test_->geometry_type() != "ORANGE")
        {
            static real_type const expected_hw_safety[] = {
                5.3612159321677,
                1,
                2.3301270189222,
                0.875,
                0.12530113594871,
                0.875,
                3,
                1,
                19,
            };
            EXPECT_VEC_NEAR(
                expected_hw_safety, result.halfway_safeties, safety_tol);
        }
    }
    {
        auto result = test_->track({0.01, -20, 0.20}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"world", "enclosing", "tiny", "enclosing", "world"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {18.5, 1.1250390198213, 0.75090449735279, 1.1240564828259, 48.5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        if (test_->geometry_type() != "ORANGE")
        {
            static real_type const expected_hw_safety[]
                = {9.25, 0.56184193052552, 0.05, 0.56135125378224, 24.25};
            EXPECT_VEC_NEAR(
                expected_hw_safety, result.halfway_safeties, safety_tol);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
