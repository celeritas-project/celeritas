//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/FourLevelsGeoTest.cc
//---------------------------------------------------------------------------//
#include "FourLevelsGeoTest.hh"

#include "corecel/math/ArrayOperators.hh"

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
FourLevelsGeoTest::FourLevelsGeoTest(GenericGeoTestInterface* geo_test)
    : test_{geo_test}
{
    CELER_EXPECT(test_);
}

//---------------------------------------------------------------------------//
//! Test geometry accessors
void FourLevelsGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();
    EXPECT_EQ(4, geo.max_depth());

    auto const& bbox = geo.bbox();
    Real3 expected_lo{-24., -24., -24.};
    Real3 expected_hi{24., 24., 24.};
    if (test_->geometry_type() == "VecGeom")
    {
        // VecGeom expands its bboxes
        expected_lo -= .001;
        expected_hi += .001;
    }

    EXPECT_VEC_SOFT_EQ(expected_lo, to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ(expected_hi, to_cm(bbox.upper()));

    static char const* const expected_vol_labels[] = {
        "Shape2",
        "Shape1",
        "Envelope",
        "World",
    };
    EXPECT_VEC_EQ(expected_vol_labels, test_->get_volume_labels());

    static char const* const expected_vol_inst_labels[] = {
        "Shape2",
        "Shape1",
        "env1",
        "env2",
        "env3",
        "env4",
        "env5",
        "env6",
        "env7",
        "env8",
        "World_PV",
    };
    EXPECT_VEC_EQ(expected_vol_inst_labels,
                  test_->get_volume_instance_labels());

    if (test_->g4world())
    {
        EXPECT_VEC_EQ(expected_vol_inst_labels, test_->get_g4pv_labels());
    }
}

//---------------------------------------------------------------------------//
void FourLevelsGeoTest::test_trace() const
{
    // Surface VecGeom needs lower safety tolerance
    real_type const safety_tol = test_->safety_tol();

    {
        SCOPED_TRACE("Rightward");
        auto result = test_->track({-10, -10, -10}, {1, 0, 0});

        static char const* const expected_volumes[] = {
            "Shape2",
            "Shape1",
            "Envelope",
            "World",
            "Envelope",
            "Shape1",
            "Shape2",
            "Shape1",
            "Envelope",
            "World",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {5, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {2.5, 0.5, 0.5, 3, 0.5, 0.5, 5, 0.5, 0.5, 3.5};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("From just inside outside edge");
        auto result = test_->track({-24 + 0.001, 10., 10.}, {1, 0, 0});

        static char const* const expected_volumes[] = {
            "World",
            "Envelope",
            "Shape1",
            "Shape2",
            "Shape1",
            "Envelope",
            "World",
            "Envelope",
            "Shape1",
            "Shape2",
            "Shape1",
            "Envelope",
            "World",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {7 - 0.001, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {3.4995, 0.5, 0.5, 5, 0.5, 0.5, 3, 0.5, 0.5, 5, 0.5, 0.5, 3.5};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("Leaving world");
        auto result = test_->track({-10, 10, 10}, {0, 1, 0});

        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 2, 6};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {2.5, 0.5, 1, 3};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("Upward");
        auto result = test_->track({-10, 10, 10}, {0, 0, 1});

        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 3, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {2.5, 0.5, 1.5, 2.5};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
