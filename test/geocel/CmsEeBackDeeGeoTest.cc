//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CmsEeBackDeeGeoTest.cc
//---------------------------------------------------------------------------//
#include "CmsEeBackDeeGeoTest.hh"

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
CmsEeBackDeeGeoTest::CmsEeBackDeeGeoTest(GenericGeoTestInterface* geo_test)
    : test_{geo_test}
{
    CELER_EXPECT(test_);
}

//---------------------------------------------------------------------------//
//! Test geometry accessors
void CmsEeBackDeeGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();
    EXPECT_EQ(3, geo.max_depth());

    auto const& bbox = geo.bbox();
    Real3 expected_lo{0., -177.5, 359.5};
    Real3 expected_hi{177.5, 177.5, 399.6};
    if (test_->geometry_type() == "VecGeom")
    {
        // VecGeom expands its bboxes
        expected_lo -= .001;
        expected_hi += .001;
    }

    EXPECT_VEC_NEAR(expected_lo, to_cm(bbox.lower()), real_type{1e-10});
    EXPECT_VEC_NEAR(expected_hi, to_cm(bbox.upper()), real_type{1e-10});

    static char const* const expected_vol_labels[] = {
        "EEBackPlate",
        "EESRing",
        "EEBackQuad",
        "EEBackDee",
        "EEBackQuad_refl",
        "EEBackPlate_refl",
        "EESRing_refl",
    };
    EXPECT_VEC_EQ(expected_vol_labels, test_->get_volume_labels());

    static char const* const expected_vol_inst_labels[] = {
        "EEBackPlate@0",
        "EESRing@0",
        "EEBackQuad@0",
        "EEBackPlate@1",
        "EESRing@1",
        "EEBackQuad@1",
        "EEBackDee_PV",
    };
    EXPECT_VEC_EQ(expected_vol_inst_labels,
                  test_->get_volume_instance_labels());

    if (test_->g4world())
    {
        EXPECT_VEC_EQ(expected_vol_inst_labels, test_->get_g4pv_labels());
    }
}

//---------------------------------------------------------------------------//
void CmsEeBackDeeGeoTest::test_trace() const
{
    // Surface VecGeom needs lower safety tolerance
    real_type const safety_tol = test_->safety_tol();

    {
        SCOPED_TRACE("+z top");
        auto result = test_->track({50, 0.1, 360.1}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"EEBackPlate", "EEBackQuad"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[]
            = {"EEBackPlate", "EEBackQuad"};
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {5.4, 34.1};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0.1, 0.1};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
    {
        SCOPED_TRACE("+z bottom");
        auto result = test_->track({50, -0.1, 360.1}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"EEBackPlate_refl", "EEBackQuad_refl"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[]
            = {"EEBackPlate", "EEBackQuad"};
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {5.4, 34.1};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {0.099999999999956, 0.099999999999953};
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
