//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoTests.cc
//---------------------------------------------------------------------------//
#include "GeoTests.hh"

#include <string_view>

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/Turn.hh"
#include "corecel/sys/Version.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/CheckedGeoTrackView.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/Types.hh"

#include "GenericGeoResults.hh"
#include "GenericGeoTestInterface.hh"
#include "TestMacros.hh"
#include "UnitUtils.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
auto const vecgeom_version = celeritas::Version::from_string(
    CELERITAS_USE_VECGEOM ? cmake::vecgeom_version : "0.0.0");
auto const geant4_version = celeritas::Version::from_string(
    CELERITAS_USE_GEANT4 ? cmake::geant4_version : "0.0.0");

auto make_geo_track_view(GenericGeoTestInterface& gti,
                         Real3 const& pos,
                         Real3 const& dir)
{
    auto tracker = gti.make_checked_track_view();
    tracker = gti.make_initializer(pos, dir);
    return tracker;
}

BoundingBox<> calc_expected_bbox(std::string_view geo_type, Real3 lo, Real3 hi)
{
    if (geo_type == "VecGeom")
    {
        // VecGeom bumps bounding boxes
        lo -= 0.001;
        hi += 0.001;
    }
    return BoundingBox<>{lo, hi};
}

void fixup_orange(GenericGeoTestInterface const& interface,
                  GenericGeoTrackingResult& ref,
                  GenericGeoTrackingResult& result,
                  std::string_view world_name = "world")
{
    if (interface.geometry_type() != "ORANGE")
        return;

    // Delete within-world safeties
    for (auto i : range(std::max(ref.volumes.size(), result.volumes.size())))
    {
        if (ref.volumes[i] == world_name)
        {
            result.halfway_safeties[i] = 0;
            ref.halfway_safeties[i] = 0;
        }
    }
}

void delete_orange_safety(GenericGeoTestInterface const& interface,
                          GenericGeoTrackingResult& ref,
                          GenericGeoTrackingResult& result)
{
    if (interface.geometry_type() != "ORANGE")
        return;

    ref.halfway_safeties.clear();
    result.halfway_safeties.clear();
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// CMS EE
//! Test geometry accessors
void CmsEeBackDeeGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();

    auto expected_bbox = calc_expected_bbox(
        test_->geometry_type(), {0., -177.5, 359.5}, {177.5, 177.5, 399.6});
    auto const& bbox = geo.bbox();
    EXPECT_VEC_NEAR(expected_bbox.lower(), to_cm(bbox.lower()), 1e-10);
    EXPECT_VEC_SOFT_EQ(expected_bbox.upper(), to_cm(bbox.upper()));
}

//---------------------------------------------------------------------------//
void CmsEeBackDeeGeoTest::test_trace() const
{
    // Surface VecGeom needs lower safety tolerance
    {
        SCOPED_TRACE("+z top");
        auto result = test_->track({50, 0.1, 360.1}, {0, 0, 1});

        GenericGeoTrackingResult ref;
        ref.volumes = {"EEBackPlate", "EEBackQuad"};
        ref.volume_instances = {"EEBackPlate@0", "EEBackQuad@0"};
        ref.distances = {5.4, 34.1};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {0.1, 0.1};

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("+z bottom");
        auto result = test_->track({50, -0.1, 360.1}, {0, 0, 1});

        GenericGeoTrackingResult ref;
        ref.volumes = {"EEBackPlate_refl", "EEBackQuad_refl"};
        ref.volume_instances = {"EEBackPlate@1", "EEBackQuad@1"};
        ref.distances = {5.4, 34.1};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {0.099999999999956, 0.099999999999953};

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// CMSE
//---------------------------------------------------------------------------//
void CmseGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("Center +z");
        auto result = test_->track({0, 0, -4000}, {0, 0, 1});
        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = {"CMStoZDC", "BEAM3", "BEAM2", "BEAM1", "BEAM", "BEAM",
            "BEAM1", "BEAM2", "BEAM3", "CMStoZDC", "CMSE", "ZDC", "CMSE",
            "ZDCtoFP420", "CMSE",};
        ref.volume_instances = {"CMStoZDC@1", "BEAM3@1", "BEAM2@1", "BEAM1@1",
            "BEAM@1", "BEAM@0", "BEAM1@0", "BEAM2@0", "BEAM3@0", "CMStoZDC@0",
            "CMSE", "ZDC@0", "CMSE", "ZDCtoFP420@0", "CMSE",};
        ref.distances = {1300, 1096.95, 549.15, 403.9, 650, 650, 403.9, 549.15,
            1096.95, 11200, 10, 180, 910, 24000, 6000,};
        ref.dot_normal = {}; // All normals are along track dir
        ref.halfway_safeties = {100, 2.15, 10.302730220674, 13.023518051921,
            6.95, 6.95, 13.023518051922, 10.302730220675, 2.15, 100, 5, 8, 100,
            100, 100,};
        if (test_->geometry_type() == "VecGeom" && CELERITAS_VECGEOM_SURFACE)
        {
            // Surface vecgeom underestimates some safety near internal
            // boundaries
            ref.halfway_safeties = {100, 2.15,
                9.62498950958252, 13.023518051922, 6.95, 6.95, 13.023518051922,
                9.62498950958252, 2.15, 100, 5, 8, 100, 100, 100};
        }
        // clang-format on

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Offset +z");
        auto result = test_->track({30, 30, -4000}, {0, 0, 1});
        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = {"CMStoZDC", "OQUA", "VCAL", "OQUA", "CMSE", "TotemT1",
            "CMSE", "MUON", "CALO", "Tracker", "CALO", "MUON", "CMSE",
            "TotemT1", "CMSE", "OQUA", "VCAL", "OQUA", "CMStoZDC", "CMSE",
            "ZDCtoFP420", "CMSE",};
        ref.volume_instances = {"CMStoZDC@1", "OQUA@1", "VCAL@1", "OQUA@1",
            "CMSE", "TotemT1@1", "CMSE", "MUON", "CALO", "Tracker", "CALO",
            "MUON", "CMSE", "TotemT1@0", "CMSE", "OQUA@0", "VCAL@0", "OQUA@0",
            "CMStoZDC@0", "CMSE", "ZDCtoFP420@0", "CMSE",};
        ref.distances = {1300, 1419.95, 165.1, 28.95, 36, 300.1,
            94.858988388759, 100.94101161124, 260.9, 586.4, 260.9,
            100.94101161124, 94.858988388759, 300.1, 36, 28.95, 165.1, 1419.95,
            11200, 1100, 24000, 6000,};
        ref.dot_normal = {1, 1, 1, 1, 1, 1, 0.98776296532907, 1, 1, 1, 1,
            0.98776296532907, 1, 1, 1, 1, 1, 1, 1, 1, 1,};
        ref.halfway_safeties = {57.573593128807, 40.276406871193,
            29.931406871193, 14.475, 18, 28.702447147997, 29.363145173005,
            32.665765921596, 34.260814069425, 39.926406871193, 34.260814069425,
            32.665765921596, 29.363145173005, 28.702447147997, 18, 14.475,
            29.931406871193, 40.276406871193, 57.573593128807, 57.573593128807,
            57.573593128807, 57.573593128807,};
        // clang-format on
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Across muon");
        auto result = test_->track({-1000, 0, -48.5}, {1, 0, 0});
        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = {"OCMS", "MUON", "CALO", "Tracker", "CMSE", "BEAM",
            "CMSE", "Tracker", "CALO", "MUON", "OCMS",};
        ref.volume_instances = {"OCMS_PV", "MUON", "CALO", "Tracker", "CMSE",
            "BEAM@1", "CMSE", "Tracker", "CALO", "MUON", "OCMS_PV",};
        ref.distances = {170, 535, 171.7, 120.8, 0.15673306650251,
            4.6865338669951, 0.15673306650247, 120.8, 171.7, 535, 920,};
        ref.dot_normal = {1, 1, 1, 1, 0.99999815098379, 0.99999815098379, 1, 1,
            1, 1,};
        ref.halfway_safeties = {85, 267.5, 85.85, 60.4, 0.078366388350267,
            2.343262600759, 0.078366388350244, 60.4, 85.85, 267.5, 460,};
        // clang-format on
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Differs between G4/VG");
        auto result = test_->track({0, 0, 1328.0}, {1, 0, 0});
        GenericGeoTrackingResult ref;
        ref.volumes = {"BEAM2", "OQUA", "CMSE", "OCMS"};
        ref.volume_instances = {"BEAM2@0", "OQUA@0", "CMSE", "OCMS_PV"};
        ref.distances = {12.495, 287.505, 530, 920};
        ref.dot_normal = {};  // All normals are along track dir
        ref.halfway_safeties = {6.2475, 47.95, 242, 460};
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// FOUR LEVELS
//---------------------------------------------------------------------------//
//! Test geometry accessors
void FourLevelsGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();

    auto expected_bbox = calc_expected_bbox(
        test_->geometry_type(), {-24., -24., -24.}, {24., 24., 24.});
    auto const& bbox = geo.bbox();
    EXPECT_VEC_SOFT_EQ(expected_bbox.lower(), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ(expected_bbox.upper(), to_cm(bbox.upper()));
}

//---------------------------------------------------------------------------//
void FourLevelsGeoTest::test_consecutive_compute() const
{
    auto geo = make_geo_track_view(*test_, {-9, -10, -10}, {1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape2", test_->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    auto next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_NEAR(4.0, to_cm(geo.find_safety()), 1e-5);

    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_NEAR(4.0, to_cm(geo.find_safety()), 1e-5);

    // Find safety from a freshly initialized state
    geo = {from_cm({-9, -10, -10}), {1, 0, 0}};
    EXPECT_SOFT_NEAR(4.0, to_cm(geo.find_safety()), 1e-5);
}

//---------------------------------------------------------------------------//
void FourLevelsGeoTest::test_detailed_tracking() const
{
    bool const check_normal = test_->supports_surface_normal();
    Propagation next;
    {
        SCOPED_TRACE("rightward along corner");
        auto geo = make_geo_track_view(*test_, {-10, -10, -10}, {1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("Shape2", test_->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces up to a distance of 4 units away
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(4.0)));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(4.0)));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        geo.move_internal(from_cm(3.5));
        EXPECT_FALSE(geo.is_on_boundary());

        // Find one a bit further, then cross it
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(4.0)));
        EXPECT_SOFT_EQ(1.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_on_boundary());
        if (check_normal)
        {
            EXPECT_VEC_SOFT_EQ((Real3{1, 0, 0}), geo.normal());
        }
        EXPECT_EQ("Shape2", test_->volume_name(geo));
        ASSERT_NO_THROW(geo.cross_boundary());
        if (check_normal)
        {
            EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
        }
        EXPECT_EQ("Shape1", test_->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        // Find the next boundary and make sure that nearer distances aren't
        // accepted
        ASSERT_NO_THROW(next = geo.find_next_step());
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(0.5)));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
    }
    {
        SCOPED_TRACE("inside out");
        auto geo = make_geo_track_view(*test_, {-23.5, 6.5, 6.5}, {-1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("World", test_->volume_name(geo));

        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(2)));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_FALSE(geo.is_outside());
        if (check_normal)
        {
            EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), geo.normal());
        }
        ASSERT_NO_THROW(geo.cross_boundary());
        EXPECT_TRUE(geo.is_outside());
    }
    {
        SCOPED_TRACE("reentrant boundary");

        // Start inside box "Shape1" in the gap outside sphere "Shape2"
        auto geo = make_geo_track_view(*test_, {15.5, 10, 10}, {-1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("Shape1", test_->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces: we should hit the outside of the sphere Shape2
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(1.0)));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        // Move left to the boundary but scatter perpendicularly, tangent
        // upward to the sphere
        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_on_boundary());
        geo.set_dir({0, 1, 0});
        EXPECT_TRUE(geo.is_on_boundary());
        EXPECT_EQ("Shape1", test_->volume_name(geo));

        // Find the next step (to top edge of Shape1) but then scatter back
        // toward the sphere
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(10.0)));
        if (test_->geometry_type() == "VecGeom" && using_solids_vg
            && vecgeom_version >= Version{2, 0})
        {
            // Solids VG navig issues here - both v1,v2 work the same though
            EXPECT_GT(1e-12, to_cm(next.distance));
            GTEST_SKIP() << "FIXME: VG_solids navig issues: 1e-13 vs. 6";
        }
        EXPECT_SOFT_EQ(6, to_cm(next.distance));
        geo.set_dir({-1, 0, 0});
        EXPECT_VEC_SOFT_EQ((Real3{15, 10, 10}), to_cm(geo.pos()));
        EXPECT_EQ("Shape1", test_->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        // Check the distance to the sphere boundary again, then scatter
        // into the sphere (this may be a "bump": 1e-13 for surface VG, Geant4;
        // 1e-8 for volume VG; BUT exactly zero for ORANGE thanks to
        // "reentrant" logic)
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(20.0)));
        EXPECT_LE(next.distance, to_cm(1e-8));
        ASSERT_TRUE(next.boundary);
        if (next.distance > 0)
        {
            // ORANGE will not accept a zero-distance move-to-boundary call
            geo.move_to_boundary();
        }
        else if (CELERITAS_DEBUG)
        {
            EXPECT_THROW(geo.move_to_boundary(), DebugError);
        }
        EXPECT_TRUE(geo.is_on_boundary());

        // Enter the spehre
        ASSERT_NO_THROW(geo.cross_boundary());
        EXPECT_EQ("Shape2", test_->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        if (test_->geometry_type() == "Geant4")
        {
            // TODO: Geant4 does not allow crossing to new volume and returning
            // to old
            if (CELERITAS_DEBUG)
            {
                EXPECT_THROW(geo.cross_boundary(), DebugError);
            }
        }
        else
        {
            geo.set_dir(Real3{1, 0, 0});
            if (test_->geometry_type() == "VecGeom"
                && CELERITAS_VECGEOM_SURFACE)
            {
                // Assertion failure in NavStateTuple::PushDaughterImpl:
                // trying to push into a daughter but there are none
                // (pv ID 1)
                GTEST_SKIP() << "FIXME: vecgeom surface breaks";
            }

            ASSERT_NO_THROW(geo.cross_boundary());
            if (test_->geometry_type() == "VecGeom")
            {
                // FIXME: boundary crossing doesn't change volume like it
                // should
                EXPECT_EQ("Shape2", test_->volume_name(geo));
            }
            else
            {
                EXPECT_EQ("Shape1", test_->volume_name(geo));
            }
            geo.set_dir(Real3{-1, 0, 0});
            ASSERT_NO_THROW(geo.cross_boundary());
        }
        EXPECT_EQ("Shape2", test_->volume_name(geo));

        // Now move just barely inside the sphere
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(1e-6)));
        EXPECT_FALSE(next.boundary);
        geo.move_internal(next.distance);
        EXPECT_FALSE(geo.is_on_boundary());

        // Exit the sphere
        geo.set_dir({1, 0, 0});
        ASSERT_NO_THROW(next = geo.find_next_step(from_cm(1)));
        EXPECT_LE(next.distance, from_cm(1e-5));
        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_on_boundary());
        if (check_normal)
        {
            EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
        }

        geo.cross_boundary();
        EXPECT_EQ("Shape1", test_->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());
        if (check_normal)
        {
            EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
        }
        EXPECT_EQ("Shape1", test_->volume_name(geo));

        // Test relocation without direction change on surface
        if (test_->geometry_type() == "Geant4")
        {
            if (CELERITAS_DEBUG)
            {
                EXPECT_THROW(geo.cross_boundary(), DebugError);
            }
        }
        else
        {
            // No crossing if direction not changed
            geo.cross_boundary();
            EXPECT_EQ("Shape1", test_->volume_name(geo));

            constexpr auto dx = real_type{1} / constants::sqrt_two;

            // No crossing if direction on boundary is not reentrant
            geo.set_dir(Real3{dx, dx, 0});
            geo.cross_boundary();
            EXPECT_EQ("Shape1", test_->volume_name(geo));
        }
    }
}

//---------------------------------------------------------------------------//
void FourLevelsGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("Rightward");
        auto result = test_->track({-10, -10, -10}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
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
        ref.volume_instances = {
            "Shape2",
            "Shape1",
            "env8",
            "World_PV",
            "env7",
            "Shape1",
            "Shape2",
            "Shape1",
            "env7",
            "World_PV",
        };
        ref.distances = {
            5,
            1,
            1,
            6,
            1,
            1,
            10,
            1,
            1,
            7,
        };
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {
            2.5,
            0.5,
            0.5,
            3,
            0.5,
            0.5,
            5,
            0.5,
            0.5,
            3.5,
        };

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("From just inside outside edge");
        auto result = test_->track({-24 + 0.001, 10., 10.}, {1, 0, 0});

        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = { "World", "Envelope", "Shape1", "Shape2", "Shape1",
            "Envelope", "World", "Envelope", "Shape1", "Shape2", "Shape1",
            "Envelope", "World", };
        ref.volume_instances = { "World_PV", "env2", "Shape1", "Shape2",
            "Shape1", "env2", "World_PV", "env1", "Shape1", "Shape2", "Shape1",
            "env1", "World_PV", };
        ref.distances = { 6.999, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7, };
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = { 3.4995, 0.5, 0.5, 5, 0.5, 0.5, 3, 0.5, 0.5, 5,
            0.5, 0.5, 3.5, };
        // clang-format on

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Leaving world");
        auto result = test_->track({-10, 10, 10}, {0, 1, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {"Shape2", "Shape1", "Envelope", "World"};
        ref.volume_instances = {"Shape2", "Shape1", "env2", "World_PV"};
        ref.distances = {5, 1, 2, 6};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {2.5, 0.5, 1, 3};

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Upward");
        auto result = test_->track({-10, 10, 10}, {0, 0, 1});

        GenericGeoTrackingResult ref;
        ref.volumes = {"Shape2", "Shape1", "Envelope", "World"};
        ref.volume_instances = {"Shape2", "Shape1", "env2", "World_PV"};
        ref.distances = {5, 1, 3, 5};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {2.5, 0.5, 1.5, 2.5};

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// LAR SPHERE
//---------------------------------------------------------------------------//
void LarSphereGeoTest::test_trace() const
{
    if (test_->geometry_type() == "VecGeom" && using_surface_vg)
    {
        GTEST_SKIP() << "Fails to cross +y";
    }

    bool const is_orange = test_->geometry_type() == "ORANGE";
    {
        SCOPED_TRACE("+y");
        auto result = test_->track({0, -120, 0}, {0, 1, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world",
            "detshell_bot",
            "sphere",
            "detshell_top",
            "world",
        };
        ref.volume_instances = {
            "world_PV",
            "detshell_bot_PV",
            "sphere_PV",
            "detshell_top_PV",
            "world_PV",
        };
        ref.distances = {10, 10, 200, 10, 890};
        ref.halfway_safeties = {5, 5, 100, 5, 445};
        ref.bumps = {};
        if (is_orange)
        {
            // TODO: at this exact point it ignores the spherical distance
            ref.halfway_safeties[2] = result.halfway_safeties[2];
        }

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
void LarSphereGeoTest::test_volume_stack() const
{
    {
        auto result = test_->volume_stack({0, 0, 0});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances = {"world_PV", "sphere_PV"};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto result = test_->volume_stack({0, -105, 0});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances = {"world_PV", "detshell_PV", "detshell_bot_PV"};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
// MULTI-LEVEL
//---------------------------------------------------------------------------//
void MultiLevelGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("high");
        auto result = test_->track({-19.9, 7.5, 0}, {1, 0, 0});

        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = { "world", "box", "sph", "box", "tri", "box", "world",
            "box", "sph", "box", "tri", "box", "world", };
        ref.volume_instances = { "world_PV", "topbox2", "boxsph2@0", "topbox2",
            "boxtri@0", "topbox2", "world_PV", "topbox1", "boxsph2@0",
            "topbox1", "boxtri@0", "topbox1", "world_PV", };
        ref.distances = { 2.4, 3, 4, 1.8452994616207, 2.3094010767585,
            3.8452994616207, 5, 3, 4, 1.8452994616207, 2.3094010767585,
            3.8452994616207, 6.5, };
        ref.dot_normal = { 1, 1, 1, 0.86602540378444, 0.86602540378444, 1, 1,
            1, 1, 0.86602540378444, 0.86602540378444, 1, };
        ref.halfway_safeties = { 1.2, 1.5, 2, 0.79903810567666, 1,
            1.6650635094611, 2.5, 1.5, 2, 0.79903810567666, 1, 1.6650635094611,
            3.25, };
        // clang-format on

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("low");
        auto result = test_->track({-19.9, -7.5, 0}, {1, 0, 0});

        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = { "world", "box", "sph", "box", "world", "box_refl",
            "sph_refl", "box_refl", "tri_refl", "box_refl", "world", };
        ref.volume_instances = { "world_PV", "topbox3", "boxsph2@0", "topbox3",
            "world_PV", "topbox4", "boxsph2@1", "topbox4", "boxtri@1",
            "topbox4", "world_PV", };
        ref.distances = { 2.4, 3, 4, 8, 5, 3, 4, 1.8452994616207,
            2.3094010767585, 3.8452994616207, 6.5, };
        ref.dot_normal = { 1, 1, 1, 1, 1, 1, 1, 0.86602540378444,
            0.86602540378444, 1, };
        ref.halfway_safeties = { 1.2, 1.5, 2, 3.0990195135928, 2.5, 1.5, 2,
            0.79903810567666, 1, 1.6650635094611, 3.25, };
        // clang-format on

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
auto MultiLevelGeoTest::get_test_points() -> VecR2
{
    // Include outer world and center sphere
    VecR2 points{R2{-5, 0}, R2{0, 0}};

    // Loop over outer and inner x and y signs
    for (auto signs : range(1 << 4))
    {
        auto get_sign = [signs](int i) {
            CELER_ASSERT(i < 4);
            return signs & (1 << i) ? -1 : 1;
        };
        R2 point{0, 0};
        point[0] += 2.75 * get_sign(0);
        point[1] += 2.75 * get_sign(1);
        point[0] += 10.0 * get_sign(2);
        point[1] += 10.0 * get_sign(3);
        points.push_back(point);
    }

    return points;
}

//---------------------------------------------------------------------------//
void MultiLevelGeoTest::test_volume_level() const
{
    std::vector<VolumeLevelId::size_type> all_levels;
    for (R2 xy : this->get_test_points())
    {
        auto geo = make_geo_track_view(*test_, {xy[0], xy[1], 0}, {0, 0, 1});
        VolumeLevelId id;
        if (!geo.is_outside())
        {
            id = geo.volume_level();
        }
        all_levels.push_back(id_to_int(id));
    }

    static unsigned int const expected_all_levels[] = {
        0u,
        1u,
        2u,
        1u,
        2u,
        2u,
        2u,
        1u,
        2u,
        2u,
        2u,
        2u,
        2u,
        1u,
        1u,
        2u,
        2u,
        2u,
    };
    EXPECT_VEC_EQ(expected_all_levels, all_levels);
}

void MultiLevelGeoTest::test_volume_stack() const
{
    std::vector<std::string> all_stacks;
    for (R2 xy : this->get_test_points())
    {
        auto result = test_->volume_stack({xy[0], xy[1], 0});
        all_stacks.emplace_back(to_string(join(result.volume_instances.begin(),
                                               result.volume_instances.end(),
                                               ",")));
    }

    static char const* const expected_all_stacks[] = {
        "world_PV",
        "world_PV,topsph1",
        "world_PV,topbox1,boxsph1@0",
        "world_PV,topbox1",
        "world_PV,topbox1,boxtri@0",
        "world_PV,topbox1,boxsph2@0",
        "world_PV,topbox2,boxsph1@0",
        "world_PV,topbox2",
        "world_PV,topbox2,boxtri@0",
        "world_PV,topbox2,boxsph2@0",
        "world_PV,topbox4,boxtri@1",
        "world_PV,topbox4,boxsph2@1",
        "world_PV,topbox4,boxsph1@1",
        "world_PV,topbox4",
        "world_PV,topbox3",
        "world_PV,topbox3,boxsph2@0",
        "world_PV,topbox3,boxsph1@0",
        "world_PV,topbox3,boxtri@0",
    };
    EXPECT_VEC_EQ(expected_all_stacks, all_stacks);
}

//---------------------------------------------------------------------------//
// OPTICAL SURFACES
//---------------------------------------------------------------------------//
void OpticalSurfacesGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("Through tubes");
        auto result = test_->track({0, 0, -21}, {0, 0, 1});

        GenericGeoTrackingResult ref;
        ref.volumes = {"world", "tube2", "tube1_mid", "tube2", "world"};
        ref.volume_instances = {
            "world_PV",
            "tube2_below_pv",
            "tube1_mid_pv",
            "tube2_above_pv",
            "world_PV",
        };
        ref.distances = {1, 10, 20, 10, 80};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {0.5, 5, 10, 5, 40};

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Across tube through lAr");
        auto result = test_->track({-11, 0, 0}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {"world", "tube1_mid", "world", "lar_sphere", "world"};
        ref.volume_instances
            = {"world_PV", "tube1_mid_pv", "world_PV", "lar_pv", "world_PV"};
        ref.distances = {1, 20, 5, 10, 75};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {0.5, 10, 2.5, 5, 37.5};

        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// POLYHEDRA
//---------------------------------------------------------------------------//
void PolyhedraGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("tri");
        auto result = test_->track({-6, 4.01, 0}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world",
            "tri",
            "world",
            "tri_third",
            "world",
            "tri_half",
            "world",
            "tri_full",
            "world",
        };
        ref.volume_instances = {
            "world_PV",
            "tri0_pv",
            "world_PV",
            "tri30_pv",
            "world_PV",
            "tri60_pv",
            "world_PV",
            "tri90_pv",
            "world_PV",
        };
        ref.distances = {
            1,
            2.9826794919243,
            0.70352222243164,
            2.3816157604626,
            0.94950303325711,
            2.9826794919243,
            2,
            2.9826794919243,
            10.017320508076,
        };
        ref.dot_normal = {
            1,
            0.5,
            0.76604444311898,
            0.93969262078591,
            0.5,
            1,
            1,
            0.5,
        };
        ref.halfway_safeties = {
            0.5,
            0.74566987298108,
            0.26946464455223,
            0.91221175947349,
            0.44612049688277,
            0.74566987298108,
            1,
            0.74566987298108,
            4.5,
        };

        auto tol = test_->tracking_tol();
        fixup_orange(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("quad");
        auto result = test_->track({-6, 0.01, 0}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world",
            "quad",
            "world",
            "quad_third",
            "world",
            "quad_half",
            "world",
            "quad_full",
            "world",
        };
        ref.volume_instances = {
            "world_PV",
            "quad0_pv",
            "world_PV",
            "quad30_pv",
            "world_PV",
            "quad60_pv",
            "world_PV",
            "quad90_pv",
            "world_PV",
        };
        ref.distances = {
            0.5957864376269,
            2.8084271247462,
            1.5631897491411,
            2.0705523608202,
            1.9620443276656,
            2,
            1.5957864376269,
            2.8084271247462,
            10.595786437627,
        };
        ref.dot_normal = {
            0.70710678118655,
            0.70710678118655,
            0.96592582628907,
            0.96592582628907,
            1,
            1,
            0.70710678118655,
            0.70710678118655,
        };
        ref.halfway_safeties = {
            0.28806684196341,
            0.99292893218813,
            0.75496267504288,
            0.9896472381959,
            0.94759464420809,
            0.99,
            0.78795667663408,
            0.99292893218813,
            4.5,
        };

        if (test_->geometry_type() == "VecGeom" && using_surface_vg)
        {
            // TODO: check if polyhedra safety can be improved in vg2.x-surface
            // Geant4 has a different safety for the halfway point
            ref.halfway_safeties[0] = 0.210641235113144;
            ref.halfway_safeties[6] = 0.56419426202774;
        }

        auto tol = test_->tracking_tol();
        fixup_orange(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("penta");
        auto result = test_->track({-6, -4.01, 0}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world",
            "penta",
            "world",
            "penta_third",
            "world",
            "penta_half",
            "world",
            "penta_full",
            "world",
        };
        ref.volume_instances = {
            "world_PV",
            "penta0_pv",
            "world_PV",
            "penta30_pv",
            "world_PV",
            "penta60_pv",
            "world_PV",
            "penta90_pv",
            "world_PV",
        };
        ref.distances = {
            1,
            2.2288025522197,
            1.6810134561273,
            2.1103990209013,
            1.7509824185319,
            2.2288025522197,
            2,
            2.2288025522197,
            10.77119744778,
        };
        ref.dot_normal = {
            1,
            0.80901699437495,
            0.9135454576426,
            0.97814760073381,
            0.80901699437495,
            1,
            1,
            0.80901699437495,
        };
        ref.halfway_safeties = {
            0.5,
            0.90156957092601,
            0.76944173562526,
            0.96397271967888,
            0.85635962580704,
            0.90156957092601,
            1,
            0.90156957092601,
            4.5,
        };

        if (test_->geometry_type() == "VecGeom" && using_surface_vg)
        {
            // TODO: check if polyhedra safety can be improved in vg2.x-surface
            // Geant4 has a different safety for the halfway point
            ref.halfway_safeties[2] = 0.679982662200928;
            ref.halfway_safeties[8] = 4.35703563690186;
        }

        auto tol = test_->tracking_tol();
        fixup_orange(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("hex");
        auto result = test_->track({-6, -8.01, 0}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world",
            "hex",
            "world",
            "hex_third",
            "world",
            "hex_half",
            "world",
            "hex_full",
            "world",
        };
        ref.volume_instances = {
            "world_PV",
            "hex0_pv",
            "world_PV",
            "hex30_pv",
            "world_PV",
            "hex60_pv",
            "world_PV",
            "hex90_pv",
            "world_PV",
        };
        ref.distances = {
            0.85107296431264,
            2.2978540713747,
            1.8338830826198,
            2.0308532237715,
            1.9863366579213,
            2,
            1.8510729643126,
            2.2978540713747,
            10.851072964313,
        };
        ref.dot_normal = {
            0.86602540378444,
            0.86602540378444,
            0.98480775301221,
            0.98480775301221,
            1,
            1,
            0.86602540378444,
            0.86602540378444,
        };
        ref.halfway_safeties = {
            0.41988207740847,
            0.99,
            0.90301113894096,
            0.99120614758428,
            0.97807987040665,
            0.99133974596216,
            0.9198173396894,
            0.99,
            4.5,
        };
        if (test_->geometry_type() == "VecGeom" && using_surface_vg)
        {
            // TODO: check if polyhedra safety can be improved in vg2.x-surface
            // Geant4 has a different safety for the halfway point
            ref.halfway_safeties[0] = 0.368524014949799;
            ref.halfway_safeties[2] = 0.897850394248962;
            ref.halfway_safeties[4] = 0.966398000717163;
            ref.halfway_safeties[6] = 0.801536321640015;
        }

        auto tol = test_->tracking_tol();
        // Bump the tolerance by 25% for safety comparisons only: this became
        // necessarily when polyhedra started using StackedExtrudedPolygons
        tol.safety *= 1.25;
        fixup_orange(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// REPLICA
//---------------------------------------------------------------------------//

void ReplicaGeoTest::test_trace() const
{
    auto tol = test_->tracking_tol();
    if (test_->geometry_type() == "Geant4")
    {
        // Replica volumes track less accurately with geant4
        tol.distance *= 10;
    }

    {
        SCOPED_TRACE("Center +z");
        auto result = test_->track({0, 0.5, -990}, {0, 0, 1}, 75);

        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world",    "firstArm",   "hodoscope1", "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "chamber1", "wirePlane1", "chamber1",   "firstArm",
            "world",    "magnetic",   "world",      "secondArm",
            "chamber2", "wirePlane2", "chamber2",   "secondArm",
            "world",
        };
        ref.volume_instances = {
            "world_PV",   "firstArm",   "hodoscope1@7", "firstArm",
            "chamber1@0", "wirePlane1", "chamber1@0",   "firstArm",
            "chamber1@1", "wirePlane1", "chamber1@1",   "firstArm",
            "chamber1@2", "wirePlane1", "chamber1@2",   "firstArm",
            "chamber1@3", "wirePlane1", "chamber1@3",   "firstArm",
            "chamber1@4", "wirePlane1", "chamber1@4",   "firstArm",
            "world_PV",   "magnetic",   "world_PV",     "fSecondArmPhys",
            "chamber2@0", "wirePlane2", "chamber2@0",   "fSecondArmPhys",
            "world_PV",
        };
        ref.distances = { 190, 149.5, 1, 48.5, 0.99, 0.02, 0.99, 48, 0.99,
            0.02, 0.99, 48, 0.99, 0.02, 0.99, 48, 0.99, 0.02, 0.99, 48, 0.99,
            0.02, 0.99, 199, 100, 200, 73.205080756887, 114.31535329955,
            1.1431535329955, 0.023094010767585, 1.1431535329955,
            110.17016486681, 600, };
        ref.dot_normal = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0.86602540378444, 0.86602540378444,
            0.86602540378444, 0.86602540378444, 0.86602540378444, 0.5, };
        ref.halfway_safeties = { 95, 74.75, 0.5, 24.25, 0.495, 0.01, 0.495, 24,
            0.495, 0.01, 0.495, 24, 0.495, 0.01, 0.495, 24, 0.495, 0.01, 0.495,
            24, 0.495, 0.01, 0.495, 99.5, 50, 99.5, 31.698729810778, 49.5,
            0.495, 0.01, 0.495, 22.457458783298, 150, };
        // clang-format on

        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("Second arm");
        Real3 dir{0, 0, 0};
        sincos(Turn{-30.0 / 360.}, &dir[0], &dir[2]);
        auto result = test_->track({0.125, 0.5, 0.0625}, dir, 100);

        // clang-format off
        GenericGeoTrackingResult ref;
        ref.volumes = {
            "magnetic",     "world",       "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "chamber2",
            "wirePlane2",   "chamber2",    "secondArm",    "hodoscope2",
            "secondArm",    "cell",        "secondArm",    "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "HadCalLayer",
            "HadCalScinti", "HadCalLayer", "HadCalScinti", "world",
        };
        ref.volume_instances = {
            "magnetic",          "world_PV",          "fSecondArmPhys",
            "chamber2@0",        "wirePlane2",        "chamber2@0",
            "fSecondArmPhys",    "chamber2@1",        "wirePlane2",
            "chamber2@1",        "fSecondArmPhys",    "chamber2@2",
            "wirePlane2",        "chamber2@2",        "fSecondArmPhys",
            "chamber2@3",        "wirePlane2",        "chamber2@3",
            "fSecondArmPhys",    "chamber2@4",        "wirePlane2",
            "chamber2@4",        "fSecondArmPhys",    "hodoscope2@12",
            "fSecondArmPhys",    "cell_param@42",     "fSecondArmPhys",
            "HadCalLayer_PV@0",  "HadCalScinti",      "HadCalLayer_PV@1",
            "HadCalScinti",      "HadCalLayer_PV@2",  "HadCalScinti",
            "HadCalLayer_PV@3",  "HadCalScinti",      "HadCalLayer_PV@4",
            "HadCalScinti",      "HadCalLayer_PV@5",  "HadCalScinti",
            "HadCalLayer_PV@6",  "HadCalScinti",      "HadCalLayer_PV@7",
            "HadCalScinti",      "HadCalLayer_PV@8",  "HadCalScinti",
            "HadCalLayer_PV@9",  "HadCalScinti",      "HadCalLayer_PV@10",
            "HadCalScinti",      "HadCalLayer_PV@11", "HadCalScinti",
            "HadCalLayer_PV@12", "HadCalScinti",      "HadCalLayer_PV@13",
            "HadCalScinti",      "HadCalLayer_PV@14", "HadCalScinti",
            "HadCalLayer_PV@15", "HadCalScinti",      "HadCalLayer_PV@16",
            "HadCalScinti",      "HadCalLayer_PV@17", "HadCalScinti",
            "HadCalLayer_PV@18", "HadCalScinti",      "HadCalLayer_PV@19",
            "HadCalScinti",      "world_PV",
        };
        ref.distances = { 100.00827610654, 50.000097305727, 99, 0.99, 0.02,
            0.99, 48, 0.99, 0.02, 0.99, 48, 0.99, 0.02, 0.99, 48, 0.99, 0.02,
            0.99, 48, 0.99, 0.02, 0.99, 48.5, 1, 184.5, 30, 35, 4, 1, 4, 1, 4,
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4,
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 304.61999618334, };
        ref.dot_normal = { 0.99999902694273, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, };
        ref.halfway_safeties = { 50.004040731528, 25.000029191686, 49.5, 0.495,
            0.01, 0.495, 24, 0.495, 0.01, 0.495, 24, 0.495, 0.01, 0.495, 24,
            0.495, 0.0099999999999545, 0.495, 24, 0.495, 0.01, 0.495, 24.25,
            0.5, 92.25, 0.13950317547318, 17.5, 0.13950317547321,
            0.13950317547316, 0.13950317547316, 0.13950317547321,
            0.13950317547312, 0.13950317547311, 0.13950317547311,
            0.13950317547311, 0.13950317547311, 0.13950317547316,
            0.13950317547316, 0.13950317547316, 0.13950317547316,
            0.13950317547315, 0.1395031754732, 0.1395031754732,
            0.13950317547311, 0.13950317547306, 0.13950317547311,
            0.13950317547311, 0.13950317547306, 0.13950317547306,
            0.13950317547306, 0.13950317547306, 0.13950317547306,
            0.13950317547306, 0.13950317547306, 0.13950317547306,
            0.1395031754731, 0.1395031754731, 0.1395031754731, 0.1395031754731,
            0.1395031754731, 0.13950317547305, 0.13950317547305,
            0.13950317547305, 0.13950317547305, 0.13950317547305,
            0.13950317547309, 0.13950317547309, 131.90432759775, };
        // clang-format on
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
        {
            // All along the track to soft precision
            ref.clear_boring_normals();
        }

        delete_orange_safety(*test_, ref, result);
        if (test_->geometry_type() != "VecGeom"
            || vecgeom_version < Version{2, 0} || CELERITAS_VECGEOM_SURFACE)
        {
            // TODO: VecGemo 2.x-solids returns wrong distance values
            EXPECT_REF_NEAR(ref, result, tol);
        }
    }
}

//---------------------------------------------------------------------------//
void ReplicaGeoTest::test_volume_stack() const
{
    {
        auto result = test_->volume_stack({-400, 0.1, 650});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances = {
            "world_PV",
            "fSecondArmPhys",
            "HadCalorimeter",
            "HadCalColumn_PV@4",
            "HadCalCell_PV@1",
            "HadCalLayer_PV@2",
        };
        EXPECT_REF_EQ(ref, result);
    }
    {
        // Geant4 gets stuck here (it's close to a boundary)
        auto result = test_->volume_stack({-342.5, 0.1, 593.22740159234});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances
            = {"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@42"};
        if ((test_->geometry_type() == "VecGeom" && !CELERITAS_VECGEOM_SURFACE)
            || (test_->geometry_type() == "ORANGE"
                && (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)))
        {
            // Slightly different answers
            ref.volume_instances.pop_back();
            ref.volume_instances.pop_back();
        }
        if (test_->geometry_type() != "VecGeom"
            || vecgeom_version < Version{2, 0} || CELERITAS_VECGEOM_SURFACE)
        {
            // TODO: VecGemo 2.x-solids returns wrong volume instances
            EXPECT_REF_EQ(ref, result);
        }
    }
    {
        // A bit further along from the stuck point
        auto result = test_->volume_stack({-343, 0.1, 596});
        GenericGeoVolumeStackResult ref;
        ref.volume_instances
            = {"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@42"};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
// SOLIDS
//---------------------------------------------------------------------------//
//! Test geometry accessors
void SolidsGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();

    auto expected_bbox = calc_expected_bbox(
        test_->geometry_type(), {-600., -300., -75.}, {600., 300., 75.});
    auto const& bbox = geo.bbox();
    EXPECT_VEC_SOFT_EQ(expected_bbox.lower(), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ(expected_bbox.upper(), to_cm(bbox.upper()));
}

//---------------------------------------------------------------------------//
void SolidsGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("Upper +x");
        auto result = test_->track({-575, 125, 0.5}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "World",     "hype1",    "World",    "hype1",     "World",
            "para1",     "World",    "tube100",  "World",     "boolean1",
            "World",     "boolean1", "World",    "polyhedr1", "World",
            "polyhedr1", "World",    "ellcone1", "World",
        };
        ref.volume_instances = {
            "World_PV", "hype1_PV",     "World_PV", "hype1_PV",
            "World_PV", "para1_PV",     "World_PV", "tube100_PV",
            "World_PV", "boolean1_PV",  "World_PV", "boolean1_PV",
            "World_PV", "polyhedr1_PV", "World_PV", "polyhedr1_PV",
            "World_PV", "ellcone1_PV",  "World_PV",
        };
        ref.distances = {
            175.99886751197,
            4.0003045405969,
            40.001655894868,
            4.0003045405969,
            71.165534178636,
            60,
            74.833333333333,
            4,
            116,
            12.5,
            20,
            17.5,
            191.92750632007,
            26.020708495029,
            14.10357036981,
            26.020708495029,
            86.977506320066,
            9.8999999999999,
            220.05,
        };
        ref.dot_normal = {
            0.99998974040889,
            0.99999451629649,
            0.99999451629649,
            0.99998974040889,
            0.83205029433784,
            0.83205029433784,
            1,
            1,
            1,
            1,
            1,
            1,
            -1.2246467991474e-16,
            0.92346406713976,
            0.92346406713976,
            0.91834027967581,
            0.99503719020999,
            0.99503719020999,
        };
        ref.halfway_safeties = {
            74.5,
            1.9994549442736,
            20.000718268824,
            1.9994549442736,
            29.606651830022,
            24.961508830135,
            31.132548513141,
            2,
            42,
            6.25,
            9.5,
            8.75,
            74.5,
            0,
            6.5120702274482,
            11.947932358344,
            43.183743254945,
            4.9254340915394,
            74.5,
        };

        if (test_->geometry_type() == "VecGeom")
        {
            // v1.2.10: unknown differences outside hyperboloid
            ref.halfway_safeties[1] = 1.99361986757606;
            ref.halfway_safeties[3] = 1.99361986757606;
        }

        if (test_->geometry_type() != "VecGeom"
            || vecgeom_version < Version{2, 0} || CELERITAS_VECGEOM_SURFACE)
        {
            // TODO: VecGemo 2.x-solids still missing some shapes
            auto tol = test_->tracking_tol();
            EXPECT_REF_NEAR(ref, result, tol);
        }
    }
    {
        SCOPED_TRACE("Center -x");
        auto result = test_->track({575, 0, 0.50}, {-1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "World", "ellipsoid1", "World", "polycone1", "World", "polycone1",
            "World", "sphere1",    "World", "box500",    "World", "cone1",
            "World", "trd1",       "World", "parabol1",  "World", "trd2",
            "World", "xtru1",      "World",
        };
        ref.volume_instances = {
            "World_PV", "ellipsoid1_PV", "World_PV", "polycone1_PV",
            "World_PV", "polycone1_PV",  "World_PV", "sphere1_PV",
            "World_PV", "box500_PV",     "World_PV", "cone1_PV",
            "World_PV", "trd1_PV",       "World_PV", "parabol1_PV",
            "World_PV", "reflNormal",    "World_PV", "xtru1_PV",
            "World_PV",
        };
        ref.distances = {
            180.00156256104,
            39.99687487792,
            94.90156256104,
            2,
            16.2,
            2,
            115.41481927853,
            39.482055599395,
            60.00312512208,
            50,
            73.06,
            53.88,
            83.01,
            30.1,
            88.604510136799,
            42.690979726401,
            88.61120889722,
            30.086602479158,
            1.4328892366113,
            15.880952380952,
            67.642857142857,
        };
        ref.dot_normal = {
            0.99998046627013,
            0.99998046627013,
            0,
            0,
            0.98058067569092,
            0.98058067569092,
            0.69670670934717,
            0.999921871948,
            1,
            1,
            0.99287683848692,
            0.99287683848692,
            0.99503719020999,
            0.99503719020999,
            0.96698859472697,
            0.96698859472697,
            0.99503719020999,
            0.99503719020999,
            0.99549547259395,
            0.98994949366117,
        };
        ref.halfway_safeties = {
            74.5,
            0.5,
            45.689062136067,
            0,
            8.0156097709407,
            0.98058067569092,
            41.027453049596,
            13.753706517458,
            30.00022317033,
            24.5,
            36.269790909927,
            24.5,
            41.2093531814,
            14.97530971266,
            35.6477449316,
            14.272587510357,
            35.651094311811,
            14.968644196913,
            0.71288903993994,
            6.5489918373272,
            33.481506089183,
        };
        if (test_->geometry_type() == "VecGeom")
        {
            // VecGeom v1.2.11 (path,Scalar) using G4VG v1.0.4+builtin and
            // Geant4 v11.3.1
            ref.halfway_safeties[4] = 7.82052980478031;
            ref.halfway_safeties[14] = 42.8397753718277;
            ref.halfway_safeties[15] = 18.8833925371992;
            ref.halfway_safeties[16] = 42.8430141842906;
        }
        if (test_->geometry_type() != "VecGeom"
            || vecgeom_version < Version{2, 0} || CELERITAS_VECGEOM_SURFACE)
        {
            // TODO: VecGemo 2.x-solids still missing some shapes
            auto tol = test_->tracking_tol();
            EXPECT_REF_NEAR(ref, result, tol);
        }
    }
    {
        SCOPED_TRACE("Lower +x");
        auto result = test_->track({-575, -125, 0.5}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "World",   "trd3_refl",  "trd3_refl", "World",    "arb8b",
            "World",   "arb8a",      "World",     "trap1",    "World",
            "tetrah1", "World",      "orb1",      "World",    "genPocone1",
            "World",   "genPocone1", "World",     "elltube1", "World",
        };
        ref.volume_instances = {
            "World_PV",      "reflected@1", "reflected@0",   "World_PV",
            "arb8b_PV",      "World_PV",    "arb8a_PV",      "World_PV",
            "trap1_PV",      "World_PV",    "tetrah1_PV",    "World_PV",
            "orb1_PV",       "World_PV",    "genPocone1_PV", "World_PV",
            "genPocone1_PV", "World_PV",    "elltube1_PV",   "World_PV",
        };
        ref.distances = {
            34.956698760421,
            30.086602479158,
            24.913397520842,
            70.093301239579,
            79.9,
            45.1,
            79.9,
            68.323075218214,
            33.591007606176,
            57.452189546021,
            53.886393227913,
            81.800459523757,
            79.99374975584,
            39.95312512208,
            15,
            60.1,
            15,
            59.95,
            40,
            205,
        };
        ref.dot_normal = {
            0.99503719020999,
            0.99503719020999,
            0.99503719020999,
            0.99503719020999,
            0.99503719020999,
            0.99503719020999,
            0.99503719020999,
            0.95838499854689,
            0.93313781368065,
            0.98803162409286,
            0.10907224622337,
            0.999921871948,
            0.999921871948,
            0.99503719020999,
            0.99503719020999,
            0.99503719020999,
            0.99503719020999,
            1,
            1,
        };
        ref.halfway_safeties = {
            17.391607656793,
            14.968644196913,
            12.394878533861,
            34.872720758987,
            39.751735748889,
            22.438088639235,
            33.070197064425,
            32.739905171863,
            15.672519698479,
            26.80540527207,
            2.9387549751221,
            4.4610799311799,
            39.5,
            19.877422680791,
            7.2794797676807,
            29.515478338297,
            0,
            29.826239776544,
            20,
            74.5,
        };
        if (test_->geometry_type() == "Geant4"
            && geant4_version < Version{11, 3})
        {
            // Older versions of Geant4 have a bug in Arb8 that overestimates
            // safety distance to twisted surfaces
            ref.halfway_safeties[4] = 38.205672682313;
            ref.halfway_safeties[6] = 38.803595749271;
        }
        else if (test_->geometry_type() == "VecGeom")
        {
            // VecGeom v1.2.11 (path,Scalar) using G4VG v1.0.4+builtin and
            // Geant4 v11.3.1
            ref.halfway_safeties[3] = 29.9966506197896;
            ref.halfway_safeties[4] = 27.7657728660916;
            ref.halfway_safeties[5] = 17.5;
            ref.halfway_safeties[6] = 21.8864641598878;
            ref.halfway_safeties[7] = 29.1115376091067;
            ref.halfway_safeties[13] = 19.0382940808067;
            ref.halfway_safeties[14] = 0.5;
            ref.halfway_safeties[17] = 28.6150602709819;
        }

        if (test_->geometry_type() == "Geant4"
            && ref.dot_normal.size() == result.dot_normal.size()
            && result.dot_normal[15] == 0.0)
        {
            CELER_LOG(warning) << "GenPocone normal seems to have a bug";
            ref.dot_normal[15] = result.dot_normal[15];
        }

        auto tol = test_->tracking_tol();
        if (test_->geometry_type() != "VecGeom"
            || vecgeom_version < Version{2, 0} || CELERITAS_VECGEOM_SURFACE)
        {
            // TODO: VecGemo 2.x-solids still missing some shapes
            EXPECT_REF_NEAR(ref, result, tol);
        }
    }
    {
        SCOPED_TRACE("Middle +y");
        auto result = test_->track({0, -250, 0.5}, {0, 1, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "World",
            "tetrah1",
            "World",
            "box500",
            "World",
            "boolean1",
            "World",
            "boolean1",
            "World",
        };
        ref.volume_instances = {
            "World_PV",
            "tetrah1_PV",
            "World_PV",
            "box500_PV",
            "World_PV",
            "boolean1_PV",
            "World_PV",
            "boolean1_PV",
            "World_PV",
        };
        ref.distances = {
            105.03085028998,
            20.463165522069,
            99.505984187954,
            50,
            75,
            15,
            20,
            15,
            150,
        };
        ref.dot_normal = {
            0.92847669088526,
            0.70710678118655,
            1,
            1,
            1,
            1,
            1,
            1,
        };
        ref.halfway_safeties = {
            48.759348159052,
            7.2348215525988,
            35.180678093972,
            24.5,
            37.5,
            7.5,
            7.5,
            7.5,
            74.5,
        };

        if (test_->geometry_type() != "VecGeom"
            || vecgeom_version < Version{2, 0} || CELERITAS_VECGEOM_SURFACE)
        {
            // TODO: VecGemo 2.x-solids still missing some shapes
            auto tol = test_->tracking_tol();
            EXPECT_REF_NEAR(ref, result, tol);
        }
    }
}

//---------------------------------------------------------------------------//
// SIMPLE CMS
//---------------------------------------------------------------------------//
void SimpleCmsGeoTest::test_detailed_tracking() const
{
    auto safety_tol = test_->tracking_tol().safety;
    bool const check_normal = test_->supports_surface_normal();

    auto geo = make_geo_track_view(*test_, {0, 0, 0}, {0, 0, 1});
    EXPECT_EQ("vacuum_tube", test_->volume_name(geo));

    auto next = geo.find_next_step(from_cm(100));
    EXPECT_SOFT_EQ(100, to_cm(next.distance));
    EXPECT_FALSE(next.boundary);
    geo.move_internal(from_cm(20));
    EXPECT_SOFT_NEAR(30, to_cm(geo.find_safety()), safety_tol);

    geo.set_dir({1, 0, 0});
    next = geo.find_next_step(from_cm(50));
    EXPECT_SOFT_EQ(30, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_FALSE(geo.is_outside());
    if (check_normal)
    {
        EXPECT_VEC_SOFT_EQ((Real3{1, 0, 0}), geo.normal());
    }

    geo.cross_boundary();
    if (check_normal)
    {
        EXPECT_VEC_SOFT_EQ((Real3{1, 0, 0}), geo.normal());
    }
    EXPECT_EQ("si_tracker", test_->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({30, 0, 20}), to_cm(geo.pos()));

    // Scatter to tangent
    geo.set_dir({0, 1, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(121.34661099511597, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_internal(from_cm(10));
    EXPECT_SOFT_NEAR(1.6227766016837926, to_cm(geo.find_safety()), safety_tol);

    // Move to boundary and scatter back inside
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(111.34661099511597, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    geo.set_dir({-1, 0, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(60., to_cm(next.distance));
}

//---------------------------------------------------------------------------//
void SimpleCmsGeoTest::test_trace() const
{
    bool const is_orange = test_->geometry_type() == "ORANGE";
    {
        SCOPED_TRACE("outward radially");
        auto result = test_->track({-75, 0, 0}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "si_tracker",
            "vacuum_tube",
            "si_tracker",
            "em_calorimeter",
            "had_calorimeter",
            "sc_solenoid",
            "fe_muon_chambers",
            "world",
        };
        ref.volume_instances = {
            "si_tracker_pv",
            "vacuum_tube_pv",
            "si_tracker_pv",
            "em_calorimeter_pv",
            "had_calorimeter_pv",
            "sc_solenoid_pv",
            "iron_muon_chambers_pv",
            "world_PV",
        };
        ref.distances = {45, 60, 95, 50, 100, 100, 325, 300};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {22.5, 30, 47.5, 25, 50, 50, 162.5, 150};

        if (is_orange)
        {
            // TODO: at this exact point it ignores the cylindrical distance
            ref.halfway_safeties[1] = result.halfway_safeties[1];
        }
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("backward along z");
        auto result = test_->track({25, 0, 701}, {0, 0, -1});

        GenericGeoTrackingResult ref;
        ref.volumes = {"world", "vacuum_tube", "world"};
        ref.volume_instances = {"world_PV", "vacuum_tube_pv", "world_PV"};
        ref.distances = {1, 1400, 1300};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {0.5, 5, 650};

        if (is_orange)
        {
            ref.halfway_safeties[2] = result.halfway_safeties[2];
        }
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// TESTEM3 NESTED
//---------------------------------------------------------------------------//
void TestEm3GeoTest::test_trace() const
{
    {
        auto result = test_->track({-20.1}, {1, 0, 0}, /*max_steps=*/250);
        result.volume_instances.clear();  // boring

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world", "pb",  "lar",   "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",    "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",   "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",    "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",   "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",    "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",   "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",    "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",   "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "pb",    "lar", "pb",  "lar", "pb",  "lar", "pb",
            "lar",   "pb",  "lar",   "pb",  "lar", "pb",  "lar", "pb",  "lar",
            "pb",    "lar", "world",
        };
        ref.volume_instances = {};
        ref.distances = {
            0.1,  0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 4,
        };
        ref.halfway_safeties = {
            0.050, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115, 0.285, 2,
        };

        auto tol = test_->tracking_tol();
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
        {
            tol.distance = 1e-5f;
        }
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// TESTEM3 FLAT
//---------------------------------------------------------------------------//
void TestEm3FlatGeoTest::test_trace() const
{
    {
        auto result = test_->track({-20.1}, {1, 0, 0}, 150);
        result.volume_instances.clear();  // boring

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "world",       "gap_0",  "absorber_0",  "gap_1",
            "absorber_1",  "gap_2",  "absorber_2",  "gap_3",
            "absorber_3",  "gap_4",  "absorber_4",  "gap_5",
            "absorber_5",  "gap_6",  "absorber_6",  "gap_7",
            "absorber_7",  "gap_8",  "absorber_8",  "gap_9",
            "absorber_9",  "gap_10", "absorber_10", "gap_11",
            "absorber_11", "gap_12", "absorber_12", "gap_13",
            "absorber_13", "gap_14", "absorber_14", "gap_15",
            "absorber_15", "gap_16", "absorber_16", "gap_17",
            "absorber_17", "gap_18", "absorber_18", "gap_19",
            "absorber_19", "gap_20", "absorber_20", "gap_21",
            "absorber_21", "gap_22", "absorber_22", "gap_23",
            "absorber_23", "gap_24", "absorber_24", "gap_25",
            "absorber_25", "gap_26", "absorber_26", "gap_27",
            "absorber_27", "gap_28", "absorber_28", "gap_29",
            "absorber_29", "gap_30", "absorber_30", "gap_31",
            "absorber_31", "gap_32", "absorber_32", "gap_33",
            "absorber_33", "gap_34", "absorber_34", "gap_35",
            "absorber_35", "gap_36", "absorber_36", "gap_37",
            "absorber_37", "gap_38", "absorber_38", "gap_39",
            "absorber_39", "gap_40", "absorber_40", "gap_41",
            "absorber_41", "gap_42", "absorber_42", "gap_43",
            "absorber_43", "gap_44", "absorber_44", "gap_45",
            "absorber_45", "gap_46", "absorber_46", "gap_47",
            "absorber_47", "gap_48", "absorber_48", "gap_49",
            "absorber_49", "world",
        };
        ref.distances = {
            0.1,  0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23,
            0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57, 0.23, 0.57,
            0.23, 0.57, 4,
        };
        ref.halfway_safeties = {
            0.0500, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115,  0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285,  0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115,  0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285,  0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115,  0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285,  0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115,  0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285,  0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115,  0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115,
            0.285,  0.115, 0.285, 0.115, 0.285, 0.115, 0.285, 0.115, 0.285,
            0.115,  0.285, 2,
        };
        auto tol = test_->tracking_tol();
        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
        {
            tol.distance = 1e-5f;
        }

        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// TILECAL PLUG
//---------------------------------------------------------------------------//
void TilecalPlugGeoTest::test_trace() const
{
    {
        SCOPED_TRACE("+z lo");
        auto result = test_->track({5.75, 0.01, -40}, {0, 0, 1});
        GenericGeoTrackingResult ref;
        ref.volumes = {
            "Tile_ITCModule",
            "Tile_Plug1Module",
            "Tile_Absorber",
            "Tile_Plug1Module",
        };
        ref.volume_instances = {
            "Tile_ITCModule_PV",
            "Tile_Plug1Module",
            "Tile_Absorber",
            "Tile_Plug1Module",
        };
        ref.distances = {22.9425, 0.115, 42, 37};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {9.7, 0.0575, 9.7, 9.7};
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        SCOPED_TRACE("+z hi");
        auto result = test_->track({6.25, 0.01, -40}, {0, 0, 1});
        GenericGeoTrackingResult ref;
        ref.volumes = {"Tile_ITCModule", "Tile_Absorber", "Tile_Plug1Module"};
        ref.volume_instances
            = {"Tile_ITCModule_PV", "Tile_Absorber", "Tile_Plug1Module"};
        ref.distances = {23.0575, 42, 37};
        // All surface normals are along track dir: ref.dot_normal = {}
        ref.halfway_safeties = {9.2, 9.2, 9.2};
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// TRANSFORMED BOX
//---------------------------------------------------------------------------//
//! Test geometry accessors
void TransformedBoxGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();

    auto expected_bbox = calc_expected_bbox(
        test_->geometry_type(), {-50., -50., -50.}, {50., 50., 50.});
    auto const& bbox = geo.bbox();
    EXPECT_VEC_SOFT_EQ(expected_bbox.lower(), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ(expected_bbox.upper(), to_cm(bbox.upper()));
}

//---------------------------------------------------------------------------//
void TransformedBoxGeoTest::test_trace() const
{
    {
        auto result = test_->track({0, 0, -25}, {0, 0, 1});
        GenericGeoTrackingResult ref;
        ref.volumes = {
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
        ref.volume_instances = {
            "world_PV",
            "transrot",
            "world_PV",
            "default",
            "rot",
            "default",
            "world_PV",
            "trans",
            "world_PV",
        };
        ref.distances = {
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
        ref.dot_normal = {
            0.5,
            0.5,
            1,
            1,
            1,
            1,
            1,
            1,
        };
        ref.halfway_safeties = {
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

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        auto result = test_->track({0.25, 0, -25}, {0, 0, 1});
        GenericGeoTrackingResult ref;
        ref.volumes = {
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
        ref.volume_instances = {
            "world_PV",
            "transrot",
            "world_PV",
            "default",
            "rot",
            "default",
            "world_PV",
            "trans",
            "world_PV",
        };
        ref.distances = {
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
        ref.dot_normal = {
            0.86602540378444,
            0.5,
            1,
            1,
            1,
            1,
            1,
            1,
        };
        ref.halfway_safeties = {
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

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        auto result = test_->track({0, 0.25, -25}, {0, 0, 1});

        GenericGeoTrackingResult ref;
        ref.volumes = {
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
        ref.volume_instances = {
            "world_PV",
            "transrot",
            "world_PV",
            "default",
            "rot",
            "default",
            "world_PV",
            "trans",
            "world_PV",
        };
        ref.distances = {
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
        ref.dot_normal = {
            0.5,
            0.5,
            1,
            1,
            1,
            1,
            1,
            1,
        };
        ref.halfway_safeties = {
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

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        auto result = test_->track({0.01, -20, 0.20}, {0, 1, 0});

        GenericGeoTrackingResult ref;
        ref.volumes = {"world", "enclosing", "tiny", "enclosing", "world"};
        ref.volume_instances
            = {"world_PV", "default", "rot", "default", "world_PV"};
        ref.distances
            = {18.5, 1.1250390198213, 0.75090449735279, 1.1240564828259, 48.5};
        ref.dot_normal = {1, 0.99879545620517, 0.99879545620517, 1};
        ref.halfway_safeties
            = {9.25, 0.56184193052552, 0.05, 0.56135125378224, 24.25};

        auto tol = test_->tracking_tol();
        delete_orange_safety(*test_, ref, result);
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// TRANSFORMED BOX
//---------------------------------------------------------------------------//
//! Test geometry accessors
void TwoBoxesGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();
    auto expected_bbox = calc_expected_bbox(
        test_->geometry_type(), {-500., -500., -500.}, {500., 500., 500.});
    auto const& bbox = geo.bbox();
    EXPECT_VEC_SOFT_EQ(expected_bbox.lower(), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ(expected_bbox.upper(), to_cm(bbox.upper()));
}

void TwoBoxesGeoTest::test_detailed_tracking() const
{
    bool const check_normal = test_->supports_surface_normal();
    auto safety_tol = test_->tracking_tol().safety;

    auto geo = make_geo_track_view(*test_, {0, 0, 0}, {0, 0, 1});
    EXPECT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", test_->volume_name(geo));

    // Shouldn't hit boundary
    auto next = geo.find_next_step(from_cm(1.25));
    EXPECT_SOFT_EQ(1.25, to_cm(next.distance));
    EXPECT_FALSE(next.boundary);

    geo.move_internal(from_cm(1.25));
    real_type expected_safety = 5 - 1.25;
    EXPECT_SOFT_NEAR(expected_safety, to_cm(geo.find_safety()), safety_tol);

    // Change direction and try again (hit)
    geo.set_dir({1, 0, 0});
    next = geo.find_next_step(from_cm(50));
    EXPECT_SOFT_EQ(5, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_FALSE(geo.is_outside());
    if (check_normal)
    {
        EXPECT_VEC_SOFT_EQ((Real3{1, 0, 0}), geo.normal());
    }
    geo.cross_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("world", test_->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({5, 0, 1.25}), to_cm(geo.pos()));
    if (geo.is_on_boundary() && CELERITAS_DEBUG)
    {
        // Don't check the safety distance on the boundary; we know by
        // definition it's zero
        EXPECT_THROW(geo.find_safety(), DebugError);
    }

    // Scatter to tangent along boundary
    constexpr real_type dx
        = (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE ? 1e-8 : 1e-4);
    geo.set_dir({dx, 1, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(500, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_internal(from_cm(2));

    // Scatter back inside
    geo.set_dir({-1, 0, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_TRUE(next.boundary);
    EXPECT_SOFT_NEAR(2 * dx, to_cm(next.distance), 1e-4);
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    if (check_normal)
    {
        EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), geo.normal());
    }

    geo.cross_boundary();
    if (!check_normal)
    {
        // Skip check
    }
    else if (test_->geometry_type() == "Geant4")
    {
        EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), geo.normal());
    }
    else
    {
        EXPECT_VEC_SOFT_EQ((Real3{1, 0, 0}), geo.normal());
    }

    EXPECT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", test_->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({5, 2, 1.25}), to_cm(geo.pos()));
}

/*!
 * Cross into a new volume and then reflect into the old.
 *
 * This is how optical physics is performed: we enter the new volume to
 * determine its characteristics, then apply the optical surface crossing,
 * which might reflect back into the original.
 */
void TwoBoxesGeoTest::test_reentrant() const
{
    bool const check_normal = test_->supports_surface_normal();
    constexpr auto dx = real_type{1} / constants::sqrt_two;

    // Starting left of edge (-), headed down right (+,-)
    CheckedGeoTrackView geo{test_->make_geo_track_view_interface()};
    geo = test_->make_initializer({5 - dx, dx, 0}, {dx, -dx, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", test_->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces up to a distance of 4 units away
    auto next = geo.find_next_step(from_cm(4.0));
    EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    // Move to boundary (-; +,-)
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    if (check_normal)
    {
        EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    }
    EXPECT_EQ("inner", test_->volume_name(geo));

    // Cross into the new volume, needed for optical physics (+; +,-)
    geo.cross_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    if (check_normal)
    {
        EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    }
    EXPECT_EQ("world", test_->volume_name(geo));

    // Reflect normal to surface  (+; -,-)
    geo.set_dir(Real3{-dx, -dx, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    if (check_normal)
    {
        EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    }
    EXPECT_EQ("world", test_->volume_name(geo));

    // Cross back into previous volume (-; -,-)
    if (CELERITAS_DEBUG && test_->geometry_type() == "Geant4")
    {
        // GeantGTV has an extra check because we know it can't do this :(
        EXPECT_THROW(geo.cross_boundary(), DebugError);
        GTEST_SKIP() << "Consecutive boundary crossing fails for G4";
    }
    else
    {
        // Typical case
        ASSERT_NO_THROW(geo.cross_boundary());
    }
    EXPECT_TRUE(geo.is_on_boundary());

    if (check_normal)
    {
        EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    }

    if (test_->geometry_type() == "VecGeom")
    {
        // VecGeom 1.2.11 seems to fail reentry *sometimes*: on the CI builds,
        // spack passes but docker fails (relwithdebinfo and debug)
        if ("world" == test_->volume_name(geo))
        {
            GTEST_SKIP() << "Unexpected failure to cross volume";
        }
        if ("[OUTSIDE]" == test_->volume_name(geo))
        {
            GTEST_SKIP() << "FIXME: Unexpected track location.";
        }
    }

    if (test_->geometry_type() == "VecGeom" && using_surface_vg)
    {
        // VecGeom with surfaces seems to have issues here
        EXPECT_EQ("[OUTSIDE]", test_->volume_name(geo));
        {
            GTEST_SKIP() << "FIXME: VecGeom v2.x-surface misses inner volume.";
        }
    }
    EXPECT_EQ("inner", test_->volume_name(geo));

    // Find the next boundary and make sure that nearer distances aren't
    // accepted
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(10 * dx, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    EXPECT_TRUE(geo.is_on_boundary());
}

/*!
 * Emulate an edge case with field propagation plus MSC.
 *
 * - Propagation moves to the boundary
 * - Field momentum update points the direction back inside
 * - MSC update points the direction back out
 */
void TwoBoxesGeoTest::test_reentrant_undo() const
{
    constexpr auto dx = real_type{1} / constants::sqrt_two;
    bool const check_normal = test_->supports_surface_normal();

    // Starting left of edge (-), headed down right (+,-)
    CheckedGeoTrackView geo{test_->make_geo_track_view_interface()};
    geo = test_->make_initializer({5 - dx, dx, 0}, {dx, -dx, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("inner", test_->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces up to a distance of 4 units away
    auto next = geo.find_next_step(from_cm(4.0));
    EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    // Propagate: move to boundary (-; +,-)
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("inner", test_->volume_name(geo));

    // Momentum update: point back inward (-; -,-)
    geo.set_dir(Real3{-dx, -dx, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("inner", test_->volume_name(geo));

    // Scatter: point back out (-; +,-)
    geo.set_dir(Real3{dx, -dx, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("inner", test_->volume_name(geo));

    // Crossing *will* change volumes (+; +,-)
    geo.cross_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("world", test_->volume_name(geo));
    if (check_normal)
    {
        EXPECT_NORMAL_EQUIV((Real3{1, 0, 0}), geo.normal());
    }

    // Make sure we're not intersecting by accident
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(10.0, to_cm(next.distance));
    EXPECT_FALSE(next.boundary);
    EXPECT_TRUE(geo.is_on_boundary());
}

/*!
 * Instead of crossing into a new volume, reflect without exiting.
 *
 * This simulates a looping track almost tangent to a geometry boundary.
 * The end-of-step direction is changed to account for the momentum vector's
 * end-of-step state, and the boundary isn't actually exited when we call cross
 * boundary.
 */
void TwoBoxesGeoTest::test_tangent() const
{
    constexpr auto dx = real_type{1} / constants::sqrt_two;

    // Starting left of edge (-), headed down right (+,-)
    CheckedGeoTrackView geo{test_->make_geo_track_view_interface()};
    {
        SCOPED_TRACE("in first volume");
        geo = test_->make_initializer({5 - dx, dx, 0}, {dx, -dx, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("inner", test_->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces up to a distance of 4 units away
        auto next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);

        // Move to boundary (-; +,-)
        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_on_boundary());
        EXPECT_EQ("inner", test_->volume_name(geo));

        // Reflect normal to surface (-; -,-)
        geo.set_dir(Real3{-dx, -dx, 0});
        EXPECT_TRUE(geo.is_on_boundary());
        EXPECT_EQ("inner", test_->volume_name(geo));
    }

    // Crossing should *not* change volumes (-; -,-)
    {
        SCOPED_TRACE("trying to cross");
        ASSERT_NO_THROW(geo.cross_boundary());
        EXPECT_TRUE(geo.is_on_boundary());
        if (test_->geometry_type() == "Geant4")
        {
            // FIXME: Geant4 changes volumes :(
            EXPECT_EQ("world", test_->volume_name(geo));
            GTEST_SKIP() << "Unexpected boundary crossing";
        }
        else if (test_->geometry_type() == "VecGeom"
                 && "world" == test_->volume_name(geo))
        {
            GTEST_SKIP() << "Unexpected boundary crossing";
        }
        EXPECT_EQ("inner", test_->volume_name(geo));
    }

    // Find the next boundary and make sure that nearer distances aren't
    // accepted
    {
        SCOPED_TRACE("checking internal distance");
        auto next = geo.find_next_step();
        EXPECT_SOFT_EQ(10.0 * dx, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
    }
}

//---------------------------------------------------------------------------//
void TwoBoxesGeoTest::test_trace() const
{
    {
        auto result = test_->track({0, 0.25, -25}, {0, 0., 1});
        GenericGeoTrackingResult ref;
        ref.volumes = {"world", "inner", "world"};
        ref.volume_instances = {"world_PV", "inner_PV", "world_PV"};
        ref.distances = {20, 10, 495};
        ref.halfway_safeties = {10, 4.75, 247.5};
        ref.bumps = {};
        auto tol = test_->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
// ZNENV
//---------------------------------------------------------------------------//
void ZnenvGeoTest::test_trace() const
{
    // NOTE: This tests the capability of the G4PVDivision conversion based on
    // an ALICE component
    static char const* const expected_mid_volumes[] = {
        "World", "ZNENV", "ZNST", "ZNST",  "ZNST",  "ZNST", "ZNST",
        "ZNST",  "ZNST",  "ZNST", "ZNST",  "ZNST",  "ZNST", "ZNST",
        "ZNST",  "ZNST",  "ZNST", "ZNST",  "ZNST",  "ZNST", "ZNST",
        "ZNST",  "ZNST",  "ZNST", "ZNENV", "World",
    };
    static real_type const expected_mid_distances[] = {
        6.38, 0.1,  0.32, 0.32, 0.32, 0.32, 0.32, 0.32,  0.32,
        0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,  0.32,
        0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.1,  46.38,
    };
    {
        auto result = test_->track({-10, 0.0001, 0}, {1, 0, 0});

        GenericGeoTrackingResult ref;
        ref.volumes.assign(std::begin(expected_mid_volumes),
                           std::end(expected_mid_volumes));
        ref.distances.assign(std::begin(expected_mid_distances),
                             std::end(expected_mid_distances));
        ref.volume_instances = {
            "World_PV",   "WorldBoxPV", "ZNST_PV@0", "ZNST_PV@1",
            "ZNST_PV@2",  "ZNST_PV@3",  "ZNST_PV@4", "ZNST_PV@5",
            "ZNST_PV@6",  "ZNST_PV@7",  "ZNST_PV@8", "ZNST_PV@9",
            "ZNST_PV@10", "ZNST_PV@0",  "ZNST_PV@1", "ZNST_PV@2",
            "ZNST_PV@3",  "ZNST_PV@4",  "ZNST_PV@5", "ZNST_PV@6",
            "ZNST_PV@7",  "ZNST_PV@8",  "ZNST_PV@9", "ZNST_PV@10",
            "WorldBoxPV", "World_PV",
        };
        ref.halfway_safeties = {
            3.19, 0.05, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,  1e-4,
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,  1e-4,
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.05, 23.19,
        };
        ref.bumps = {};

        auto tol = test_->tracking_tol();
        fixup_orange(*test_, ref, result, "World");
        EXPECT_REF_NEAR(ref, result, tol);
    }
    {
        auto result = test_->track({0.0001, -10, 0}, {0, 1, 0});

        GenericGeoTrackingResult ref;
        ref.volumes.assign(std::begin(expected_mid_volumes),
                           std::end(expected_mid_volumes));
        ref.distances.assign(std::begin(expected_mid_distances),
                             std::end(expected_mid_distances));
        ref.volume_instances = {
            "World_PV",  "WorldBoxPV", "ZNST_PV@0", "ZNST_PV@0", "ZNST_PV@0",
            "ZNST_PV@0", "ZNST_PV@0",  "ZNST_PV@0", "ZNST_PV@0", "ZNST_PV@0",
            "ZNST_PV@0", "ZNST_PV@0",  "ZNST_PV@0", "ZNST_PV@0", "ZNST_PV@0",
            "ZNST_PV@0", "ZNST_PV@0",  "ZNST_PV@0", "ZNST_PV@0", "ZNST_PV@0",
            "ZNST_PV@0", "ZNST_PV@0",  "ZNST_PV@0", "ZNST_PV@0", "WorldBoxPV",
            "World_PV",
        };
        ref.halfway_safeties = {
            3.19, 0.05, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,  1e-4,
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,  1e-4,
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.05, 23.19,
        };
        ref.bumps = {};

        auto tol = test_->tracking_tol();
        fixup_orange(*test_, ref, result, "World");
        if (using_solids_vg && vecgeom_version >= Version{2, 0})
        {
            GTEST_SKIP() << "FIXME: Znenv VecGeom model construction failure.";
        }
        EXPECT_REF_NEAR(ref, result, tol);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
