//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/Vecgeom.test.cc
//---------------------------------------------------------------------------//
#include "Vecgeom.test.hh"

#include <string_view>

#include "celeritas_cmake_strings.h"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Version.hh"
#include "celeritas/GenericGeoTestBase.hh"
#include "celeritas/ext/GeantGeoUtils.hh"
#include "celeritas/ext/VecgeomData.hh"
#include "celeritas/ext/VecgeomParams.hh"
#include "celeritas/ext/VecgeomTrackView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

// Since VecGeom is currently CUDA-only, we cannot use the TEST_IF_CELER_DEVICE
// macro (which also allows HIP).
#if CELERITAS_USE_CUDA
#    define TEST_IF_CELERITAS_CUDA(name) name
#else
#    define TEST_IF_CELERITAS_CUDA(name) DISABLED_##name
#endif

namespace
{
auto const vecgeom_version
    = celeritas::Version::from_string(celeritas_vecgeom_version);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
/*!
 * Preserve the VecGeom geometry across test cases.
 *
 * Test cases should be matched to unique geometries.
 */
class VecgeomTestBase : public GenericVecgeomTestBase
{
  public:
    //! Helper function: build with VecGeom using VGDML
    SPConstGeo load_vgdml(std::string_view filename) const
    {
        return std::make_shared<VecgeomParams>(
            this->test_data_path("celeritas", filename));
    }

    //! Helper function: build via Geant4 GDML reader
    SPConstGeo load_g4_gdml(std::string_view filename) const
    {
        return std::make_shared<VecgeomParams>(::celeritas::load_geant_geometry(
            this->test_data_path("celeritas", filename)));
    }
};

//---------------------------------------------------------------------------//
// FOUR-LEVELS TEST
//---------------------------------------------------------------------------//

class FourLevelsTest : public VecgeomTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_vgdml("four-levels.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, accessors)
{
    auto const& geom = *this->geometry();
    EXPECT_EQ(4, geom.max_depth());

    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-24.001, -24.001, -24.001}), bbox.lower());
    EXPECT_VEC_SOFT_EQ((Real3{24.001, 24.001, 24.001}), bbox.upper());

    ASSERT_EQ(4, geom.num_volumes());
    EXPECT_EQ("Shape2", geom.id_to_label(VolumeId{0}).name);
    EXPECT_EQ("Shape1", geom.id_to_label(VolumeId{1}).name);
    EXPECT_EQ("Envelope", geom.id_to_label(VolumeId{2}).name);
    EXPECT_EQ("World", geom.id_to_label(VolumeId{3}).name);
    EXPECT_EQ(Label("World", "0xdeadbeef"), geom.id_to_label(VolumeId{3}));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, detailed_track)
{
    {
        SCOPED_TRACE("rightward along corner");
        auto geo = this->make_geo_track_view({-10, -10, -10}, {1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces up to a distance of 4 units away
        auto next = geo.find_next_step(4.0);
        EXPECT_SOFT_EQ(4.0, next.distance);
        EXPECT_FALSE(next.boundary);
        next = geo.find_next_step(4.0);
        EXPECT_SOFT_EQ(4.0, next.distance);
        EXPECT_FALSE(next.boundary);
        geo.move_internal(3.5);
        EXPECT_FALSE(geo.is_on_boundary());

        // Find one a bit further, then cross it
        next = geo.find_next_step(4.0);
        EXPECT_SOFT_EQ(1.5, next.distance);
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        geo.cross_boundary();
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
        EXPECT_TRUE(geo.is_on_boundary());

        // Find the next boundary and make sure that nearer distances aren't
        // accepted
        next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, next.distance);
        EXPECT_TRUE(next.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
        next = geo.find_next_step(0.5);
        EXPECT_SOFT_EQ(0.5, next.distance);
        EXPECT_FALSE(next.boundary);
    }
    {
        SCOPED_TRACE("outside in");
        auto geo = this->make_geo_track_view({-25, 6.5, 6.5}, {1, 0, 0});
        EXPECT_TRUE(geo.is_outside());

        auto next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, next.distance);
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_TRUE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ(VolumeId{3}, geo.volume_id());
    }
    {
        SCOPED_TRACE("inside out");
        auto geo = this->make_geo_track_view({-23.5, 6.5, 6.5}, {-1, 0, 0});
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ(VolumeId{3}, geo.volume_id());

        auto next = geo.find_next_step(2);
        EXPECT_SOFT_EQ(0.5, next.distance);
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_FALSE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_TRUE(geo.is_outside());

        next = geo.find_next_step();
        EXPECT_GT(next.distance, 1e10);
        EXPECT_FALSE(next.boundary);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, reentrant_boundary)
{
    auto geo = this->make_geo_track_view();
    geo = GeoTrackInitializer{{15.5, 10, 10}, {-1, 0, 0}};
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces: we should hit the outside of the sphere Shape2
    auto next = geo.find_next_step(1.0);
    EXPECT_SOFT_EQ(0.5, next.distance);
    // Move to the boundary but scatter perpendicularly, away from the sphere
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, 1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());

    // Move a bit internally, then scatter back toward the sphere
    next = geo.find_next_step(10.0);
    EXPECT_SOFT_EQ(6, next.distance);
    geo.set_dir({-1, 0, 0});
    EXPECT_EQ(VolumeId{1}, geo.volume_id());

    // Move to the sphere boundary then scatter still into the sphere
    next = geo.find_next_step(10.0);
    EXPECT_SOFT_EQ(1e-8, next.distance);
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, -1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_TRUE(geo.is_on_boundary());

    // Travel nearly tangent to the right edge of the sphere, then scatter to
    // still outside
    next = geo.find_next_step(1.0);
    EXPECT_SOFT_EQ(0.00031622777925735285, next.distance);
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({1, 0, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_TRUE(geo.is_on_boundary());
    next = geo.find_next_step(10.0);
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, tracking)
{
    {
        SCOPED_TRACE("Rightward");
        auto result = this->track({-10, -10, -10}, {1, 0, 0});

        static char const* const expected_volumes[] = {"Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World",
                                                       "Envelope",
                                                       "Shape1",
                                                       "Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {5, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {2.5, 0.5, 0.5, 3, 0.5, 0.5, 5, 0.5, 0.5, 3.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("From just inside outside edge");
        auto result = this->track({-24 + 0.001, 10., 10.}, {1, 0, 0});

        static char const* const expected_volumes[] = {"World",
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
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {7 - 0.001, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {3.4995, 0.5, 0.5, 5, 0.5, 0.5, 3, 0.5, 0.5, 5, 0.5, 0.5, 3.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Leaving world");
        auto result = this->track({-10, 10, 10}, {0, 1, 0});

        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 2, 6};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {2.5, 0.5, 1, 3};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Upward");
        auto result = this->track({-10, 10, 10}, {0, 0, 1});

        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 3, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {2.5, 0.5, 1.5, 2.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, safety)
{
    auto geo = this->make_geo_track_view();
    std::vector<real_type> safeties;

    for (auto i : range(11))
    {
        real_type r = 2.0 * i + 0.1;
        geo = {{r, r, r}, {1, 0, 0}};

        if (!geo.is_outside())
        {
            geo.find_next_step();
            safeties.push_back(geo.find_safety());
        }
    }

    static double const expected_safeties[] = {2.9,
                                               0.9,
                                               0.1,
                                               1.7549981495186,
                                               1.7091034656191,
                                               4.8267949192431,
                                               1.3626933041054,
                                               1.9,
                                               0.1,
                                               1.1,
                                               3.1};
    EXPECT_VEC_SOFT_EQ(expected_safeties, safeties);
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, TEST_IF_CELERITAS_CUDA(device))
{
    using StateStore = CollectionStateStore<VecgeomStateData, MemSpace::device>;

    // Set up test input
    VGGTestInput input;
    input.init = {{{10, 10, 10}, {1, 0, 0}},
                  {{10, 10, -10}, {1, 0, 0}},
                  {{10, -10, 10}, {1, 0, 0}},
                  {{10, -10, -10}, {1, 0, 0}},
                  {{-10, 10, 10}, {-1, 0, 0}},
                  {{-10, 10, -10}, {-1, 0, 0}},
                  {{-10, -10, 10}, {-1, 0, 0}},
                  {{-10, -10, -10}, {-1, 0, 0}}};
    StateStore device_states(this->geometry()->host_ref(), input.init.size());
    input.max_segments = 5;
    input.params = this->geometry()->device_ref();
    input.state = device_states.ref();

    // Run kernel
    auto output = vgg_test(input);

    static int const expected_ids[]
        = {1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3,
           1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3};

    static double const expected_distances[]
        = {5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3,
           5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3};

    // Check results
    EXPECT_VEC_EQ(expected_ids, output.ids);
    EXPECT_VEC_SOFT_EQ(expected_distances, output.distances);
}

//---------------------------------------------------------------------------//
// SOLIDS TEST
//---------------------------------------------------------------------------//

class SolidsTest : public VecgeomTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_vgdml("solids.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, DISABLED_dump)
{
    this->geometry();
    auto const& geomgr = vecgeom::GeoManager::Instance();
    auto const* world = geomgr.GetWorld();
    CELER_ASSERT(world);
    world->PrintContent();
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, accessors)
{
    if (vecgeom_version <= Version(1, 1, 17))
    {
        FAIL() << "VecGeom 1.1.17 crashes when trying to load unknown solids";
    }

    auto const& geom = *this->geometry();
    EXPECT_EQ(2, geom.max_depth());

    if (vecgeom_version < Version(1, 2, 2))
    {
        ADD_FAILURE()
            << "VecGeom " << vecgeom_version
            << " is missing features: upgrade to 1.2.2 to pass this test";
    }

    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-600.001, -300.001, -75.001}), bbox.lower());
    EXPECT_VEC_SOFT_EQ((Real3{600.001, 300.001, 75.001}), bbox.upper());

    // NOTE: because SolidsTest gets loaded after FourLevelsTest, the existing
    // volumes still have incremented the volume ID counter, so there is an
    // offset. This value will be zero if running the solids test as
    // standalone.
    int const offset = 4;
    ASSERT_EQ(23 + offset, geom.num_volumes());
    ASSERT_EQ(27, geom.num_volumes());
    EXPECT_EQ("box500", geom.id_to_label(VolumeId{0 + offset}).name);
    EXPECT_EQ("cone1", geom.id_to_label(VolumeId{1 + offset}).name);
    EXPECT_EQ("World", geom.id_to_label(VolumeId{20 + offset}).name);
    EXPECT_EQ("trd1_refl", geom.id_to_label(VolumeId{22 + offset}).name);
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, trace)
{
    {
        SCOPED_TRACE("Center -x");
        auto result = this->track({375, 0, 0}, {-1, 0, 0});

        static char const* const expected_volumes[] = {"ellipsoid1",
                                                       "World",
                                                       "polycone1",
                                                       "World",
                                                       "sphere1",
                                                       "World",
                                                       "box500",
                                                       "World",
                                                       "cone1",
                                                       "World",
                                                       "trd1",
                                                       "World",
                                                       "parabol1",
                                                       "World",
                                                       "trd1",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {20,
                                                       95,
                                                       20,
                                                       125,
                                                       40,
                                                       60,
                                                       50,
                                                       73,
                                                       54,
                                                       83,
                                                       30,
                                                       88.786678713601,
                                                       42.426642572799,
                                                       88.7866787136,
                                                       30,
                                                       85};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0,
                                                       45.496748548005,
                                                       0,
                                                       44.8347556812201,
                                                       13.934134186943,
                                                       30,
                                                       25,
                                                       36.240004604773,
                                                       25,
                                                       41.2043887972073,
                                                       14.92555785315,
                                                       42.910442345001,
                                                       18.741024106017,
                                                       42.910442345001,
                                                       14.92555785315,
                                                       42.289080583925};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Upper +x");
        auto result = this->track({-375, 125, 0}, {1, 0, 0});

        static char const* const expected_volumes[] = {"World",
                                                       "hype1",
                                                       "World",
                                                       "para1",
                                                       "World",
                                                       "tube100",
                                                       "World",
                                                       "boolean1",
                                                       "World",
                                                       "boolean1",
                                                       "World",
                                                       "polyhedr1",
                                                       "World",
                                                       "polyhedr1",
                                                       "World",
                                                       "ellcone1",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {20,
                                                       4,
                                                       71,
                                                       60,
                                                       75,
                                                       4,
                                                       116.000001,
                                                       12.499999,
                                                       20.000001,
                                                       17.499999,
                                                       191.98703789108,
                                                       1e-08,
                                                       40.048511400819,
                                                       25.977412807017,
                                                       86.987037891082,
                                                       10,
                                                       220};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0,
                                                       1.9937213884673,
                                                       0,
                                                       24.961508830135,
                                                       31.201886037669,
                                                       2,
                                                       42.0000005,
                                                       6.2499995,
                                                       9.9999995,
                                                       8.7499995,
                                                       75,
                                                       0,
                                                       11.928052271225,
                                                       43.188475615448,
                                                       4.9751859510499,
                                                       75};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Lower +x");
        auto result = this->track({-375, -125, 0}, {1, 0, 0});

        static char const* const expected_volumes[] = {"arb8b",
                                                       "World",
                                                       "arb8a",
                                                       "World",
                                                       "trap1",
                                                       "World",
                                                       "tetrah1",
                                                       "World",
                                                       "orb1",
                                                       "World",
                                                       "genPocone1",
                                                       "World",
                                                       "genPocone1",
                                                       "World",
                                                       "elltube1",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {40,
                                                       45,
                                                       80,
                                                       68.125,
                                                       33.75,
                                                       57.519332346491,
                                                       50.605667653509,
                                                       85,
                                                       80,
                                                       40,
                                                       45,
                                                       127.5,
                                                       3.7499999999998,
                                                       60,
                                                       40,
                                                       205};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {19.9007438042,
                                                       17.5,
                                                       21.951571334408,
                                                       29.0625,
                                                       15.746700605861,
                                                       26.836732015088,
                                                       2.7598369213007,
                                                       4.6355704644931,
                                                       40,
                                                       19.156525704423,
                                                       0,
                                                       0,
                                                       0,
                                                       28.734788556635,
                                                       20,
                                                       75};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//
// CONSTRUCT FROM GEANT4
//---------------------------------------------------------------------------//

#define FourLevelsGeantTest TEST_IF_CELERITAS_GEANT(FourLevelsGeantTest)
class FourLevelsGeantTest : public VecgeomTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_g4_gdml("four-levels.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsGeantTest, accessors)
{
    auto const& geom = *this->geometry();
    EXPECT_EQ(4, geom.max_depth());

    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-24.001, -24.001, -24.001}), bbox.lower());
    EXPECT_VEC_SOFT_EQ((Real3{24.001, 24.001, 24.001}), bbox.upper());

    ASSERT_EQ(4, geom.num_volumes());
    EXPECT_EQ("World", geom.id_to_label(VolumeId{0}).name);
    EXPECT_EQ("Envelope", geom.id_to_label(VolumeId{1}).name);
    EXPECT_EQ("Shape1", geom.id_to_label(VolumeId{2}).name);
    EXPECT_EQ("Shape2", geom.id_to_label(VolumeId{3}).name);
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsGeantTest, tracking)
{
    {
        SCOPED_TRACE("Rightward");
        auto result = this->track({-10, -10, -10}, {1, 0, 0});
        static char const* const expected_volumes[] = {"Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World",
                                                       "Envelope",
                                                       "Shape1",
                                                       "Shape2",
                                                       "Shape1",
                                                       "Envelope",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {5, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("From outside edge");
        auto result = this->track({-24, 10., 10.}, {1, 0, 0});
        static char const* const expected_volumes[] = {"[OUTSIDE]",
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
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {1e-13, 7.0 - 1e-13, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Leaving world");
        auto result = this->track({-10, 10, 10}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 2, 6};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Upward");
        auto result = this->track({-10, 10, 10}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 1, 3, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        // Formerly in linear propagator test, used to fail
        SCOPED_TRACE("From just outside world");
        auto result = this->track({-24, 10, 10}, {1, 0, 0});
        static char const* const expected_volumes[] = {"[OUTSIDE]",
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
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {1e-13, 7, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//

#define SolidsGeantTest TEST_IF_CELERITAS_GEANT(SolidsGeantTest)
class SolidsGeantTest : public VecgeomTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_g4_gdml("solids.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(SolidsGeantTest, DISABLED_dump)
{
    this->geometry();
    auto const* world = vecgeom::GeoManager::Instance().GetWorld();
    world->PrintContent();
}

//---------------------------------------------------------------------------//

TEST_F(SolidsGeantTest, accessors)
{
    if (vecgeom_version <= Version(1, 1, 17))
    {
        FAIL() << "VecGeom 1.1.17 crashes when trying to load unknown solids";
    }

    auto const& geom = *this->geometry();
    EXPECT_EQ(2, geom.max_depth());

    if (vecgeom_version < Version(1, 2, 2))
    {
        ADD_FAILURE()
            << "VecGeom " << vecgeom_version
            << " is missing features: upgrade to 1.2.2 to pass this test";
    }

    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-600.001, -300.001, -75.001}), bbox.lower());
    EXPECT_VEC_SOFT_EQ((Real3{600.001, 300.001, 75.001}), bbox.upper());

    ASSERT_EQ(28, geom.num_volumes());
    EXPECT_EQ("World", geom.id_to_label(VolumeId{0}).name);
    EXPECT_EQ("box500", geom.id_to_label(VolumeId{1}).name);
    EXPECT_EQ("cone1", geom.id_to_label(VolumeId{2}).name);
    EXPECT_EQ("", geom.id_to_label(VolumeId{9}).name);
    EXPECT_EQ("boolean1", geom.id_to_label(VolumeId{16}).name);
}

//---------------------------------------------------------------------------//

TEST_F(SolidsGeantTest, trace)
{
    {
        SCOPED_TRACE("Center -x");
        auto result = this->track({375, 0, 0}, {-1, 0, 0});
        static char const* const expected_volumes[] = {"ellipsoid1",
                                                       "World",
                                                       "polycone1",
                                                       "World",
                                                       "sphere1",
                                                       "World",
                                                       "box500",
                                                       "World",
                                                       "cone1",
                                                       "World",
                                                       "trd1",
                                                       "World",
                                                       "parabol1",
                                                       "World",
                                                       "trd1",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {20,
                                                       95,
                                                       20,
                                                       125,
                                                       40,
                                                       60,
                                                       50,
                                                       73,
                                                       54,
                                                       83,
                                                       30,
                                                       88.786678713601,
                                                       42.426642572799,
                                                       88.7866787136,
                                                       30,
                                                       85};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Upper +x");
        auto result = this->track({-375, 125, 0}, {1, 0, 0});
        static char const* const expected_volumes[] = {"World",
                                                       "hype1",
                                                       "World",
                                                       "para1",
                                                       "World",
                                                       "tube100",
                                                       "World",
                                                       "boolean1",
                                                       "World",
                                                       "boolean1",
                                                       "World",
                                                       "polyhedr1",
                                                       "World",
                                                       "polyhedr1",
                                                       "World",
                                                       "ellcone1",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {20,
                                                       4,
                                                       71,
                                                       60,
                                                       75,
                                                       4,
                                                       116.000001,
                                                       12.499999,
                                                       20.000001,
                                                       17.499999,
                                                       191.98703789108,
                                                       1e-08,
                                                       40.048511400819,
                                                       25.977412807017,
                                                       86.987037891082,
                                                       10,
                                                       220};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Lower +x");
        auto result = this->track({-375, -125, 0}, {1, 0, 0});
        static char const* const expected_volumes[] = {"arb8b",
                                                       "World",
                                                       "arb8a",
                                                       "World",
                                                       "trap1",
                                                       "World",
                                                       "tetrah1",
                                                       "World",
                                                       "orb1",
                                                       "World",
                                                       "genPocone1",
                                                       "World",
                                                       "genPocone1",
                                                       "World",
                                                       "elltube1",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {40,
                                                       45,
                                                       80,
                                                       68.125,
                                                       33.75,
                                                       57.519332346491,
                                                       50.605667653509,
                                                       85,
                                                       80,
                                                       40,
                                                       45,
                                                       127.5,
                                                       3.7499999999998,
                                                       60,
                                                       40,
                                                       205};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
