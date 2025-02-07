//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeo.test.cc
//---------------------------------------------------------------------------//
#include <string_view>
#include <G4LogicalVolume.hh>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Version.hh"
#include "geocel/GeoParamsOutput.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/g4/GeantGeoData.hh"
#include "geocel/g4/GeantGeoParams.hh"
#include "geocel/g4/GeantGeoTrackView.hh"
#include "geocel/rasterize/SafetyImager.hh"

#include "GeantGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
auto const geant4_version = celeritas::Version::from_string(
    CELERITAS_USE_GEANT4 ? celeritas_geant4_version : "0.0.0");

}  // namespace

//---------------------------------------------------------------------------//

class GeantGeoTest : public GeantGeoTestBase
{
  public:
    using SpanStringView = Span<std::string_view const>;

    SPConstGeo build_geometry() final
    {
        ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                    LogLevel::warning};
        auto result = this->build_geometry_from_basename();
        EXPECT_VEC_EQ(this->expected_log_levels(), scoped_log_.levels())
            << scoped_log_;
        return result;
    }

    virtual SpanStringView expected_log_levels() const { return {}; }
};

//---------------------------------------------------------------------------//
class FourLevelsTest : public GeantGeoTest
{
    std::string geometry_basename() const override { return "four-levels"; }
};

TEST_F(FourLevelsTest, accessors)
{
    auto const& geom = *this->geometry();
    EXPECT_EQ(4, geom.max_depth());

    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-24., -24., -24.}), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{24., 24., 24.}), to_cm(bbox.upper()));

    ASSERT_EQ(4, geom.volumes().size());
    EXPECT_EQ("Shape2", geom.volumes().at(VolumeId{0}).name);
    EXPECT_EQ("Shape1", geom.volumes().at(VolumeId{1}).name);
    EXPECT_EQ("Envelope", geom.volumes().at(VolumeId{2}).name);
    EXPECT_EQ("World", geom.volumes().at(VolumeId{3}).name);
    EXPECT_EQ(Label("World", "0xdeadbeef"), geom.volumes().at(VolumeId{3}));

    auto const& vol_instances = geom.volume_instances();
    std::vector<std::string> instance_names;
    for (auto viid : range(VolumeInstanceId{vol_instances.size()}))
    {
        instance_names.push_back(vol_instances.at(viid).name);
    }
    static char const* const expected_instance_names[] = {
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
        "World",
    };
    EXPECT_VEC_EQ(expected_instance_names, instance_names);

    auto const* lv = geom.id_to_lv(VolumeId{2});
    ASSERT_TRUE(lv);
    EXPECT_EQ("Envelope", lv->GetName());

    auto const* pv = geom.id_to_pv(VolumeInstanceId{2});
    ASSERT_TRUE(pv);
    EXPECT_EQ("Envelope", lv->GetName());
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, consecutive_compute)
{
    auto geo = this->make_geo_track_view({-9, -10, -10}, {1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape2", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    auto next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));

    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));

    // Find safety from a freshly initialized state
    geo = {from_cm({-9, -10, -10}), {1, 0, 0}};
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, detailed_track)
{
    {
        SCOPED_TRACE("rightward along corner");
        auto geo = this->make_geo_track_view({-10, -10, -10}, {1, 0, 0});
        ASSERT_FALSE(geo.is_outside());
        EXPECT_EQ("Shape2", this->volume_name(geo));
        EXPECT_FALSE(geo.is_on_boundary());

        // Check for surfaces up to a distance of 4 units away
        auto next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
        geo.move_internal(from_cm(3.5));
        EXPECT_FALSE(geo.is_on_boundary());

        // Find one a bit further, then cross it
        next = geo.find_next_step(from_cm(4.0));
        EXPECT_SOFT_EQ(1.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        geo.move_to_boundary();
        EXPECT_EQ("Shape2", this->volume_name(geo));

        geo.cross_boundary();
        EXPECT_EQ("Shape1", this->volume_name(geo));
        EXPECT_TRUE(geo.is_on_boundary());

        // Find the next boundary and make sure that nearer distances aren't
        // accepted
        next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);
        EXPECT_TRUE(geo.is_on_boundary());
        next = geo.find_next_step(from_cm(0.5));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_FALSE(next.boundary);
    }
    {
        SCOPED_TRACE("inside out");
        auto geo = this->make_geo_track_view({-23.5, 6.5, 6.5}, {-1, 0, 0});
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ("World", this->volume_name(geo));

        auto next = geo.find_next_step(from_cm(2));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_FALSE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_TRUE(geo.is_outside());
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, reentrant_boundary)
{
    auto geo = this->make_geo_track_view({15.5, 10, 10}, {-1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces: we should hit the outside of the sphere Shape2
    auto next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
    // Move to the boundary but scatter perpendicularly, away from the sphere
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, 1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ(VolumeId{1}, geo.volume_id());

    // Move a bit internally, then scatter back toward the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(6, to_cm(next.distance));
    geo.set_dir({-1, 0, 0});
    EXPECT_EQ(VolumeId{1}, geo.volume_id());

    // Move to the sphere boundary then scatter still into the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(1e-13, to_cm(next.distance));
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
    next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(9.9794624025613538e-07, to_cm(next.distance));
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({1, 0, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ(VolumeId{1}, geo.volume_id());
    EXPECT_TRUE(geo.is_on_boundary());
    next = geo.find_next_step(from_cm(10.0));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, tracking)
{
    constexpr real_type safety_tol{1e-10};
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
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
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
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
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
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
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
        EXPECT_VEC_NEAR(
            expected_hw_safety, result.halfway_safeties, safety_tol);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, safety)
{
    auto geo = this->make_geo_track_view();
    std::vector<real_type> safeties;
    std::vector<real_type> lim_safeties;

    for (auto i : range(11))
    {
        real_type r = from_cm(2.0 * i + 0.1);
        geo = {{r, r, r}, {1, 0, 0}};
        if (!geo.is_outside())
        {
            geo.find_next_step();
            safeties.push_back(to_cm(geo.find_safety()));
            lim_safeties.push_back(to_cm(geo.find_safety(from_cm(1.5))));
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

    static double const expected_lim_safeties[] = {2.9,
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
    EXPECT_VEC_SOFT_EQ(expected_lim_safeties, lim_safeties);
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, levels)
{
    auto geo = this->make_geo_track_view({10.0, 10.0, 10.0}, {1, 0, 0});
    EXPECT_EQ("World@0xdeadbeef_PV/env1/Shape1/Shape2",
              this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World@0xdeadbeef_PV/env1/Shape1",
              this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World@0xdeadbeef_PV/env1", this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World@0xdeadbeef_PV", this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("[OUTSIDE]", this->all_volume_instance_names(geo));
}

//---------------------------------------------------------------------------//
class SolidsTest : public GeantGeoTest
{
    std::string geometry_basename() const override { return "solids"; }

    SpanStringView expected_log_levels() const final
    {
        if (geant4_version < Version{11})
            return {};

        static std::string_view const levels[] = {"error"};
        return make_span(levels);
    }
};

TEST_F(SolidsTest, accessors)
{
    auto const& geom = *this->geometry();
    EXPECT_EQ(2, geom.max_depth());

    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-600., -300., -75.}), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{600., 300., 75.}), to_cm(bbox.upper()));

    // NOTE: because SolidsTest gets loaded after FourLevelsTest, the existing
    // volumes still have incremented the volume ID counter, so there is an
    // offset. This value will be zero if running the solids test as
    // standalone.
    int const offset = 4;
    ASSERT_EQ(26 + offset, geom.volumes().size());
    EXPECT_EQ("box500", geom.volumes().at(VolumeId{0 + offset}).name);
    EXPECT_EQ("cone1", geom.volumes().at(VolumeId{1 + offset}).name);
    EXPECT_EQ("World", geom.volumes().at(VolumeId{24 + offset}).name);
    EXPECT_EQ("trd3_refl", geom.volumes().at(VolumeId{25 + offset}).name);
}

//---------------------------------------------------------------------------//
TEST_F(SolidsTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-600.0,-300.0,-75.0],[600.0,300.0,75.0]],"max_depth":2,"supports_safety":true,"volumes":{"label":["","","","","box500","cone1","para1","sphere1","parabol1","trap1","trd1","trd2","","trd3_refl","tube100","boolean1","polycone1","genPocone1","ellipsoid1","tetrah1","orb1","polyhedr1","hype1","elltube1","ellcone1","arb8b","arb8a","xtru1","World","trd3_refl"]}})json",
            to_string(out));
    }
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, trace)
{
    if (CELERITAS_UNITS != CELERITAS_UNITS_SI)
    {
        /*
         * Note: with SI, this trace skips over the sphere! (During
         * reinitialization halfway between the polycone and expected sphere
         * boundary.) Perhaps it's some really really weird floating point edge
         * case?
         *
         * Value of: next.distance / 2
         * Actual: 0.57499999999999996
         * Expected: new_next.distance
         * Which is: 1.5750000000000004
         * reinitialized distance mismatch at index 5: {1.8250000000000004,0,0}
         * Actual values: {"ellipsoid1", "World", "polycone1", "World",
         * "polycone1", "World", "box500", ...}
         */
        SCOPED_TRACE("Center -x");
        auto result = this->track({375, 0, 0}, {-1, 0, 0});

        static char const* const expected_volumes[]
            = {"ellipsoid1", "World",   "polycone1", "World",  "polycone1",
               "World",      "sphere1", "World",     "box500", "World",
               "cone1",      "World",   "trd1",      "World",  "parabol1",
               "World",      "trd2",    "World",     "xtru1",  "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {20,
                                                       95,
                                                       2,
                                                       16,
                                                       2,
                                                       115,
                                                       40,
                                                       60,
                                                       50,
                                                       73,
                                                       54,
                                                       83,
                                                       30,
                                                       88.786678713601,
                                                       42.426642572798,
                                                       88.786678713601,
                                                       30,
                                                       1.4761904761905,
                                                       15.880952380952,
                                                       67.642857142857};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0,
                                                       45.496748548005,
                                                       0,
                                                       8,
                                                       0.98058067569092,
                                                       41.247975226723,
                                                       13.934134186943,
                                                       30,
                                                       25,
                                                       36.240004604773,
                                                       25,
                                                       41.204388797207,
                                                       14.92555785315,
                                                       35.6066606432,
                                                       14.09753916278,
                                                       35.6066606432,
                                                       14.92555785315,
                                                       0.73443221182165,
                                                       6.5489918373272,
                                                       33.481506089183};
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
                                                       116,
                                                       12.5,
                                                       20,
                                                       17.5,
                                                       191.98703789108,
                                                       25.977412807017,
                                                       14.071098603801,
                                                       25.977412807017,
                                                       86.987037891082,
                                                       9.9999999999996,
                                                       220};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {10,
                                                       1.9994563574043,
                                                       29.537785448993,
                                                       24.961508830135,
                                                       31.201886037669,
                                                       2,
                                                       42,
                                                       6.25,
                                                       10,
                                                       8.75,
                                                       75,
                                                       0,
                                                       6.4970769728954,
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
                                                       50.6056676535089,
                                                       85,
                                                       80,
                                                       40,
                                                       15,
                                                       60,
                                                       15,
                                                       60,
                                                       40,
                                                       205};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        std::vector<real_type> expected_hw_safety = {
            19.9007438042,
            22.388336779725,
            33.1715371545092,
            32.644989013003,
            15.746700605861,
            26.836732015088,
            2.7598369213007,
            4.6355704644931,
            40,
            19.9007438042,
            7.1836971391586,
            29.417420270728,
            0,
            29.8511157063,
            20,
            75,
        };
        if (geant4_version < Version{11, 3})
        {
            // Older versions of Geant4 have a bug in Arb8 that overestimates
            // safety distance to twisted surfaces
            expected_hw_safety[2] = 38.858788181402;
        }

        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Low +y");
        auto result = this->track({-500, -250, 0}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"World", "trd3_refl", "World", "trd2", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {96.555879457157,
                                                       52.35421982848,
                                                       77.179801428726,
                                                       52.35421982848,
                                                       271.55587945716};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {37.766529475342,
                                                       15.038346086645,
                                                       26.6409955055738,
                                                       15.038346086645,
                                                       75};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, reflected_vol)
{
    auto geo = this->make_geo_track_view({-500, -125, 0}, {0, 1, 0});
    EXPECT_EQ(VolumeId{29}, geo.volume_id());
    auto const& label = this->geometry()->volumes().at(geo.volume_id());
    EXPECT_EQ("trd3_refl", label.name);
    EXPECT_FALSE(ends_with(label.ext, "_refl"));
}

//---------------------------------------------------------------------------//
class TilecalPlugTest : public GeantGeoTest
{
    std::string geometry_basename() const final { return "tilecal-plug"; }
};

TEST_F(TilecalPlugTest, trace)
{
    {
        SCOPED_TRACE("+z");
        auto result = this->track({5.75, 0.01, -40}, {0, 0, 1});
        static char const* const expected_volumes[] = {"Tile_ITCModule",
                                                       "Tile_Plug1Module",
                                                       "Tile_Absorber",
                                                       "Tile_Plug1Module"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {22.9425, 0.115, 42, 37};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("+z");
        auto result = this->track({6.25, 0.01, -40}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"Tile_ITCModule", "Tile_Absorber", "Tile_Plug1Module"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {23.0575, 42, 37};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//
class TransformedBoxTest : public GeantGeoTest
{
    std::string geometry_basename() const override
    {
        return "transformed-box";
    }
};

TEST_F(TransformedBoxTest, trace)
{
    {
        auto result = this->track({0, 0, -25}, {0, 0, 1});
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
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        auto result = this->track({0.25, 0, -25}, {0., 0, 1});
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
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        auto result = this->track({0, 0.25, -25}, {0, 0., 1});
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
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        auto result = this->track({0.01, -20, 0.20}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"world", "enclosing", "tiny", "enclosing", "world"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {18.5, 1.1250390198213, 0.75090449735279, 1.1240564828259, 48.5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {9.25, 0.56184193052552, 0.05, 0.56135125378224, 24.25};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//
class CmseTest : public GeantGeoTest
{
  public:
    std::string geometry_basename() const override { return "cmse"; }
};

TEST_F(CmseTest, trace)
{
    // clang-format off
    {
        SCOPED_TRACE("Center +z");
        auto result = this->track({0, 0, -4000}, {0, 0, 1});
        static char const* const expected_volumes[] = {"CMStoZDC", "BEAM3",
            "BEAM2", "BEAM1", "BEAM", "BEAM", "BEAM1", "BEAM2", "BEAM3",
            "CMStoZDC", "CMSE", "ZDC", "CMSE", "ZDCtoFP420", "CMSE"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1300, 1096.95, 549.15,
            403.9, 650, 650, 403.9, 549.15, 1096.95, 11200, 9.9999999999992,
            180, 910, 24000, 6000};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {100, 2.1499999999997,
            10.3027302206744, 13.023518051922, 6.95, 6.95, 13.023518051922,
            10.3027302206745, 2.15, 100, 5, 8, 100, 100, 100};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Offset +z");
        auto result = this->track({30, 30, -4000}, {0, 0, 1});
        static char const* const expected_volumes[] = {"CMStoZDC", "OQUA",
            "VCAL", "OQUA", "CMSE", "TotemT1", "CMSE", "MUON", "CALO",
            "Tracker", "CALO", "MUON", "CMSE", "TotemT1", "CMSE", "OQUA",
            "VCAL", "OQUA", "CMStoZDC", "CMSE", "ZDCtoFP420", "CMSE"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1300, 1419.95, 165.1,
            28.95, 36, 300.1, 94.858988388759, 100.94101161124, 260.9, 586.4,
            260.9, 100.94101161124, 94.858988388759, 300.1, 36, 28.95, 165.1,
            1419.95, 11200, 1100, 24000, 6000};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {57.573593128807,
            40.276406871193, 29.931406871193, 14.475, 18, 28.702447147997,
            29.363145173005, 32.665765921596, 34.260814069425, 39.926406871193,
            34.260814069425, 32.665765921596, 29.363145173005, 28.702447147997,
            18, 14.475, 29.931406871193, 40.276406871193, 57.573593128807,
            57.573593128807, 57.573593128807, 57.573593128807};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Across muon");
        auto result = this->track({-1000, 0, -48.5}, {1, 0, 0});
        static char const* const expected_volumes[] = {"OCMS", "MUON", "CALO",
            "Tracker", "CMSE", "BEAM", "CMSE", "Tracker", "CALO", "MUON",
            "OCMS"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {170, 535, 171.7, 120.8,
            0.15673306650246, 4.6865338669951, 0.15673306650246, 120.8, 171.7,
            535, 920};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {85, 267.5, 85.85,
            60.4, 0.078366388350241, 2.343262600759, 0.078366388350241,
            60.4, 85.85, 267.5, 460};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Differs between G4/VG");
        auto result = this->track({0, 0, 1328.0}, {1, 0, 0});
        static char const* const expected_volumes[] = {"BEAM2", "OQUA", "CMSE",
            "OCMS"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {12.495, 287.505, 530,
            920};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {6.2475, 47.95, 242, 460};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    // clang-format on
}

TEST_F(CmseTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({0, 0, 0});
    inp.upper_right = from_cm({350, 0, 1700});
    inp.rightward = {0.0, 0.0, 1.0};
    inp.vertical_pixels = 8;

    write_image(ImageParams{inp}, "g4-cmse-xz-mid.jsonl");
}

//---------------------------------------------------------------------------//
class ZnenvTest : public GeantGeoTest
{
  public:
    std::string geometry_basename() const override { return "znenv"; }
};

TEST_F(ZnenvTest, trace)
{
    static char const* const expected_mid_volumes[]
        = {"World", "ZNENV", "ZNST", "ZNST",  "ZNST", "ZNST", "ZNST",
           "ZNST",  "ZNST",  "ZNST", "ZNST",  "ZNST", "ZNST", "ZNST",
           "ZNST",  "ZNST",  "ZNST", "ZNST",  "ZNST", "ZNST", "ZNST",
           "ZNST",  "ZNST",  "ZNST", "ZNENV", "World"};
    static real_type const expected_mid_distances[]
        = {6.38, 0.1,  0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
           0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
           0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.1,  46.38};
    {
        auto result = this->track({-10, 0.0001, 0}, {1, 0, 0});
        EXPECT_VEC_EQ(expected_mid_volumes, result.volumes);
        EXPECT_VEC_SOFT_EQ(expected_mid_distances, result.distances);
    }
    {
        auto result = this->track({0.0001, -10, 0}, {0, 1, 0});
        EXPECT_VEC_EQ(expected_mid_volumes, result.volumes);
        EXPECT_VEC_SOFT_EQ(expected_mid_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//
class MultiLevelTest : public GeantGeoTest
{
    std::string geometry_basename() const override { return "multi-level"; }
};

TEST_F(MultiLevelTest, accessors)
{
    auto const& geo = *this->geometry();
    EXPECT_EQ(3, geo.max_depth());

    auto vol_names = [&geo] {
        size_type const offset = 72;
        auto const& vols = geo.volumes();
        std::vector<std::string> result;
        for (auto vid : range(offset, vols.size()))
        {
            result.push_back(vols.at(VolumeId{vid}).name);
        }
        return result;
    }();
    static char const* const expected_vol_names[] = {"sph", "box", "world"};
    EXPECT_VEC_EQ(expected_vol_names, vol_names);

    auto vol_inst_names = [&geo] {
        size_type const offset = 92;
        auto const& vols = geo.volume_instances();
        std::vector<std::string> result;
        for (auto viid : range(offset, vols.size()))
        {
            result.push_back(vols.at(VolumeInstanceId{viid}).name);
        }
        return result;
    }();
    static char const* const expected_vol_inst_names[] = {
        "boxsph1",
        "boxsph2",
        "topsph1",
        "topbox1",
        "topbox2",
        "topbox3",
        "topsph2",
        "world_PV",
    };
    EXPECT_VEC_EQ(expected_vol_inst_names, vol_inst_names);
}

TEST_F(MultiLevelTest, trace)
{
    {
        auto result = this->track({-19.9, 7.5, 0}, {1, 0, 0});

        static char const* const expected_volumes[] = {
            "world",
            "box",
            "sph",
            "box",
            "world",
            "box",
            "sph",
            "box",
            "world",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
            "world_PV",
            "topbox2",
            "boxsph2",
            "topbox2",
            "world_PV",
            "topbox1",
            "boxsph2",
            "topbox1",
            "world_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
            2.4,
            3,
            4,
            8,
            5,
            3,
            4,
            8,
            6.5,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {
            1.2,
            1.5,
            2,
            3.0990195135928,
            2.5,
            1.5,
            2,
            3.0990195135928,
            3.25,
        };
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

TEST_F(MultiLevelTest, DISABLED_level_strings)
{
    using R2 = Array<real_type, 2>;

    auto const& vol_inst = this->geometry()->volume_instances();
    std::vector<VolumeInstanceId> ids;
    std::vector<std::string> names;
    std::ostringstream os;

    for (R2 xy : {R2{-5, 0},
                  R2{0, 0},
                  R2{-5, 0},
                  R2{7.5, 7.5},
                  R2{7.5, 12.5},
                  R2{12.5, 12.5},
                  R2{-12.5, 7.5},
                  R2{-7.5, 12.5},
                  R2{-7.5, -7.5},
                  R2{-12.5, -12.5}})
    {
        auto geo = this->make_geo_track_view({xy[0], xy[1], 0.0}, {1, 0, 0});

        auto level = geo.level();
        CELER_ASSERT(level && level >= LevelId{0});
        ids.resize(level.get() + 1);
        geo.volume_instance_id(make_span(ids));
        names.resize(ids.size());
        for (auto i : range(ids.size()))
        {
            names[i] = vol_inst.at(ids[i]).name;
        }
        PRINT_EXPECTED(names);
    }
}

//---------------------------------------------------------------------------//
class PincellTest : public GeantGeoTest
{
    std::string geometry_basename() const override { return "pincell"; }
};

TEST_F(PincellTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-12, -12, 0});
    inp.upper_right = from_cm({12, 12, 0});
    inp.rightward = {1.0, 0.0, 0.0};
    inp.vertical_pixels = 8;

    write_image(ImageParams{inp}, "g4-pincell-xy-mid.jsonl");

    inp.lower_left[2] = inp.upper_right[2] = from_cm(-5.5);
    write_image(ImageParams{inp}, "g4-pincell-xy-lo.jsonl");

    inp.lower_left = from_cm({-12, 0, -12});
    inp.upper_right = from_cm({12, 0, 12});
    write_image(ImageParams{inp}, "g4-pincell-xz-mid.jsonl");
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
