//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SolidsGeoTest.cc
//---------------------------------------------------------------------------//
#include "SolidsGeoTest.hh"

#include "corecel/Config.hh"

#include "corecel/math/ArrayOperators.hh"
#include "corecel/sys/Version.hh"

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
    CELERITAS_USE_VECGEOM ? celeritas_vecgeom_version : "0.0.0");
auto const geant4_version = celeritas::Version::from_string(
    CELERITAS_USE_GEANT4 ? celeritas_geant4_version : "0.0.0");

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with tracking test interface.
 */
SolidsGeoTest::SolidsGeoTest(GenericGeoTestInterface* geo_test)
    : test_{geo_test}
{
    CELER_EXPECT(test_);
}

//---------------------------------------------------------------------------//
//! Test geometry accessors
void SolidsGeoTest::test_accessors() const
{
    auto const& geo = *test_->geometry_interface();
    EXPECT_EQ(2, geo.max_depth());

    Real3 expected_lo{-600., -300., -75.};
    Real3 expected_hi{600., 300., 75.};
    if (test_->geometry_type() == "VecGeom")
    {
        // VecGeom expands its bboxes
        expected_lo -= .001;
        expected_hi += .001;
    }
    auto const& bbox = geo.bbox();
    EXPECT_VEC_SOFT_EQ(expected_lo, to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ(expected_hi, to_cm(bbox.upper()));

    static char const* const expected_vol_labels[] = {
        "box500",   "cone1",     "para1",      "sphere1",     "parabol1",
        "trap1",    "trd1",      "trd2",       "trd3_refl@1", "tube100",
        "boolean1", "polycone1", "genPocone1", "ellipsoid1",  "tetrah1",
        "orb1",     "polyhedr1", "hype1",      "elltube1",    "ellcone1",
        "arb8b",    "arb8a",     "xtru1",      "World",       "trd3_refl@0",
    };
    EXPECT_VEC_EQ(expected_vol_labels, test_->get_volume_labels());

    static char const* const expected_vol_inst_labels[] = {
        "box500_PV",   "cone1_PV",     "para1_PV",      "sphere1_PV",
        "parabol1_PV", "trap1_PV",     "trd1_PV",       "reflNormal",
        "reflected@0", "reflected@1",  "tube100_PV",    "boolean1_PV",
        "orb1_PV",     "polycone1_PV", "hype1_PV",      "polyhedr1_PV",
        "tetrah1_PV",  "arb8a_PV",     "arb8b_PV",      "ellipsoid1_PV",
        "elltube1_PV", "ellcone1_PV",  "genPocone1_PV", "xtru1_PV",
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
void SolidsGeoTest::test_trace() const
{
    // VecGeom adds bumps through boolean volumes
    real_type const bool_tol
        = (test_->geometry_type() == "VecGeom" ? 1e-7 : 1e-12);

    {
        SCOPED_TRACE("Upper +x");
        auto result = test_->track({-575, 125, 0.5}, {1, 0, 0});
        static char const* const expected_volumes[] = {
            "World",     "hype1",    "World",    "hype1",     "World",
            "para1",     "World",    "tube100",  "World",     "boolean1",
            "World",     "boolean1", "World",    "polyhedr1", "World",
            "polyhedr1", "World",    "ellcone1", "World",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
            "World_PV", "hype1_PV",     "World_PV", "hype1_PV",
            "World_PV", "para1_PV",     "World_PV", "tube100_PV",
            "World_PV", "boolean1_PV",  "World_PV", "boolean1_PV",
            "World_PV", "polyhedr1_PV", "World_PV", "polyhedr1_PV",
            "World_PV", "ellcone1_PV",  "World_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
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
        EXPECT_VEC_NEAR(expected_distances, result.distances, bool_tol);
        std::vector<real_type> expected_hw_safety = {
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
            // v1.2.10: unknown differences
            expected_hw_safety[1] = 1.99361986757606;
            expected_hw_safety[3] = 1.99361986757606;
        }
        EXPECT_VEC_NEAR(expected_hw_safety, result.halfway_safeties, bool_tol);
    }
    {
        SCOPED_TRACE("Center -x");
        auto result = test_->track({575, 0, 0.50}, {-1, 0, 0});
        static char const* const expected_volumes[] = {
            "World", "ellipsoid1", "World", "polycone1", "World", "polycone1",
            "World", "sphere1",    "World", "box500",    "World", "cone1",
            "World", "trd1",       "World", "parabol1",  "World", "trd2",
            "World", "xtru1",      "World",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
            "World_PV", "ellipsoid1_PV", "World_PV", "polycone1_PV",
            "World_PV", "polycone1_PV",  "World_PV", "sphere1_PV",
            "World_PV", "box500_PV",     "World_PV", "cone1_PV",
            "World_PV", "trd1_PV",       "World_PV", "parabol1_PV",
            "World_PV", "reflNormal",    "World_PV", "xtru1_PV",
            "World_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
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
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        std::vector<real_type> expected_hw_safety = {
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
            // v1.2.10: unknown differences
            expected_hw_safety[4] = 7.82052980478031;
            expected_hw_safety[14] = 42.8397753718277;
            expected_hw_safety[15] = 18.8833925371992;
            expected_hw_safety[16] = 42.8430141842906;
        }

        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Lower +x");
        auto result = test_->track({-575, -125, 0.5}, {1, 0, 0});
        static char const* const expected_volumes[] = {
            "World",   "trd3_refl",  "trd3_refl", "World",    "arb8b",
            "World",   "arb8a",      "World",     "trap1",    "World",
            "tetrah1", "World",      "orb1",      "World",    "genPocone1",
            "World",   "genPocone1", "World",     "elltube1", "World",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        std::vector<std::string> expected_volume_instances = {
            "World_PV",      "reflected", "reflected",     "World_PV",
            "arb8b_PV",      "World_PV",  "arb8a_PV",      "World_PV",
            "trap1_PV",      "World_PV",  "tetrah1_PV",    "World_PV",
            "orb1_PV",       "World_PV",  "genPocone1_PV", "World_PV",
            "genPocone1_PV", "World_PV",  "elltube1_PV",   "World_PV",
        };
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
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
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        std::vector<real_type> expected_hw_safety = {
            17.391607656793,
            14.968644196913,
            12.394878533861,
            34.872720758987,
            39.7517357488891,
            22.438088639235,
            33.0701970644251,
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
            expected_hw_safety[4] = 38.205672682313;
            expected_hw_safety[6] = 38.803595749271;
        }
        else if (test_->geometry_type() == "VecGeom")
        {
            expected_hw_safety = {
                17.391607656793,
                14.968644196913,
                12.394878533861,
                29.99665061979,
                27.765772866092,
                17.5,
                21.886464159888,
                29.111537609107,
                15.672519698479,
                26.80540527207,
                2.9387549751221,
                4.4610799311799,
                39.5,
                19.038294080807,
                0.5,
                29.515478338297,
                0,
                28.615060270982,
                20,
                74.5,
            };
        }

        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("Middle +y");
        auto result = test_->track({0, -250, 0.5}, {0, 1, 0});
        static char const* const expected_volumes[] = {
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
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static char const* const expected_volume_instances[] = {
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
        EXPECT_VEC_EQ(expected_volume_instances, result.volume_instances);
        static real_type const expected_distances[] = {
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
        EXPECT_VEC_NEAR(expected_distances, result.distances, bool_tol);
        static real_type const expected_hw_safety[] = {
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
        EXPECT_VEC_NEAR(expected_hw_safety, result.halfway_safeties, bool_tol);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
