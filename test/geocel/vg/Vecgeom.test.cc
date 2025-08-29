//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/Vecgeom.test.cc
//---------------------------------------------------------------------------//
#include "Vecgeom.test.hh"

#include <string_view>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/StringSimplifier.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/Version.hh"
#include "geocel/GeantImportVolumeResult.hh"
#include "geocel/GenericGeoParameterizedTest.hh"
#include "geocel/GeoParamsOutput.hh"
#include "geocel/GeoTests.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/rasterize/SafetyImager.hh"
#include "geocel/vg/VecgeomData.hh"
#include "geocel/vg/VecgeomParams.hh"
#include "geocel/vg/VecgeomTrackView.hh"

#include "VecgeomTestBase.hh"
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
    = celeritas::Version::from_string(cmake::vecgeom_version);

}  // namespace

//---------------------------------------------------------------------------//
// TEST HARNESSES
//---------------------------------------------------------------------------//
class VecgeomTestBaseImpl : public VecgeomTestBase
{
  public:
    using SpanStringView = Span<std::string_view const>;

    virtual SpanStringView expected_log_levels() const { return {}; }

    //! Get the safety tolerance: lower for surface geo
    real_type safety_tol() const override
    {
        if (CELERITAS_VECGEOM_SURFACE)
            return 5e-5;
        return VecgeomTestBase::safety_tol();
    }
};

//! Load a geometry using VecGeom's GDML reader
class VecgeomVgdmlTestBase : public VecgeomTestBaseImpl
{
  public:
    SPConstGeo build_geometry() const final
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::warning};
        std::string filename{this->gdml_basename()};
        filename += ".gdml";
        auto result = VecgeomParams::from_gdml_vg(
            this->test_data_path("geocel", filename));
        EXPECT_VEC_EQ(this->expected_log_levels(), scoped_log_.levels())
            << scoped_log_;
        return result;
    }
};

//! Load a geometry using G4VG
class VecgeomGeantTestBase : public VecgeomTestBaseImpl
{
  public:
    using GeantVolResult = GeantImportVolumeResult;

    //! Construct via persistent geant_geo; see LazyGeantGeoManager
    SPConstGeo build_geometry() const final
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::warning};
        auto result = VecgeomTestBaseImpl::build_geometry();
        EXPECT_VEC_EQ(this->expected_log_levels(), scoped_log_.levels())
            << scoped_log_;
        return result;
    }

    //! Test conversion for Geant4 geometry
    GeantVolResult get_import_geant_volumes()
    {
        return GeantVolResult::from_import(*this->geometry());
    }

    SpanStringView expected_log_levels() const override { return {}; }
};

//---------------------------------------------------------------------------//
// VGDML TESTS
//---------------------------------------------------------------------------//

using FourLevelsVgdmlTest
    = GenericGeoParameterizedTest<VecgeomVgdmlTestBase, FourLevelsGeoTest>;

TEST_F(FourLevelsVgdmlTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(FourLevelsVgdmlTest, consecutive_compute)
{
    auto geo = this->make_geo_track_view({-9, -10, -10}, {1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape2", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    auto next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_NEAR(4.0, to_cm(geo.find_safety()), safety_tol());

    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_NEAR(4.0, to_cm(geo.find_safety()), safety_tol());

    // Find safety from a freshly initialized state
    geo = {from_cm({-9, -10, -10}), {1, 0, 0}};
    EXPECT_SOFT_NEAR(4.0, to_cm(geo.find_safety()), safety_tol());
}

TEST_F(FourLevelsVgdmlTest, detailed_track)
{
    // Templated test
    FourLevelsGeoTest::test_detailed_tracking(this);
}

TEST_F(FourLevelsVgdmlTest, trace)
{
    this->impl().test_trace();
}

TEST_F(FourLevelsVgdmlTest, safety)
{
    auto geo = this->make_geo_track_view();
    std::vector<real_type> safeties;
    std::vector<real_type> lim_safeties;

    for (auto i : range(11))
    {
        real_type r = 2.0 * i + 0.1;
        geo = {from_cm({r, r, r}), {1, 0, 0}};

        if (!geo.is_outside())
        {
            geo.find_next_step();
            safeties.push_back(to_cm(geo.find_safety()));
            lim_safeties.push_back(to_cm(geo.find_safety(from_cm(1.5))));
        }
    }

    static double const expected_safeties[] = {
        2.9,
        0.9,
        0.1,
        1.7549981495186,
        1.7091034656191,
        4.8267949192431,
        1.3626933041054,
        1.9,
        0.1,
        1.1,
        3.1,
    };
    EXPECT_VEC_NEAR(expected_safeties, safeties, safety_tol());

    static double const expected_lim_safeties[]
        = {1.5, 0.9, 0.1, 1.5, 1.5, 1.5, 1.3626933041054, 1.5, 0.1, 1.1, 1.5};
    EXPECT_VEC_NEAR(expected_lim_safeties, lim_safeties, safety_tol());
}

TEST_F(FourLevelsVgdmlTest, TEST_IF_CELERITAS_CUDA(device))
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

using MultiLevelVgdmlTest
    = GenericGeoParameterizedTest<VecgeomVgdmlTestBase, MultiLevelGeoTest>;

TEST_F(MultiLevelVgdmlTest, volume_stack)
{
    this->impl().test_volume_stack();
}

TEST_F(MultiLevelVgdmlTest, trace)
{
    TestImpl(this).test_trace();
}

//---------------------------------------------------------------------------//

using SimpleCmsVgdmlTest
    = GenericGeoParameterizedTest<VecgeomVgdmlTestBase, SimpleCmsGeoTest>;

TEST_F(SimpleCmsVgdmlTest, accessors)
{
    auto const& geom = *this->geometry();
    EXPECT_EQ(2, geom.max_depth());
    EXPECT_EQ(7, geom.impl_volumes().size());
}

TEST_F(SimpleCmsVgdmlTest, trace)
{
    this->impl().test_trace();
}

TEST_F(SimpleCmsVgdmlTest, detailed_track)
{
    // Templated test
    SimpleCmsGeoTest::test_detailed_tracking(this);
}

TEST_F(SimpleCmsVgdmlTest, TEST_IF_CELERITAS_CUDA(device))
{
    using StateStore = CollectionStateStore<VecgeomStateData, MemSpace::device>;

    // Set up test input
    VGGTestInput input;
    input.init = {{{10, 0, 0}, {1, 0, 0}},
                  {{29.99, 0, 0}, {1, 0, 0}},
                  {{150, 0, 0}, {0, 1, 0}},
                  {{174, 0, 0}, {0, 1, 0}},
                  {{0, -250, 100}, {-1, 0, 0}},
                  {{250, -250, 100}, {-1, 0, 0}},
                  {{250, 0, 100}, {0, -1, 0}},
                  {{-250, 0, 100}, {0, -1, 0}}};
    StateStore device_states(this->geometry()->host_ref(), input.init.size());
    input.max_segments = 5;
    input.params = this->geometry()->device_ref();
    input.state = device_states.ref();

    // Run kernel
    auto output = vgg_test(input);

    static int const expected_ids[] = {
        1, 2, 3, 4,  5,  1, 2, 3, 4, 5,  3, 4, 5, 6,  -2, 3, 4, 5, 6,  -2,
        4, 5, 6, -2, -3, 3, 4, 5, 6, -2, 4, 5, 6, -2, -3, 4, 5, 6, -2, -3,
    };
    static double const expected_distances[] = {
        20,
        95,
        50,
        100,
        100,
        0.010,
        95,
        50,
        100,
        100,
        90.1387818866,
        140.34982954572,
        113.20456568937,
        340.04653943718,
        316.26028344113,
        18.681541692269,
        194.27150477573,
        119.23515320201,
        345.84129821338,
        321.97050211661,
        114.5643923739,
        164.94410481358,
        374.32634434363,
        346.1651584689,
        -3,
        135.4356076261,
        229.12878474779,
        164.94410481358,
        374.32634434363,
        346.1651584689,
        114.5643923739,
        164.94410481358,
        374.32634434363,
        346.1651584689,
        -3,
        114.5643923739,
        164.94410481358,
        374.32634434363,
        346.1651584689,
        -3,
    };

    // Check results
    EXPECT_VEC_EQ(expected_ids, output.ids);
    EXPECT_VEC_SOFT_EQ(expected_distances, output.distances);
}

//---------------------------------------------------------------------------//

class SolidsVgdmlTest
    : public GenericGeoParameterizedTest<VecgeomVgdmlTestBase, SolidsGeoTest>
{
    SpanStringView expected_log_levels() const final
    {
        if (vecgeom_version >= Version{2})
        {
            static std::string_view const levels[] = {"warning", "warning"};
            return make_span(levels);
        }
        else
        {
            return {};
        }
    }
};

TEST_F(SolidsVgdmlTest, DISABLED_dump)
{
    this->geometry();
    auto const& geomgr = vecgeom::GeoManager::Instance();
    auto const* world = geomgr.GetWorld();
    CELER_ASSERT(world);
    world->PrintContent();
}

TEST_F(SolidsVgdmlTest, accessors)
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
    EXPECT_VEC_SOFT_EQ((Real3{-600.001, -300.001, -75.001}),
                       to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{600.001, 300.001, 75.001}), to_cm(bbox.upper()));
}

TEST_F(SolidsVgdmlTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-600.001,-300.001,-75.001],[600.001,300.001,75.001]],"supports_safety":true,"volumes":{"label":["","","","","box500","cone1","para1","sphere1","parabol1","trap1","trd1","trd2","","trd3_refl@1","tube100","boolean1","polycone1","genPocone1","ellipsoid1","tetrah1","orb1","polyhedr1","hype1","elltube1","ellcone1","arb8b","arb8a","xtru1","World","","trd3_refl@0"]}})json",
            to_string(out));
    }
}

TEST_F(SolidsVgdmlTest, reflected_vol)
{
    auto geo = this->make_geo_track_view({-500, -125, 0}, {0, 1, 0});
    auto const& label
        = this->geometry()->impl_volumes().at(geo.impl_volume_id());
    // Note: through GDML there is only one trd3_refl
    EXPECT_EQ("trd3_refl", label.name);
    EXPECT_FALSE(ends_with(label.ext, "_refl"));
}

//---------------------------------------------------------------------------//

class TwoBoxesVgdmlTest
    : public GenericGeoParameterizedTest<VecgeomVgdmlTestBase, TwoBoxesGeoTest>
{
};

TEST_F(TwoBoxesVgdmlTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TwoBoxesVgdmlTest, track)
{
    // Templated test
    TwoBoxesGeoTest::test_detailed_tracking(this);
}

//---------------------------------------------------------------------------//
// G4VG TESTS
//---------------------------------------------------------------------------//

using CmsEeBackDeeTest
    = GenericGeoParameterizedTest<VecgeomGeantTestBase, CmsEeBackDeeGeoTest>;

TEST_F(CmsEeBackDeeTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(CmsEeBackDeeTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

using CmseTest = GenericGeoParameterizedTest<VecgeomGeantTestBase, CmseGeoTest>;

TEST_F(CmseTest, trace)
{
    this->impl().test_trace();
}

TEST_F(CmseTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-550, 0, -4000});
    inp.upper_right = from_cm({550, 0, 2000});
    inp.rightward = {0.0, 0.0, 1.0};
    inp.vertical_pixels = 8;

    std::string prefix = "vg";
    if (VecgeomParams::use_surface_tracking())
    {
        prefix += "surf";
    }

    write_image(ImageParams{inp}, prefix + "-cmse.jsonl");
}

//---------------------------------------------------------------------------//

using FourLevelsTest
    = GenericGeoParameterizedTest<VecgeomGeantTestBase, FourLevelsGeoTest>;

TEST_F(FourLevelsTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(FourLevelsTest, trace)
{
    this->impl().test_trace();
}

TEST_F(FourLevelsTest, levels)
{
    auto geo = this->make_geo_track_view({10.0, 10.0, 10.0}, {1, 0, 0});
    EXPECT_EQ("World_PV/env1/Shape1/Shape2", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV/env1/Shape1", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV/env1", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV", this->unique_volume_name(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("[OUTSIDE]", this->unique_volume_name(geo));
}

//---------------------------------------------------------------------------//

using MultiLevelTest
    = GenericGeoParameterizedTest<VecgeomGeantTestBase, MultiLevelGeoTest>;

TEST_F(MultiLevelTest, volume_stack)
{
    this->impl().test_volume_stack();
}

TEST_F(MultiLevelTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

using PolyhedraTest
    = GenericGeoParameterizedTest<VecgeomGeantTestBase, PolyhedraGeoTest>;

TEST_F(PolyhedraTest, trace)
{
    TestImpl(this).test_trace();
}

//---------------------------------------------------------------------------//
class ReplicaTest
    : public GenericGeoParameterizedTest<VecgeomGeantTestBase, ReplicaGeoTest>
{
    real_type safety_tol() const final
    {
        if (CELERITAS_VECGEOM_SURFACE)
            return 1e-5;
        return 1e-10;
    }
};

TEST_F(ReplicaTest, trace)
{
    this->impl().test_trace();
}

TEST_F(ReplicaTest, volume_stack)
{
    this->impl().test_volume_stack();
}

//---------------------------------------------------------------------------//

class SolidsTest
    : public GenericGeoParameterizedTest<VecgeomGeantTestBase, SolidsGeoTest>
{
  public:
    // trd_refl is in the GDML *and* generated by ReflFactory
    SpanStringView expected_log_levels() const final
    {
        static std::string_view const levels[] = {"error"};
        return make_span(levels);
    }
};

TEST_F(SolidsTest, DISABLED_dump)
{
    this->geometry();
    auto const* world = vecgeom::GeoManager::Instance().GetWorld();
    world->PrintContent();
}

TEST_F(SolidsTest, accessors)
{
    if (vecgeom_version <= Version(1, 1, 17))
    {
        FAIL() << "VecGeom 1.1.17 crashes when trying to load unknown solids";
    }

    if (vecgeom_version < Version(1, 2, 2))
    {
        ADD_FAILURE()
            << "VecGeom " << vecgeom_version
            << " is missing features: upgrade to 1.2.2 to pass this test";
    }

    TestImpl(this).test_accessors();
}

TEST_F(SolidsTest, trace)
{
    TestImpl(this).test_trace();
}

TEST_F(SolidsTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        auto out_str = StringSimplifier{1}(to_string(out));

        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-6e2,-3e2,-8e1],[6e2,3e2,8e1]],"supports_safety":true,"volumes":{"label":["box500","cone1","para1","sphere1","parabol1","trap1","trd1","trd2","trd3_refl@1","tube100","","","","","boolean1","polycone1","genPocone1","ellipsoid1","tetrah1","orb1","polyhedr1","hype1","elltube1","ellcone1","arb8b","arb8a","xtru1","World","","trd3_refl@0"]}})json",
            out_str);
    }
}

TEST_F(SolidsTest, geant_volumes)
{
    {
        auto result = this->get_import_geant_volumes();
        static int const expected_volumes[] = {
            0,  1,  2,  3,  4,  5,  6,  7,  -1, 8,  9,  14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29,
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        EXPECT_EQ(0, result.missing_labels.size())
            << repr(result.missing_labels);
    }
}

TEST_F(SolidsTest, reflected_vol)
{
    auto geo = this->make_geo_track_view({-500, -125, 0}, {0, 1, 0});
    auto const& label
        = this->geometry()->impl_volumes().at(geo.impl_volume_id());
    EXPECT_EQ("trd3_refl@0", to_string(label));
}

TEST_F(SolidsTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-550, -250, 5});
    inp.upper_right = from_cm({550, 250, 5});
    inp.rightward = {1.0, 0.0, 0.0};
    inp.vertical_pixels = 8;

    std::string prefix = "vg";
    if (VecgeomParams::use_surface_tracking())
    {
        prefix += "surf";
    }

    write_image(ImageParams{inp}, prefix + "-solids-xy-hi.jsonl");

    inp.lower_left[2] = inp.upper_right[2] = from_cm(-5);
    write_image(ImageParams{inp}, prefix + "-solids-xy-lo.jsonl");
}

//---------------------------------------------------------------------------//
using TransformedBoxTest
    = GenericGeoParameterizedTest<VecgeomGeantTestBase, TransformedBoxGeoTest>;

TEST_F(TransformedBoxTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TransformedBoxTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using ZnenvTest
    = GenericGeoParameterizedTest<VecgeomGeantTestBase, ZnenvGeoTest>;

TEST_F(ZnenvTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
// UTILITIES
//---------------------------------------------------------------------------//

class ArbitraryVecgeomTest : public VecgeomTestBase
{
  public:
    void SetUp() override
    {
        filename_ = celeritas::getenv("GDML");
        CELER_VALIDATE(
            !filename_.empty(),
            << R"(Set the "GDML" environment variable and run this test with '--gtest_filter=*)"
            << ::testing::UnitTest::GetInstance()
                       ->current_test_info()
                       ->test_suite_name()
            << "*' --gtest_also_run_disabled_tests)");
    }

    // Basename used as key for cached geo
    std::string_view gdml_basename() const final
    {
        CELER_ASSERT_UNREACHABLE();
    }

    // Filename loaded from user
    std::string filename() const { return filename_; }

  private:
    std::string filename_;
};

//---------------------------------------------------------------------------//

class ArbitraryVgdmlTest : public ArbitraryVecgeomTest
{
  public:
    SPConstGeo build_geometry() const final
    {
        return VecgeomParams::from_gdml_vg(this->filename());
    }
};

TEST_F(ArbitraryVgdmlTest, DISABLED_dump)
{
    this->geometry();
    auto const* world = vecgeom::GeoManager::Instance().GetWorld();
    world->PrintContent();
}

//---------------------------------------------------------------------------//

class ArbitraryGeantTest : public ArbitraryVecgeomTest
{
  public:
    SPConstGeo build_geometry() const final
    {
        return VecgeomParams::from_gdml_g4(this->filename());
    }
};

TEST_F(ArbitraryGeantTest, DISABLED_convert)
{
    auto result = GeantImportVolumeResult::from_import(*this->geometry());
    result.print_expected();
    EXPECT_EQ(0, result.missing_labels.size());
}

TEST_F(ArbitraryGeantTest, DISABLED_dump)
{
    this->geometry();
    auto const* world = vecgeom::GeoManager::Instance().GetWorld();
    world->PrintContent();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
