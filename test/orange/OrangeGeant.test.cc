//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeGeant.test.cc
//---------------------------------------------------------------------------//
#include <string>

#include "corecel/Config.hh"

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/StringSimplifier.hh"
#include "corecel/Types.hh"
#include "geocel/GenericGeoParameterizedTest.hh"
#include "geocel/GeoTests.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeToString.hh"
#include "geocel/detail/LengthUnits.hh"
#include "geocel/rasterize/SafetyImager.hh"
#include "orange/Debug.hh"
#include "orange/OrangeTypes.hh"

#include "OrangeTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//! Avoid relying on integer size assumptions and overflow
int vluint_to_int(vol_level_uint vl)
{
    if (vl == static_cast<vol_level_uint>(-1))
        return -1;

    return vl;
}

}  // namespace

//---------------------------------------------------------------------------//

class GeantOrangeTest : public OrangeTestBase
{
  protected:
    void SetUp() final { this->geometry(); }

    //! Check log messages
    SPConstGeo build_geometry() const final
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::error};
        auto result = OrangeTestBase::build_geometry();
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
        return result;
    }

    Constant unit_length() const final { return lengthunits::centimeter; }
};

//---------------------------------------------------------------------------//
using FourLevelsTest
    = GenericGeoParameterizedTest<GeantOrangeTest, FourLevelsGeoTest>;

TEST_F(FourLevelsTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(FourLevelsTest, trace)
{
    this->impl().test_trace();
}

TEST_F(FourLevelsTest, consecutive_compute)
{
    this->impl().test_consecutive_compute();
}

TEST_F(FourLevelsTest, detailed_track)
{
    this->impl().test_detailed_tracking();
}

//---------------------------------------------------------------------------//
using LarSphereTest
    = GenericGeoParameterizedTest<GeantOrangeTest, LarSphereGeoTest>;

TEST_F(LarSphereTest, trace)
{
    this->impl().test_trace();
}

TEST_F(LarSphereTest, volume_stack)
{
    this->impl().test_volume_stack();
}

//---------------------------------------------------------------------------//
class MultiLevelTest
    : public GenericGeoParameterizedTest<GeantOrangeTest, MultiLevelGeoTest>
{
};

// Test the stack/volume points to see what universe and local volume they
// translate to
TEST_F(MultiLevelTest, univ_levels)
{
    std::vector<int> univ_levels;
    std::vector<int> univ_ids;
    std::vector<int> local_volumes;
    for (auto xy : this->impl().get_test_points())
    {
        auto geo = this->make_geo_track_view().track_view();
        geo = this->make_initializer({xy[0], xy[1], 0}, {0, 0, 1});
        univ_levels.push_back(id_to_int(geo.univ_level()));
        auto lsa = geo.make_lsa();
        local_volumes.push_back(id_to_int(lsa.vol()));
        univ_ids.push_back(id_to_int(lsa.univ()));
    }
    // clang-format off
    static int const expected_univ_levels[] = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,};
    static int const expected_univ_id[] = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,};
    static int const expected_local_volumes[] = {6, 5, 1, 4, 3, 2, 1, 4, 3, 2, 3, 2, 1, 4, 4, 2, 1, 3,};
    // clang-format on
    EXPECT_VEC_EQ(expected_univ_levels, univ_levels);
    EXPECT_VEC_EQ(expected_univ_id, univ_ids);
    EXPECT_VEC_EQ(expected_local_volumes, local_volumes);
}

// Check the explicit "local volume level" and "parent" for each impl volume
TEST_F(MultiLevelTest, manual_volumes)
{
    // VolumeToString to_string(*this->volumes());
    auto const& universe_labels = this->geometry()->universes();
    auto const& impl_volumes = this->geometry()->impl_volumes();
    TrackerVisitor visit_tracker{this->geometry()->host_ref()};

    std::vector<std::vector<int>> local_level;
    std::vector<std::vector<int>> local_parent;
    std::vector<std::vector<std::string>> volume_names;
    ImplVolumeId global_vol{0};
    for (auto uid : range(UnivId{universe_labels.size()}))
    {
        auto num_local_vols = visit_tracker(
            [](auto const& t) { return t.num_volumes(); }, uid);

        std::vector<int> cur_local_level;
        std::vector<int> cur_local_parent;
        std::vector<std::string> cur_volume_names;
        for (auto lv_id : range(LocalVolumeId{num_local_vols}))
        {
            cur_local_level.push_back(vluint_to_int(visit_tracker(
                [lv_id](auto const& t) { return t.local_vol_level(lv_id); },
                uid)));
            cur_local_parent.push_back(id_to_int(visit_tracker(
                [lv_id](auto const& t) { return t.local_parent(lv_id); }, uid)));
            cur_volume_names.push_back(impl_volumes.at(global_vol++).name);
        }
        local_level.emplace_back(std::move(cur_local_level));
        local_parent.emplace_back(std::move(cur_local_parent));
        volume_names.emplace_back(std::move(cur_volume_names));
    }
    static std::vector<int> const expected_local_level[]
        = {{-1, 1, 1, 1, 1, 1, 0}, {-1, 1, 1, 1, 0}, {-1, 1, 1, 1, 0}};
    static std::vector<int> const expected_local_parent[]
        = {{-1, 6, 6, 6, 6, 6, -1}, {-1, 4, 4, 4, -1}, {-1, 4, 4, 4, -1}};
    static std::vector<std::string> const expected_volume_names[]
        = {{"[EXTERIOR]", "box", "box", "box", "box_refl", "sph", "world"},
           {"[EXTERIOR]", "sph", "sph", "tri", "box"},
           {"[EXTERIOR]", "sph_refl", "sph_refl", "tri_refl", "box_refl"}};
    EXPECT_VEC_EQ(expected_local_level, local_level);
    EXPECT_VEC_EQ(expected_local_parent, local_parent);
    EXPECT_VEC_EQ(expected_volume_names, volume_names);
}

// Test that the reconstructed total levels are correct
TEST_F(MultiLevelTest, volume_level)
{
    this->impl().test_volume_level();
}

// Test that the reconstructed volume instance hierarchy is correct
TEST_F(MultiLevelTest, volume_stack)
{
    this->impl().test_volume_stack();
}

TEST_F(MultiLevelTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

class PincellTest : public GeantOrangeTest
{
    std::string_view gdml_basename() const final { return "pincell"; }
};

TEST_F(PincellTest, imager)
{
    SafetyImager write_image{this->geometry()};

    ImageInput inp;
    inp.lower_left = from_cm({-12, -12, 0});
    inp.upper_right = from_cm({12, 12, 0});
    inp.rightward = {1.0, 0.0, 0.0};
    inp.vertical_pixels = 16;

    write_image(ImageParams{inp}, "org-pincell-xy-mid.jsonl");

    inp.lower_left[2] = inp.upper_right[2] = from_cm(-5.5);
    write_image(ImageParams{inp}, "org-pincell-xy-lo.jsonl");

    inp.lower_left = from_cm({-12, 0, -12});
    inp.upper_right = from_cm({12, 0, 12});
    write_image(ImageParams{inp}, "org-pincell-xz-mid.jsonl");
}

//---------------------------------------------------------------------------//
using PolyhedraTest
    = GenericGeoParameterizedTest<GeantOrangeTest, PolyhedraGeoTest>;

TEST_F(PolyhedraTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
class ReplicaTest
    : public GenericGeoParameterizedTest<GeantOrangeTest, ReplicaGeoTest>
{
  public:
    //! Distance is slightly off for single precision
    GenericGeoTrackingTolerance tracking_tol() const override
    {
        auto result = GeantOrangeTest::tracking_tol();

        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
        {
            result.distance *= 10;
        }

        return result;
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
using SimpleCmsTest
    = GenericGeoParameterizedTest<GeantOrangeTest, SimpleCmsGeoTest>;

TEST_F(SimpleCmsTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using TestEm3Test
    = GenericGeoParameterizedTest<GeantOrangeTest, TestEm3GeoTest>;

TEST_F(TestEm3Test, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
using TestEm3FlatTest
    = GenericGeoParameterizedTest<GeantOrangeTest, TestEm3FlatGeoTest>;

TEST_F(TestEm3FlatTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//
class TilecalPlugTest : public GeantOrangeTest
{
    std::string_view gdml_basename() const final { return "tilecal-plug"; }
};

TEST_F(TilecalPlugTest, trace)
{
    {
        SCOPED_TRACE("lo x");
        auto result = this->track({5.75, 0.01, -40}, {0, 0, 1});
        static char const* const expected_volumes[] = {
            "Tile_ITCModule",
            "Tile_Plug1Module",
            "Tile_Absorber",
            "Tile_Plug1Module",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {22.9425, 0.115, 42, 37};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("hi x");
        auto result = this->track({6.25, 0.01, -40}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"Tile_ITCModule", "Tile_Absorber", "Tile_Plug1Module"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {23.0575, 42, 37};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//
using TransformedBoxTest
    = GenericGeoParameterizedTest<GeantOrangeTest, TransformedBoxGeoTest>;

TEST_F(TransformedBoxTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TransformedBoxTest, trace)
{
    this->impl().test_trace();
}

//---------------------------------------------------------------------------//

class TwoBoxesTest
    : public GenericGeoParameterizedTest<GeantOrangeTest, TwoBoxesGeoTest>
{
};

TEST_F(TwoBoxesTest, accessors)
{
    this->impl().test_accessors();
}

TEST_F(TwoBoxesTest, reentrant)
{
    this->impl().test_reentrant();
}

TEST_F(TwoBoxesTest, reentrant_undo)
{
    this->impl().test_reentrant_undo();
}

TEST_F(TwoBoxesTest, tangent)
{
    this->impl().test_tangent();
}

TEST_F(TwoBoxesTest, track)
{
    this->impl().test_detailed_tracking();
}

//---------------------------------------------------------------------------//
using ZnenvTest = GenericGeoParameterizedTest<GeantOrangeTest, ZnenvGeoTest>;

TEST_F(ZnenvTest, trace)
{
    this->impl().test_trace();
}

TEST_F(ZnenvTest, debug)
{
    auto geo = this->make_geo_track_view();
    geo = GeoTrackInitializer{{0.1, 0.0001, 0}, {1, 0, 0}};
    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(
            R"json({"levels":[
{"dir":[1.0,0.0,0.0],"pos":[0.1,1e-4,0.0],"universe":"World","volume":{"canonical":"ZNTX","impl":"ZNTX","instance":"ZNTX_PV@1","local":2}},
{"dir":[1.0,0.0,0.0],"pos":[-1.66,1e-4,0.0],"universe":"ZNTX","volume":{"canonical":"ZN1","impl":"ZN1","instance":"ZN1_PV@1","local":2}},
{"dir":[1.0,0.0,0.0],"pos":[-1.66,-1.76,0.0],"universe":"ZN1","volume":{"canonical":"ZNSL","impl":"ZNSL","instance":"ZNSL_PV@0","local":1}},
{"dir":[1.0,0.0,0.0],"pos":[-1.66,-0.160,0.0],"universe":"ZNSL","volume":{"canonical":"ZNST","impl":"ZNST","instance":"ZNST_PV@0","local":1}},
{"dir":[1.0,0.0,0.0],"pos":[-0.0600,-0.160,0.0],"universe":"ZNST","volume":{"canonical":"ZNST","impl":"ZNST","instance":null,"local":5}}],
"surface":null})json",
            StringSimplifier{3}(to_json_string(geo.track_view())));
    }
    else
    {
        GTEST_SKIP() << "no gold results for this unit system";
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
