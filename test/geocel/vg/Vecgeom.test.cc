//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/Vecgeom.test.cc
//---------------------------------------------------------------------------//
#include "Vecgeom.test.hh"

#include <string_view>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/Version.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeoParamsOutput.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/vg/VecgeomData.hh"
#include "geocel/vg/VecgeomParams.hh"
#include "geocel/vg/VecgeomTrackView.hh"

#include "VecgeomTestBase.hh"
#include "celeritas_test.hh"

#if CELERITAS_USE_GEANT4
#    include <G4VPhysicalVolume.hh>
#endif

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
auto const geant4_version = celeritas::Version::from_string(
    CELERITAS_USE_GEANT4 ? celeritas_geant4_version : "0.0.0");

}  // namespace

//---------------------------------------------------------------------------//
// TEST HARNESSES
//---------------------------------------------------------------------------//
class VecgeomTestBaseImpl : public VecgeomTestBase
{
  protected:
    using SpanStringView = Span<std::string_view const>;

    virtual SpanStringView expected_log_levels() const { return {}; }
};

/*!
 * Preserve the VecGeom geometry across test cases.
 *
 * Test cases should be matched to unique geometries.
 */
class VecgeomVgdmlTestBase : public VecgeomTestBaseImpl
{
  public:
    //! Helper function: build with VecGeom using VGDML
    SPConstGeo load_vgdml(std::string_view filename)
    {
        ScopedLogStorer scoped_log_{&celeritas::world_logger(),
                                    LogLevel::warning};
        auto result = std::make_shared<VecgeomParams>(
            this->test_data_path("geocel", filename));
        EXPECT_VEC_EQ(this->expected_log_levels(), scoped_log_.levels())
            << scoped_log_;
        return result;
    }
};

//---------------------------------------------------------------------------//

class VecgeomGeantTestBase : public VecgeomTestBaseImpl
{
  public:
    //! Helper function: build via Geant4 GDML reader
    SPConstGeo load_g4_gdml(std::string_view filename)
    {
        if (world_volume_)
        {
            // Clear old geant4 data
            ::celeritas::reset_geant_geometry();
        }
        ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                    LogLevel::warning};
        world_volume_ = ::celeritas::load_geant_geometry_native(
            this->test_data_path("geocel", filename));
        auto result = std::make_shared<VecgeomParams>(world_volume_);
        EXPECT_VEC_EQ(this->expected_log_levels(), scoped_log_.levels())
            << scoped_log_;
        return result;
    }

    //! Test conversion for Geant4 geometry
    GeantVolResult get_direct_geant_volumes()
    {
        this->geometry();
        return VecgeomTestBase::get_direct_geant_volumes(world_volume_);
    }

    //! Test conversion for Geant4 geometry
    GeantVolResult get_import_geant_volumes()
    {
        this->geometry();
        return VecgeomTestBase::get_import_geant_volumes(world_volume_);
    }

    virtual SpanStringView expected_log_levels() const { return {}; }

  protected:
    // Note that this is static because the geometry may be loaded
    static G4VPhysicalVolume* world_volume_;
};

G4VPhysicalVolume* VecgeomGeantTestBase::world_volume_{nullptr};

//---------------------------------------------------------------------------//
// SIMPLE CMS TEST
//---------------------------------------------------------------------------//

class SimpleCmsTest : public VecgeomVgdmlTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_vgdml("simple-cms.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, accessors)
{
    auto const& geom = *this->geometry();
    EXPECT_EQ(2, geom.max_depth());
    EXPECT_EQ(7, geom.volumes().size());
}

//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, track)
{
    auto geo = this->make_geo_track_view({0, 0, 0}, {0, 0, 1});
    EXPECT_EQ("vacuum_tube", this->volume_name(geo));

    auto next = geo.find_next_step(from_cm(100));
    EXPECT_SOFT_EQ(100, to_cm(next.distance));
    EXPECT_FALSE(next.boundary);
    geo.move_internal(from_cm(20));
    EXPECT_SOFT_EQ(30, to_cm(geo.find_safety()));

    geo.set_dir({1, 0, 0});
    next = geo.find_next_step(from_cm(50));
    EXPECT_SOFT_EQ(30, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);

    geo.move_to_boundary();
    EXPECT_FALSE(geo.is_outside());
    geo.cross_boundary();
    EXPECT_EQ("si_tracker", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({30, 0, 20}), to_cm(geo.pos()));

    // Scatter to tangent
    geo.set_dir({0, 1, 0});
    next = geo.find_next_step(from_cm(1000));
    EXPECT_SOFT_EQ(121.34661099511597, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_internal(from_cm(10));
    EXPECT_SOFT_EQ(1.6227766016837926, to_cm(geo.find_safety()));

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

TEST_F(SimpleCmsTest, TEST_IF_CELERITAS_CUDA(device))
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
// FOUR-LEVELS TEST
//---------------------------------------------------------------------------//

class FourLevelsTest : public VecgeomVgdmlTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_vgdml("four-levels.gdml");
    }

    SpanStringView expected_log_levels() const final
    {
        if (vecgeom_version >= Version{2})
        {
            static std::string_view const levels[] = {"warning"};
            return make_span(levels);
        }
        else
        {
            return {};
        }
    }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, accessors)
{
    auto const& geom = *this->geometry();
    EXPECT_EQ(4, geom.max_depth());

    auto const& bbox = geom.bbox();
    EXPECT_VEC_SOFT_EQ((Real3{-24.001, -24.001, -24.001}), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{24.001, 24.001, 24.001}), to_cm(bbox.upper()));

    ASSERT_EQ(4, geom.volumes().size());
    EXPECT_EQ("Shape2", geom.volumes().at(VolumeId{0}).name);
    EXPECT_EQ("Shape1", geom.volumes().at(VolumeId{1}).name);
    EXPECT_EQ("Envelope", geom.volumes().at(VolumeId{2}).name);
    EXPECT_EQ("World", geom.volumes().at(VolumeId{3}).name);
    EXPECT_EQ(Label("World", "0xdeadbeef"), geom.volumes().at(VolumeId{3}));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, consecutive_compute)
{
    auto geo = this->make_geo_track_view({-9, -10, -10}, {1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ(VolumeId{0}, geo.volume_id());
    EXPECT_FALSE(geo.is_on_boundary());

    auto next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));

    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(4.0, to_cm(next.distance));
    EXPECT_SOFT_EQ(4.0, to_cm(geo.find_safety()));
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
        EXPECT_EQ(VolumeId{0}, geo.volume_id());
        geo.cross_boundary();
        EXPECT_EQ(VolumeId{1}, geo.volume_id());
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
        SCOPED_TRACE("outside in");
        auto geo = this->make_geo_track_view({-25, 6.5, 6.5}, {1, 0, 0});
        EXPECT_TRUE(geo.is_outside());

        auto next = geo.find_next_step();
        EXPECT_SOFT_EQ(1.0, to_cm(next.distance));
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

        auto next = geo.find_next_step(from_cm(2));
        EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
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
    auto geo = this->make_geo_track_view({15.5, 10, 10}, {-1, 0, 0});
    ASSERT_FALSE(geo.is_outside());
    EXPECT_EQ("Shape1", this->volume_name(geo));
    EXPECT_FALSE(geo.is_on_boundary());

    // Check for surfaces: we should hit the outside of the sphere Shape2
    auto next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(0.5, to_cm(next.distance));
    // Move to the boundary but scatter perpendicularly, away from the sphere
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, 1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    EXPECT_EQ("Shape1", this->volume_name(geo));

    // Move a bit internally, then scatter back toward the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(6, to_cm(next.distance));
    geo.set_dir({-1, 0, 0});
    EXPECT_EQ("Shape1", this->volume_name(geo));

    // Move to the sphere boundary then scatter still into the sphere
    next = geo.find_next_step(from_cm(10.0));
    EXPECT_SOFT_EQ(1e-8, to_cm(next.distance));
    EXPECT_TRUE(next.boundary);
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({0, -1, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ("Shape2", this->volume_name(geo));
    EXPECT_TRUE(geo.is_on_boundary());

    // Travel nearly tangent to the right edge of the sphere, then scatter to
    // still outside
    next = geo.find_next_step(from_cm(1.0));
    EXPECT_SOFT_EQ(0.00031622777925735285, to_cm(next.distance));
    geo.move_to_boundary();
    EXPECT_TRUE(geo.is_on_boundary());
    geo.set_dir({1, 0, 0});
    EXPECT_TRUE(geo.is_on_boundary());
    geo.cross_boundary();
    EXPECT_EQ("Shape1", this->volume_name(geo));

    EXPECT_TRUE(geo.is_on_boundary());
    next = geo.find_next_step(from_cm(10.0));
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
    std::vector<real_type> lim_safeties;

    for (auto i : range(11))
    {
        real_type r = 2.0 * i + 0.1;
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

    static double const expected_lim_safeties[]
        = {1.5, 0.9, 0.1, 1.5, 1.5, 1.5, 1.3626933041054, 1.5, 0.1, 1.1, 1.5};
    EXPECT_VEC_SOFT_EQ(expected_lim_safeties, lim_safeties);
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
// MULTI-LEVEL TEST
//---------------------------------------------------------------------------//

class MultiLevelTest : public VecgeomVgdmlTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_vgdml("multi-level.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(MultiLevelTest, accessors)
{
    auto const& geo = *this->geometry();
    EXPECT_EQ(3, geo.max_depth());

    auto vol_names = [&geo] {
        auto const& vols = geo.volumes();
        std::vector<std::string> result;
        for (auto vid : range(VolumeId{vols.size()}))
        {
            result.push_back(vols.at(vid).name);
        }
        return result;
    }();
    static std::string const expected_vol_names[] = {"sph", "box", "world"};
    EXPECT_VEC_EQ(expected_vol_names, vol_names);

    auto vol_inst_names = [&geo] {
        auto const& vols = geo.volume_instances();
        std::vector<std::string> result;
        for (auto viid : range(VolumeInstanceId{vols.size()}))
        {
            result.push_back(vols.at(viid).name);
        }
        return result;
    }();
    static std::string const expected_vol_inst_names[] = {
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

//---------------------------------------------------------------------------//

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
//---------------------------------------------------------------------------//
// SOLIDS TEST
//---------------------------------------------------------------------------//

class SolidsTest : public VecgeomVgdmlTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_vgdml("solids.gdml");
    }

    SpanStringView expected_log_levels() const final
    {
        if (vecgeom_version >= Version{2})
        {
            static std::string_view const levels[]
                = {"warning", "warning", "warning"};
            return make_span(levels);
        }
        else if (geant4_version >= Version{11})
        {
            static std::string_view const levels[] = {"warning"};
            return make_span(levels);
        }
        else
        {
            // Vecgeom 1 and Geant4 10 have no warnings
            return {};
        }
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
    EXPECT_VEC_SOFT_EQ((Real3{-600.001, -300.001, -75.001}),
                       to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{600.001, 300.001, 75.001}), to_cm(bbox.upper()));
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, names)
{
    auto const& geom = *this->geometry();
    std::vector<std::string> labels;
    for (auto vid : range(VolumeId{geom.volumes().size()}))
    {
        labels.push_back(
            this->genericize_pointers(to_string(geom.volumes().at(vid))));
    }

    // clang-format off
    static char const* const expected_labels[] = {"b500", "b100", "union1",
        "b100", "box500", "cone1", "para1", "sphere1", "parabol1", "trap1",
        "trd1", "trd2", "trd3", "trd3_refl", "tube100", "boolean1",
        "polycone1", "genPocone1", "ellipsoid1", "tetrah1", "orb1",
        "polyhedr1", "hype1", "elltube1", "ellcone1", "arb8b", "arb8a",
        "xtru1", "World", "", "trd3_refl"};
    // clang-format on
    EXPECT_VEC_EQ(expected_labels, labels);
}

//---------------------------------------------------------------------------//
TEST_F(SolidsTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        auto out_str = this->genericize_pointers(to_string(out));

        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-600.001,-300.001,-75.001],[600.001,300.001,75.001]],"max_depth":2,"supports_safety":true,"volumes":{"label":["b500","b100","union1","b100","box500","cone1","para1","sphere1","parabol1","trap1","trd1","trd2","trd3","trd3_refl","tube100","boolean1","polycone1","genPocone1","ellipsoid1","tetrah1","orb1","polyhedr1","hype1","elltube1","ellcone1","arb8b","arb8a","xtru1","World","","trd3_refl"]}})json",
            out_str);
    }
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
                                                       "trd2",
                                                       "World",
                                                       "xtru1",
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
                                                       88.786678713601,
                                                       30,
                                                       1.4761904761905,
                                                       15.880952380952,
                                                       67.642857142857};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0,
                                                       45.496748548005,
                                                       0,
                                                       44.83475568122,
                                                       13.934134186943,
                                                       30,
                                                       25,
                                                       36.240004604773,
                                                       25,
                                                       41.204388797207,
                                                       14.92555785315,
                                                       42.910442345001,
                                                       18.741024106017,
                                                       42.910442345001,
                                                       14.92555785315,
                                                       0.7344322118216,
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
                                                       116.000001,
                                                       12.499999,
                                                       20.000001,
                                                       17.499999,
                                                       191.98703789108,
                                                       25.9774128070174,
                                                       14.0710986038011,
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

        if (vecgeom_version < Version(1, 2, 3))
        {
            ADD_FAILURE()
                << "VecGeom " << vecgeom_version
                << " does not correctly trace through some polycones: "
                   "upgrade to 1.2.3 to pass this test";
        }

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
    auto const& label = this->geometry()->volumes().at(geo.volume_id());
    // Note: through GDML there is only one trd3_refl
    EXPECT_EQ("trd3_refl", label.name);
    EXPECT_FALSE(ends_with(label.ext, "_refl"));
}

//---------------------------------------------------------------------------//
// CMS EXTERIOR TEST
//---------------------------------------------------------------------------//

class CmseTest : public VecgeomVgdmlTestBase
{
  public:
    SPConstGeo build_geometry() final { return this->load_vgdml("cmse.gdml"); }

    SpanStringView expected_log_levels() const final
    {
        if (vecgeom_version >= Version(2))
        {
            static std::string_view const levels[] = {"warning"};
            return make_span(levels);
        }
        else
        {
            return {};
        }
    }
};

//---------------------------------------------------------------------------//

TEST_F(CmseTest, DISABLED_trace)
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
            0.52499999999986, 13.023518051922, 6.95, 6.95, 13.023518051922,
            0.52499999999986, 2.15, 100, 5, 8, 100, 100, 100};
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
            58.4195, 0.078366388350241, 2.343262600759, 0.078366388350241,
            58.4195, 85.85, 267.5, 460};
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
        static real_type const expected_hw_safety[] = {1, 1, 242, 460};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    // clang-format on
}

//---------------------------------------------------------------------------//
// ARBITRARY VGDML TEST
//---------------------------------------------------------------------------//

#define ArbitraryVgdmlTest DISABLED_ArbitraryVgdmlTest
class ArbitraryVgdmlTest : public VecgeomTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        auto filename = celeritas::getenv("GDML");
        CELER_VALIDATE(!filename.empty(),
                       << "Set the 'GDML' environment variable and run this "
                          "test with "
                          "--gtest_filter=*ArbitraryVgdmlTest* "
                          "--gtest_also_run_disabled_tests");
        return std::make_shared<VecgeomParams>(filename);
    }
};

TEST_F(ArbitraryVgdmlTest, dump)
{
    this->geometry();
    auto const* world = vecgeom::GeoManager::Instance().GetWorld();
    world->PrintContent();
}

//---------------------------------------------------------------------------//
// CONSTRUCT FROM GEANT4
//---------------------------------------------------------------------------//

#define FourLevelsGeantTest TEST_IF_CELERITAS_GEANT(FourLevelsGeantTest)
class FourLevelsGeantTest : public VecgeomGeantTestBase
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
    EXPECT_VEC_SOFT_EQ((Real3{-24.001, -24.001, -24.001}), to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{24.001, 24.001, 24.001}), to_cm(bbox.upper()));

    ASSERT_EQ(4, geom.volumes().size());
    EXPECT_EQ("Shape2", geom.volumes().at(VolumeId{0}).name);
    EXPECT_EQ("Shape1", geom.volumes().at(VolumeId{1}).name);
    EXPECT_EQ("Envelope", geom.volumes().at(VolumeId{2}).name);
    EXPECT_EQ("World", geom.volumes().at(VolumeId{3}).name);
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

TEST_F(FourLevelsGeantTest, levels)
{
    auto geo = this->make_geo_track_view({10.0, 10.0, 10.0}, {1, 0, 0});
    EXPECT_EQ("World_PV/env1/Shape1/Shape2",
              this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV/env1/Shape1", this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV/env1", this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("World_PV", this->all_volume_instance_names(geo));
    geo.find_next_step();
    geo.move_to_boundary();
    geo.cross_boundary();

    EXPECT_EQ("[OUTSIDE]", this->all_volume_instance_names(geo));
}

//---------------------------------------------------------------------------//

class MultiLevelGeantTest : public VecgeomGeantTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_g4_gdml("multi-level.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(MultiLevelGeantTest, accessors)
{
    auto const& geo = *this->geometry();
    EXPECT_EQ(3, geo.max_depth());

    auto vol_names = [&geo] {
        auto const& vols = geo.volumes();
        std::vector<std::string> result;
        for (auto vid : range(VolumeId{vols.size()}))
        {
            result.push_back(vols.at(vid).name);
        }
        return result;
    }();
    static char const* const expected_vol_names[] = {"sph", "box", "world"};
    EXPECT_VEC_EQ(expected_vol_names, vol_names);

    auto vol_inst_names = [&geo] {
        auto const& vols = geo.volume_instances();
        std::vector<std::string> result;
        for (auto viid : range(VolumeInstanceId{vols.size()}))
        {
            result.push_back(vols.at(viid).name);
        }
        return result;
    }();
    static char const* const expected_vol_inst_names[] = {
        "topsph1",
        "boxsph1",
        "boxsph2",
        "topbox1",
        "topbox2",
        "topbox3",
        "topsph2",
        "world_PV",
    };
    EXPECT_VEC_EQ(expected_vol_inst_names, vol_inst_names);

    auto g4names = [&geo] {
        std::vector<std::string> result;
        for (auto viid : range(VolumeInstanceId{geo.volume_instances().size()}))
        {
#if CELERITAS_USE_GEANT4
            if (auto* g4pv = geo.id_to_pv(viid))
            {
                result.push_back(g4pv->GetName());
            }
            else
#endif
            {
                result.push_back("<NULL>");
            }
        }
        return result;
    }();
    EXPECT_VEC_EQ(expected_vol_inst_names, g4names);
}

//---------------------------------------------------------------------------//

#define SolidsGeantTest TEST_IF_CELERITAS_GEANT(SolidsGeantTest)
class SolidsGeantTest : public VecgeomGeantTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_g4_gdml("solids.gdml");
    }

    SpanStringView expected_log_levels() const final
    {
        static std::string_view const levels[] = {"error"};
        return make_span(levels);
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
    EXPECT_VEC_SOFT_EQ((Real3{-600.001, -300.001, -75.001}),
                       to_cm(bbox.lower()));
    EXPECT_VEC_SOFT_EQ((Real3{600.001, 300.001, 75.001}), to_cm(bbox.upper()));
}

//---------------------------------------------------------------------------//

TEST_F(SolidsGeantTest, names)
{
    auto const& geom = *this->geometry();
    std::vector<std::string> labels;
    for (auto vid : range(VolumeId{geom.volumes().size()}))
    {
        labels.push_back(
            this->genericize_pointers(to_string(geom.volumes().at(vid))));
    }

    // clang-format off
    static char const* const expected_labels[] = {"box500@0x0", "cone1@0x0",
        "para1@0x0", "sphere1@0x0", "parabol1@0x0", "trap1@0x0", "trd1@0x0",
        "trd2@0x0", "trd3@0x0", "trd3_refl@0x0", "tube100@0x0", "", "", "", "",
        "boolean1@0x0", "polycone1@0x0", "genPocone1@0x0", "ellipsoid1@0x0",
        "tetrah1@0x0", "orb1@0x0", "polyhedr1@0x0", "hype1@0x0",
        "elltube1@0x0", "ellcone1@0x0", "arb8b@0x0", "arb8a@0x0", "xtru1@0x0",
        "World@0x0", "", "trd3@0x0_refl"};;
    // clang-format on
    EXPECT_VEC_EQ(expected_labels, labels);
}

//---------------------------------------------------------------------------//
TEST_F(SolidsGeantTest, output)
{
    GeoParamsOutput out(this->geometry());
    EXPECT_EQ("geometry", out.label());

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        auto out_str = this->genericize_pointers(to_string(out));

        EXPECT_JSON_EQ(
            R"json({"_category":"internal","_label":"geometry","bbox":[[-600.001,-300.001,-75.001],[600.001,300.001,75.001]],"max_depth":2,"supports_safety":true,"volumes":{"label":["box500@0x0","cone1@0x0","para1@0x0","sphere1@0x0","parabol1@0x0","trap1@0x0","trd1@0x0","trd2@0x0","trd3@0x0","trd3_refl@0x0","tube100@0x0","","","","","boolean1@0x0","polycone1@0x0","genPocone1@0x0","ellipsoid1@0x0","tetrah1@0x0","orb1@0x0","polyhedr1@0x0","hype1@0x0","elltube1@0x0","ellcone1@0x0","arb8b@0x0","arb8a@0x0","xtru1@0x0","World@0x0","","trd3@0x0_refl"]}})json",
            out_str);
    }
}

//---------------------------------------------------------------------------//

TEST_F(SolidsGeantTest, geant_volumes)
{
    {
        auto result = this->get_import_geant_volumes();
        static int const expected_volumes[]
            = {0,  1,  2,  3,  4,  5,  6,  7,  -1, 9,  10, 15, 16,
               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        EXPECT_EQ(0, result.missing_names.size()) << repr(result.missing_names);
    }
    {
        auto result = this->get_direct_geant_volumes();
        static int const expected_volumes[]
            = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 15, 16,
               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, -2};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);

        static char const* const expected_missing[] = {"trd3_refl"};
        EXPECT_VEC_EQ(expected_missing, result.missing_names);
    }
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
                                                       "trd2",
                                                       "World",
                                                       "xtru1",
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
                                                       88.786678713601,
                                                       30,
                                                       1.4761904761905,
                                                       15.880952380952,
                                                       67.642857142857};
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
                                                       25.9774128070174,
                                                       14.0710986038011,
                                                       25.977412807017,
                                                       86.987037891082,
                                                       10,
                                                       220};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Lower +x");

        if (vecgeom_version < Version(1, 2, 3))
        {
            ADD_FAILURE()
                << "VecGeom " << vecgeom_version
                << " does not correctly trace through some polycones: "
                   "upgrade to 1.2.3 to pass this test";
        }

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
    {
        SCOPED_TRACE("Low +y");
        auto result = this->track({-500, -250, 0}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"World", "trd3", "World", "trd2", "World"};
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

TEST_F(SolidsGeantTest, reflected_vol)
{
    auto geo = this->make_geo_track_view({-500, -125, 0}, {0, 1, 0});
    EXPECT_EQ(VolumeId{30}, geo.volume_id());
    auto const& label = this->geometry()->volumes().at(geo.volume_id());
    EXPECT_EQ("trd3", label.name);
    EXPECT_TRUE(ends_with(label.ext, "_refl"));
}

//---------------------------------------------------------------------------//

#define ZnenvGeantTest TEST_IF_CELERITAS_GEANT(ZnenvGeantTest)
class ZnenvGeantTest : public VecgeomGeantTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        return this->load_g4_gdml("znenv.gdml");
    }
};

//---------------------------------------------------------------------------//

TEST_F(ZnenvGeantTest, trace)
{
    // NOTE: This tests the capability of the G4PVDivision conversion based on
    // an ALICE component
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

#define ArbitraryGeantTest DISABLED_ArbitraryGeantTest
class ArbitraryGeantTest : public VecgeomGeantTestBase
{
  public:
    SPConstGeo build_geometry() final
    {
        auto filename = celeritas::getenv("GDML");
        CELER_VALIDATE(!filename.empty(),
                       << "Set the 'GDML' environment variable and run this "
                          "test with "
                          "--gtest_filter=*ArbitraryGeantTest* "
                          "--gtest_also_run_disabled_tests");
        if (world_volume_)
        {
            // Clear old geant4 data
            ::celeritas::reset_geant_geometry();
        }
        world_volume_ = ::celeritas::load_geant_geometry_native(filename);
        return std::make_shared<VecgeomParams>(world_volume_);
    }
};

TEST_F(ArbitraryGeantTest, conversion)
{
    auto result = this->get_import_geant_volumes();
    result.print_expected();
    EXPECT_EQ(0, result.missing_names.size());
}

TEST_F(ArbitraryGeantTest, dump)
{
    this->geometry();
    auto const* world = vecgeom::GeoManager::Instance().GetWorld();
    world->PrintContent();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
