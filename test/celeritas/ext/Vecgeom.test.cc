//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/Vecgeom.test.cc
//---------------------------------------------------------------------------//
#include "Vecgeom.test.hh"

#include "celeritas_cmake_strings.h"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/GlobalTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"
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

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

class VecgeomTestBase : virtual public GlobalTestBase
{
  public:
    //!@{
    using HostStateStore
        = CollectionStateStore<VecgeomStateData, MemSpace::host>;
    //!@}

    struct TrackingResult
    {
        std::vector<std::string> volumes;
        std::vector<real_type>   distances;

        void print_expected();
    };

  public:
    //! Create a host track view
    VecgeomTrackView make_geo_track_view()
    {
        if (!host_state)
        {
            host_state = HostStateStore(this->geometry()->host_ref(), 1);
        }
        return VecgeomTrackView(
            this->geometry()->host_ref(), host_state.ref(), ThreadId(0));
    }

    //! Find linear segments until outside
    TrackingResult track(const Real3& pos, const Real3& dir);

  private:
    HostStateStore host_state;
};

auto VecgeomTestBase::track(const Real3& pos, const Real3& dir)
    -> TrackingResult
{
    const auto& params = *this->geometry();

    TrackingResult result;

    VecgeomTrackView geo = this->make_geo_track_view();
    geo                  = {pos, dir};

    if (geo.is_outside())
    {
        // Initial step is outside but may approach insidfe
        result.volumes.push_back("[OUTSIDE]");
        auto next = geo.find_next_step();
        result.distances.push_back(next.distance);
        if (next.boundary)
        {
            geo.move_to_boundary();
            geo.cross_boundary();
            EXPECT_TRUE(geo.is_on_boundary());
        }
    }

    while (!geo.is_outside())
    {
        result.volumes.push_back(params.id_to_label(geo.volume_id()).name);
        auto next = geo.find_next_step();
        result.distances.push_back(next.distance);
        if (!next.boundary)
        {
            // Failure to find the next boundary while inside the geometry
            ADD_FAILURE();
            result.volumes.push_back("[NO INTERCEPT]");
            break;
        }
        geo.move_to_boundary();
        geo.cross_boundary();
    }

    return result;
}

void VecgeomTestBase::TrackingResult::print_expected()
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
         << "static const char* const expected_volumes[] = "
         << repr(this->volumes) << ";\n"
         << "EXPECT_VEC_EQ(expected_volumes, result.volumes);\n"
         << "static const real_type expected_distances[] = "
         << repr(this->distances) << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);\n"
         << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
// FOUR-LEVELS TEST
//---------------------------------------------------------------------------//

class FourLevelsTest : public GlobalGeoTestBase,
                       public OnlyGeoTestBase,
                       public VecgeomTestBase
{
  public:
    const char* geometry_basename() const final { return "four-levels"; }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, accessors)
{
    const auto& geom = *this->geometry();
    EXPECT_EQ(4, geom.num_volumes());
    EXPECT_EQ(4, geom.max_depth());

    EXPECT_EQ("Shape2", geom.id_to_label(VolumeId{0}).name);
    EXPECT_EQ("Shape1", geom.id_to_label(VolumeId{1}).name);
    EXPECT_EQ("Envelope", geom.id_to_label(VolumeId{2}).name);
    EXPECT_EQ("World", geom.id_to_label(VolumeId{3}).name);
    EXPECT_EQ(Label("World", "0xdeadbeef"), geom.id_to_label(VolumeId{3}));
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, detailed_track)
{
    VecgeomTrackView geo = this->make_geo_track_view();
    geo                  = GeoTrackInitializer{{-10, -10, -10}, {1, 0, 0}};
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

    {
        SCOPED_TRACE("outside in");
        geo = GeoTrackInitializer{{-25, 6.5, 6.5}, {1, 0, 0}};
        EXPECT_TRUE(geo.is_outside());

        next = geo.find_next_step(0.5);
        EXPECT_SOFT_EQ(0.5, next.distance);
        EXPECT_FALSE(next.boundary);

        next = geo.find_next_step(2);
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
        geo = GeoTrackInitializer{{-23.5, 6.5, 6.5}, {-1, 0, 0}};
        EXPECT_FALSE(geo.is_outside());
        EXPECT_EQ(VolumeId{3}, geo.volume_id());

        next = geo.find_next_step(2);
        EXPECT_SOFT_EQ(0.5, next.distance);
        EXPECT_TRUE(next.boundary);

        geo.move_to_boundary();
        EXPECT_FALSE(geo.is_outside());
        geo.cross_boundary();
        EXPECT_TRUE(geo.is_outside());

        next = geo.find_next_step(2);
        EXPECT_SOFT_EQ(2, next.distance);
        EXPECT_FALSE(next.boundary);

        next = geo.find_next_step();
        EXPECT_GT(next.distance, 1e10);
        EXPECT_FALSE(next.boundary);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, tracking)
{
    {
        SCOPED_TRACE("Rightward");
        auto result = this->track({-10, -10, -10}, {1, 0, 0});
        // result.print_expected();
        static const char* const expected_volumes[] = {"Shape2",
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
        static const real_type expected_distances[]
            = {5, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("From outside edge");
        auto result = this->track({-24, 10., 10.}, {1, 0, 0});
        static const char* const expected_volumes[] = {"[OUTSIDE]",
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
        static const real_type expected_distances[]
            = {1e-13, 7.0 - 1e-13, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Leaving world");
        auto result = this->track({-10, 10, 10}, {0, 1, 0});
        static const char* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static const real_type expected_distances[] = {5, 1, 2, 6};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Upward");
        auto result = this->track({-10, 10, 10}, {0, 0, 1});
        static const char* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static const real_type expected_distances[] = {5, 1, 3, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        // Formerly in linear propagator test, used to fail
        SCOPED_TRACE("From just outside world");
        auto result = this->track({-24, 6.5, 6.5}, {1, 0, 0});
        static const char* const expected_volumes[] = {"[OUTSIDE]",
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
        static const real_type expected_distances[] = {1e-13,
                                                       6.9999999999999,
                                                       1,
                                                       5.2928932188135,
                                                       1.4142135623731,
                                                       5.2928932188135,
                                                       1,
                                                       6,
                                                       1,
                                                       5.2928932188135,
                                                       1.4142135623731,
                                                       5.2928932188135,
                                                       1,
                                                       7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//

TEST_F(FourLevelsTest, safety)
{
    VecgeomTrackView       geo = this->make_geo_track_view();
    std::vector<real_type> safeties;

    for (auto i : range(11))
    {
        real_type r = 2.0 * i;
        geo         = {{r, r, r}, {1, 0, 0}};

        if (!geo.is_outside())
        {
            safeties.push_back(geo.find_safety());
        }
    }

    static const real_type expected_safeties[] = {3,
                                                  1,
                                                  0,
                                                  1.92820323027551,
                                                  1.53589838486225,
                                                  5,
                                                  1.53589838486225,
                                                  1.92820323027551,
                                                  0,
                                                  1,
                                                  3};
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
    input.params       = this->geometry()->device_ref();
    input.state        = device_states.ref();

    // Run kernel
    auto output = vgg_test(input);

    static const int expected_ids[]
        = {1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3,
           1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3, 1, 2, 3, -2, -3};

    static const double expected_distances[]
        = {5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3,
           5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3, 5, 1, 1, 7, -3};

    // Check results
    EXPECT_VEC_EQ(expected_ids, output.ids);
    EXPECT_VEC_SOFT_EQ(expected_distances, output.distances);
}

//---------------------------------------------------------------------------//
// SOLIDS TEST
//---------------------------------------------------------------------------//

class SolidsTest : public VecgeomTestBase,
                   public OnlyGeoTestBase,
                   public GlobalGeoTestBase
{
  public:
    const char* geometry_basename() const final { return "solids"; }
};

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, accessors)
{
    if (starts_with(celeritas_vecgeom_version, "1.1"))
    {
        GTEST_SKIP() << "This geometry crashes when loading in VecGeom 1.1.17";
    }

    const auto& geom = *this->geometry();
    // TODO: this should be 31ish?
    EXPECT_EQ(25, geom.num_volumes());
    EXPECT_EQ(2, geom.max_depth());

    EXPECT_EQ("World", geom.id_to_label(VolumeId{24}).name);
    EXPECT_EQ("vol0", geom.id_to_label(VolumeId{4}).name);
    EXPECT_EQ("vol1", geom.id_to_label(VolumeId{5}).name);
    EXPECT_EQ("vol11", geom.id_to_label(VolumeId{9}).name);
}

//---------------------------------------------------------------------------//

TEST_F(SolidsTest, DISABLED_trace)
{
    {
        SCOPED_TRACE("Center +x");
        auto result = this->track({-100, 0, 0}, {1, 0, 0});
        result.print_expected();
    }
    {
        SCOPED_TRACE("Upper +x");
        auto result = this->track({-100, 12.5, 0}, {1, 0, 0});
        result.print_expected();
    }
    {
        SCOPED_TRACE("Lower +x");
        auto result = this->track({-100, -12.5, 0}, {1, 0, 0});
        result.print_expected();
    }
    ADD_FAILURE() << "These are all wrong...";
}

//---------------------------------------------------------------------------//
// CONSTRUCT FROM GEANT4
//---------------------------------------------------------------------------//

class GeantBuilderTestBase : virtual public GeantTestBase,
                             public VecgeomTestBase
{
  public:
    static void SetUpTestCase()
    {
        // Make sure existing VecGeom geometry has been cleared
        GlobalGeoTestBase::reset_geometry();
    }

    SPConstGeo build_geometry() override
    {
        return std::make_shared<VecgeomParams>(this->get_world_volume());
    }

    bool      enable_fluctuation() const final { return false; }
    bool      enable_msc() const final { return false; }
    bool      combined_brems() const final { return false; }
    real_type secondary_stack_factor() const final { return 0; }
};

//---------------------------------------------------------------------------//

#define FourLevelsGeantTest TEST_IF_CELERITAS_GEANT(FourLevelsGeantTest)
class FourLevelsGeantTest : public GeantBuilderTestBase
{
  public:
    const char* geometry_basename() const final { return "four-levels"; }
};

//---------------------------------------------------------------------------//

TEST_F(FourLevelsGeantTest, accessors)
{
    const auto& geom = *this->geometry();
    EXPECT_EQ(4, geom.num_volumes());
    EXPECT_EQ(4, geom.max_depth());

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
        // result.print_expected();
        static const char* const expected_volumes[] = {"Shape2",
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
        static const real_type expected_distances[]
            = {5, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("From outside edge");
        auto result = this->track({-24, 10., 10.}, {1, 0, 0});
        static const char* const expected_volumes[] = {"[OUTSIDE]",
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
        static const real_type expected_distances[]
            = {1e-13, 7.0 - 1e-13, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Leaving world");
        auto result = this->track({-10, 10, 10}, {0, 1, 0});
        static const char* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static const real_type expected_distances[] = {5, 1, 2, 6};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Upward");
        auto result = this->track({-10, 10, 10}, {0, 0, 1});
        static const char* const expected_volumes[]
            = {"Shape2", "Shape1", "Envelope", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static const real_type expected_distances[] = {5, 1, 3, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        // Formerly in linear propagator test, used to fail
        SCOPED_TRACE("From just outside world");
        auto result = this->track({-24, 10, 10}, {1, 0, 0});
        static const char* const expected_volumes[] = {"[OUTSIDE]",
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
        static const real_type expected_distances[]
            = {1e-13, 7, 1, 1, 10, 1, 1, 6, 1, 1, 10, 1, 1, 7};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

//---------------------------------------------------------------------------//

#define SolidsGeantTest TEST_IF_CELERITAS_GEANT(DISABLED_SolidsGeantTest)

class SolidsGeantTest : public GeantBuilderTestBase
{
  public:
    const char* geometry_basename() const final { return "solids"; }
};

//---------------------------------------------------------------------------//

TEST_F(SolidsGeantTest, accessors)
{
    const auto& geom = *this->geometry();
    EXPECT_EQ(31, geom.num_volumes());
    EXPECT_EQ(2, geom.max_depth());

    EXPECT_EQ("World", geom.id_to_label(VolumeId{0}).name);
    EXPECT_EQ("vol0", geom.id_to_label(VolumeId{1}).name);
    EXPECT_EQ("vol1", geom.id_to_label(VolumeId{2}).name);
    EXPECT_EQ("vol11", geom.id_to_label(VolumeId{9}).name);
}

//---------------------------------------------------------------------------//

TEST_F(SolidsGeantTest, DISABLED_trace)
{
    {
        SCOPED_TRACE("Center +x");
        auto                     result = this->track({-100, 0, 0}, {1, 0, 0});
        static const char* const expected_volumes[] = {"World",
                                                       "vol1",
                                                       "World",
                                                       "vol2",
                                                       "World",
                                                       "vol0",
                                                       "World",
                                                       "vol3",
                                                       "World",
                                                       "vol4",
                                                       "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static const real_type expected_distances[]
            = {72.25, 5.5, 9.25, 1, 11.5, 1, 7, 10, 7, 1, 474.5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Upper +x");
        auto result = this->track({-100, 12.5, 0}, {1, 0, 0});
        static const char* const expected_volumes[]
            = {"World", "vol11", "World", "vol21", "World", "vol31", "World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static const real_type expected_distances[]
            = {74.5, 1, 11, 2, 22.000001, 4.999999, 484.5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("Lower +x");
        auto result = this->track({-100, -12.5, 0}, {1, 0, 0});
        static const char* const expected_volumes[] = {"World"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static const real_type expected_distances[] = {600};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    ADD_FAILURE() << "These are all wrong...";
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
