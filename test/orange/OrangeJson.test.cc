//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeJson.test.cc
//---------------------------------------------------------------------------//

#include <string>
#include <vector>

#include "corecel/StringSimplifier.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/Types.hh"
#include "orange/Debug.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeParamsOutput.hh"
#include "orange/OrangeTrackView.hh"

#include "OrangeGeoTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Test precompiled geometry loaded from OrangeInput
class JsonOrangeTest : public OrangeGeoTestBase
{
  public:
    using Initializer_t = GeoTrackInitializer;

    //! Use multiple track slots to check detailed initialization
    size_type num_track_slots() const override { return 2; }
    //! Length scale is hardcoded into JSON files
    UnitLength unit_length() const override { return {Constant{1}, "length"}; }

    virtual std::string_view geometry_basename() const = 0;

    void SetUp() final
    {
        this->build_geometry(std::string{this->geometry_basename()}
                             + ".org.json");
    }

    std::string surface_name(WrappedGeoTrack const& geo) const
    {
        if (!geo.is_on_boundary())
        {
            return "---";
        }
        // Access OrangeTrackView through WrappedTrackView
        auto impl_surface = geo.track_view().impl_surface_id();
        return this->geometry()->surfaces().at(impl_surface).name;
    }
};

class InputBuilderTest : public JsonOrangeTest
{
  public:
    std::string_view geometry_basename() const final
    {
        if (basename_.empty())
        {
            auto* mthis = const_cast<InputBuilderTest*>(this);
            mthis->basename_ = mthis->make_unique_filename();
        }
        return basename_;
    }

    std::string_view gdml_basename() const override
    {
        return geometry_basename();
    }

  protected:
    bool supports_surface_normal_{true};

  private:
    std::string basename_;
};

//---------------------------------------------------------------------------//
class FiveVolumesTest : public JsonOrangeTest
{
    std::string_view geometry_basename() const final { return "five-volumes"; }
};

TEST_F(FiveVolumesTest, params)
{
    OrangeParams const& geo = this->params();

    EXPECT_EQ(6, geo.impl_volumes().size());
    EXPECT_EQ(12, geo.surfaces().size());
    EXPECT_FALSE(geo.supports_safety());
}

//---------------------------------------------------------------------------//
class UniversesTest : public JsonOrangeTest
{
    std::string_view geometry_basename() const final { return "universes"; }
};

TEST_F(UniversesTest, params)
{
    OrangeParams const& geo = this->params();
    EXPECT_EQ(12, geo.impl_volumes().size());
    EXPECT_EQ(25, geo.surfaces().size());
    EXPECT_EQ(3, geo.num_univ_levels());
    EXPECT_FALSE(geo.supports_safety());

    EXPECT_VEC_SOFT_EQ(Real3({-2, -6, -1}), geo.bbox().lower());
    EXPECT_VEC_SOFT_EQ(Real3({8, 4, 2}), geo.bbox().upper());

    std::vector<std::string> expected = {
        "[EXTERIOR]",
        "inner_a",
        "inner_b",
        "bobby",
        "johnny",
        "[EXTERIOR]",
        "inner_c",
        "a",
        "b",
        "c",
        "[EXTERIOR]",
        "patty",
    };
    std::vector<std::string> actual;
    for (auto const id : range(ImplVolumeId{geo.impl_volumes().size()}))
    {
        actual.push_back(geo.impl_volumes().at(id).name);
    }

    EXPECT_VEC_EQ(expected, actual);
}

TEST_F(UniversesTest, tracking)
{
    {
        SCOPED_TRACE("patty");
        auto result = this->track({-1.0, -3.75, 0.75}, {1, 0, 0});
        static char const* const expected_volumes[]
            = {"johnny", "patty", "c", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1, 0.5, 5.5, 2};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0.25, 0.25, 0.25, 0.25};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("inner +x");
        auto result = this->track({-1, -2, 1.0}, {1, 0, 0});
        static char const* const expected_volumes[]
            = {"johnny", "c", "a", "b", "c", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1, 1, 2, 2, 1, 2};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {0.5, 0, 0.5, 0.5, 0.5, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("inner +y");
        auto result = this->track({4, -5, 1.0}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"johnny", "c", "b", "c", "bobby", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1, 1, 2, 1, 2, 2};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {0.5, 0, 0.5, 0.5, 0.5, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("inner +z");
        auto result = this->track({4, -2, -0.75}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"johnny", "b", "b", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {0.25, 1, 1, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0.125, 0.5, 0.5, 0.25};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

TEST_F(UniversesTest, TEST_IF_CELERITAS_DOUBLE(output))
{
    OrangeParamsOutput out(this->geometry());
    EXPECT_EQ("orange", out.label());

    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"orange","scalars":{"num_univ_levels":3,"max_faces":14,"max_intersections":14,"max_csg_levels":1,"tol":{"abs":1.5e-08,"rel":1.5e-08}},"sizes":{"bih":{"bboxes":12,"inner_nodes":6,"leaf_nodes":9,"local_volume_ids":10},"connectivity_records":25,"daughters":3,"fast_real3s":0,"local_surface_ids":55,"local_volume_ids":21,"logic_ints":164,"obz_records":0,"real_ids":25,"reals":24,"rect_arrays":0,"simple_units":3,"surface_types":25,"transforms":3,"universe_indexer":{"surfaces":4,"volumes":4},"univ_indices":3,"univ_types":3,"volume_ids":12,"volume_instance_ids":12,"volume_records":12}})json",
        to_string(out));
}

TEST_F(UniversesTest, initialize_with_multiple_universes)
{
    auto geo = this->make_geo_track_view();

    // Initialize in outermost universe
    geo = Initializer_t{{-1, -2, 1}, {1, 0, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({-1, -2, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ("johnny", this->volume_name(geo));
    EXPECT_FALSE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

    // Initialize in daughter universe
    geo = Initializer_t{{0.625, -2, 1}, {1, 0, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({0.625, -2, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ("c", this->volume_name(geo));
    EXPECT_FALSE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

    // Initialize in daughter universe using "this == &other"
    geo = Initializer_t{geo.pos(), {0, 1, 0}, TrackSlotId{0}};
    EXPECT_VEC_SOFT_EQ(Real3({0.625, -2, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.dir());
    EXPECT_EQ("c", this->volume_name(geo));
    EXPECT_FALSE(geo.is_outside());
    EXPECT_FALSE(geo.is_on_boundary());

    {
        // Initialize a separate track slot
        auto other = this->make_geo_track_view(TrackSlotId{1});
        other = Initializer_t{geo.pos(), {1, 0, 0}, TrackSlotId{0}};
        EXPECT_VEC_SOFT_EQ(Real3({0.625, -2, 1}), other.pos());
        EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), other.dir());
        EXPECT_EQ(
            "c", this->params().impl_volumes().at(other.impl_volume_id()).name);
        EXPECT_FALSE(other.is_outside());
        EXPECT_FALSE(other.is_on_boundary());

        EXPECT_JSON_EQ(
            R"json({"levels":[{"dir":[0.0,1.0,0.0],"pos":[0.625,-2.0,1.0],"universe":"outer","volume":{"impl":"inner_b@outer","local":2}},{"dir":[0.0,1.0,0.0],"pos":[-1.375,0.0,0.5],"universe":"inner","volume":{"impl":"c@inner","local":4}}],"surface":null})json",
            to_json_string(geo.track_view()));
        EXPECT_JSON_EQ(
            R"json({"levels":[{"dir":[1.0,0.0,0.0],"pos":[0.625,-2.0,1.0],"universe":"outer","volume":{"impl":"inner_b@outer","local":2}},{"dir":[1.0,0.0,0.0],"pos":[-1.375,0.0,0.5],"universe":"inner","volume":{"impl":"c@inner","local":4}}],"surface":null})json",
            to_json_string(other.track_view()));
    }
}

TEST_F(UniversesTest, move_internal_multiple_universes)
{
    auto geo = this->make_geo_track_view();

    // Initialize in daughter universe
    geo = Initializer_t{{0.625, -2, 1}, {0, 1, 0}};

    // Move internally, then check that the distance to boundary is correct
    geo.move_internal({0.625, -1, 1});
    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    // Move again, using other move_internal method
    geo.move_internal(0.1);
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.9, next.distance);
}

// Set direction in a daughter universe and then make sure the direction is
// correctly returned at the top level
TEST_F(UniversesTest, change_dir_daughter_universe)
{
    auto geo = this->make_geo_track_view();

    // Initialize inside daughter universe a
    geo = Initializer_t{{1.5, -2.0, 1.0}, {1.0, 0.0, 0.0}};

    // Change the direction
    geo.set_dir({0.0, 1.0, 0.0});

    // Get the direction
    EXPECT_VEC_EQ(Real3({0.0, 1.0, 0.0}), geo.dir());
}

// Cross into daughter universe for the case where the hole cell does not share
// a boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_daughter_non_coincident)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{2, -5, 0.75}, {0, 1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("johnny", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 0.75}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("c", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 0.75}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, -1, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("alpha.my", this->surface_name(geo));
}

// Cross into parent universe for the case where the hole cell does not share a
// boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_parent_non_coincident)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{2, -3.25, 0.75}, {0, -1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.75, next.distance);
    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("c", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 0.75}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, -1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("johnny", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, -4, 0.75}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("john.my", this->surface_name(geo));
}

// Cross into daughter universe for the case where the hole cell shares a
// boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_daughter_coincident)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{2, 1, 1}, {0, -1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.my", this->surface_name(geo));
    EXPECT_EQ("bobby", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, -1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("bob.my", this->surface_name(geo));
    EXPECT_EQ("c", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("alpha.py", this->surface_name(geo));
}

// Cross into parent universe for the case where the hole cell shares a
// boundary with another with a parent cell
TEST_F(UniversesTest, cross_into_parent_coincident)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{2, -0.5, 1}, {0, 1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.my", this->surface_name(geo));
    EXPECT_EQ("c", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("bob.my", this->surface_name(geo));
    EXPECT_EQ("bobby", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, 0, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, -1, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.py", this->surface_name(geo));
}

// Cross into daughter universe that is two levels down
TEST_F(UniversesTest, cross_into_daughter_doubly_coincident)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{0.25, -4.5, 1}, {0, 1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("johnny", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("patty", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, -1, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_c.py", this->surface_name(geo));
}

// Cross into parent universe that is two levels down
TEST_F(UniversesTest, cross_into_parent_doubly_coincident)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{0.25, -3.75, 1}, {0, -1, 0}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.25, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("patty", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, -1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("johnny", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("john.my", this->surface_name(geo));
}

// Cross between two daughter universes that share a boundary
TEST_F(UniversesTest, cross_between_daughters)
{
    auto geo = this->make_geo_track_view();

    // Initialize in outermost universe
    geo = Initializer_t{{2, -2, 0.7}, {0, 0, -1}};

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.2, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("inner_a.pz", this->surface_name(geo));
    EXPECT_EQ("a", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, -2, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, -1}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_a.pz", this->surface_name(geo));
    EXPECT_EQ("a", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, -2, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, 1}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after crossing
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("bob.mz", this->surface_name(geo));
}

// Change direction on a universe boundary to reenter the cell
TEST_F(UniversesTest, reentrant)
{
    auto geo = this->make_geo_track_view();

    // Initialize in innermost universe
    geo = Initializer_t{{0.25, -3.7, 0.7}, {0, 1, 0}};
    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.2, next.distance);

    // Move to universe boundary
    geo.move_to_boundary();
    EXPECT_EQ("inner_c.py", this->surface_name(geo));
    EXPECT_EQ("patty", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -3.5, 0.7}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Change direction on the universe boundary such that we are no longer
    // exiting the universe
    geo.set_dir({0, -1, 0});

    // Remain in same cell after crossing boundary
    geo.cross_boundary();
    EXPECT_EQ("inner_c.py", this->surface_name(geo));
    EXPECT_EQ("patty", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -3.5, 0.7}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, 1, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    // Make sure we can take another step after calling cross_boundary
    next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);
    geo.move_to_boundary();
    EXPECT_EQ("inner_a.my", this->surface_name(geo));
    EXPECT_EQ("patty", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({0.25, -4, 0.7}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({0, -1, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);
}

//---------------------------------------------------------------------------//
class RectArrayTest : public JsonOrangeTest
{
    std::string_view geometry_basename() const final { return "rect-array"; }
};

TEST_F(RectArrayTest, params)
{
    OrangeParams const& geo = this->params();
    EXPECT_EQ(35, geo.impl_volumes().size());
    EXPECT_EQ(22, geo.surfaces().size());
    EXPECT_EQ(4, geo.num_univ_levels());
    EXPECT_FALSE(geo.supports_safety());

    EXPECT_VEC_SOFT_EQ(Real3({-12, -4, -5}), geo.bbox().lower());
    EXPECT_VEC_SOFT_EQ(Real3({12, 10, 5}), geo.bbox().upper());
}

TEST_F(RectArrayTest, tracking)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{-1, 1, -1}, {1, 0, 0}};

    EXPECT_VEC_SOFT_EQ(Real3({-1, 1, -1}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ("Hfill", this->volume_name(geo));
}

//---------------------------------------------------------------------------//

class NestedRectArraysTest : public JsonOrangeTest
{
    std::string_view geometry_basename() const final
    {
        return "nested-rect-arrays";
    }
};

TEST_F(NestedRectArraysTest, tracking)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{1.5, 0.5, 0.5}, {1, 0, 0}};

    EXPECT_VEC_SOFT_EQ(Real3({1.5, 0.5, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ("Afill", this->volume_name(geo));

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("{x,1}", this->surface_name(geo));
    EXPECT_EQ("Afill", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, 0.5, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("{x,1}", this->surface_name(geo));
    EXPECT_EQ("Bfill", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({2, 0.5, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({-1, 0, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    next = geo.find_next_step();
    EXPECT_SOFT_EQ(1, next.distance);
}

TEST_F(NestedRectArraysTest, leaving)
{
    auto geo = this->make_geo_track_view();
    geo = Initializer_t{{3.5, 1.5, 0.5}, {1, 0, 0}};

    EXPECT_JSON_EQ(
        R"json({"levels":[{"dir":[1.0,0.0,0.0],"pos":[3.5,1.5,0.5],"universe":"global","volume":{"impl":"arrfill@global","local":1}},{"dir":[1.0,0.0,0.0],"pos":[3.5,1.5,0.5],"universe":"parent","volume":{"impl":"parent+@parent","local":1}},{"dir":[1.0,0.0,0.0],"pos":[3.5,1.5,0.5],"universe":"parent+","volume":{"impl":"{1,0,0}@parent+","local":2}},{"dir":[1.0,0.0,0.0],"pos":[1.5,1.5,0.5],"universe":"arr","volume":{"impl":"arr+@arr","local":1}},{"dir":[1.0,0.0,0.0],"pos":[1.5,1.5,0.5],"universe":"arr+","volume":{"impl":"{1,1,0}@arr+","local":3}},{"dir":[1.0,0.0,0.0],"pos":[0.5,0.5,0.5],"universe":"B","volume":{"impl":"Bfill@B","local":1}}],"surface":null})json",
        StringSimplifier{3}(to_json_string(geo.track_view())));

    EXPECT_VEC_SOFT_EQ(Real3({3.5, 1.5, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ("Bfill", this->volume_name(geo));

    auto next = geo.find_next_step();
    EXPECT_SOFT_EQ(0.5, next.distance);

    geo.move_to_boundary();
    EXPECT_EQ("arrfill.px", this->surface_name(geo));
    EXPECT_EQ("Bfill", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({4, 1.5, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.normal());
    EXPECT_GT(dot_product(geo.normal(), geo.dir()), 0);

    // Cross universe boundary
    geo.cross_boundary();
    EXPECT_EQ("arrfill.px", this->surface_name(geo));
    EXPECT_EQ("interior", this->volume_name(geo));
    EXPECT_VEC_SOFT_EQ(Real3({4, 1.5, 0.5}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({-1, 0, 0}), geo.normal());
    EXPECT_LT(dot_product(geo.normal(), geo.dir()), 0);

    next = geo.find_next_step();
    EXPECT_SOFT_EQ(16, next.distance);
}

//---------------------------------------------------------------------------//
class Geant4Testem15Test : public JsonOrangeTest
{
    std::string_view geometry_basename() const final
    {
        return "geant4-testem15";
    }
};

TEST_F(Geant4Testem15Test, safety)
{
    auto geo = this->make_geo_track_view();

    geo = Initializer_t{{0, 0, 0}, {1, 0, 0}};
    EXPECT_VEC_SOFT_EQ(Real3({0, 0, 0}), geo.pos());
    EXPECT_VEC_SOFT_EQ(Real3({1, 0, 0}), geo.dir());
    EXPECT_EQ(ImplVolumeId{1}, geo.impl_volume_id());
    EXPECT_FALSE(geo.is_on_boundary());
    EXPECT_FALSE(geo.is_outside());

    // Safety at middle should be to the box boundary
    EXPECT_SOFT_EQ(5000.0, geo.find_safety());

    // Check safety near face
    auto next = geo.find_next_step(4995.0);
    EXPECT_SOFT_EQ(4995.0, next.distance);
    EXPECT_FALSE(next.boundary);
    geo.move_internal(4995.0);
    EXPECT_SOFT_EQ(5.0, geo.find_safety());

    // Check safety near edge
    geo.set_dir({0, 1, 0});
    next = geo.find_next_step();
    geo.move_internal(4990.0);
    EXPECT_SOFT_EQ(5.0, geo.find_safety());
    geo.set_dir({-1, 0, 0});
    next = geo.find_next_step();
    geo.move_internal(6.0);
    EXPECT_SOFT_EQ(10.0, geo.find_safety());
}

//---------------------------------------------------------------------------//

class HexArrayTest : public JsonOrangeTest
{
    std::string_view geometry_basename() const final { return "hex-array"; }
};

TEST_F(HexArrayTest, TEST_IF_CELERITAS_DOUBLE(output))
{
    OrangeParamsOutput out(this->geometry());
    EXPECT_EQ("orange", out.label());

    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"orange","scalars":{"num_univ_levels":3,"max_faces":9,"max_intersections":10,"max_csg_levels":1,"tol":{"abs":1.5e-08,"rel":1.5e-08}},"sizes":{"bih":{"bboxes":58,"inner_nodes":49,"leaf_nodes":53,"local_volume_ids":55},"connectivity_records":53,"daughters":51,"fast_real3s":0,"local_surface_ids":191,"local_volume_ids":348,"logic_ints":500,"obz_records":0,"real_ids":53,"reals":272,"rect_arrays":0,"simple_units":4,"surface_types":53,"transforms":51,"universe_indexer":{"surfaces":5,"volumes":5},"univ_indices":4,"univ_types":4,"volume_ids":58,"volume_instance_ids":58,"volume_records":58}})json",
        to_string(out));
}

TEST_F(HexArrayTest, track_out)
{
    auto result = this->track(
        {-6.9258369494022292, -4.9982766629573767, -10.8378536157757495},
        {0.6750034206933703, -0.3679917428721818, 0.6394939086732125});

    static char const* const expected_volumes[]
        = {"interior", "cfill", "dfill", "interior"};
    EXPECT_VEC_EQ(expected_volumes, result.volumes);
    static real_type const expected_distances[] = {
        1.9914318088046, 5.3060674310398, 0.30636846908014, 5.9880767678838};
    EXPECT_VEC_NEAR(
        expected_distances, result.distances, 10 * SoftEqual<>{}.rel());
    static real_type const expected_hw_safety[] = {
        0.20109936014143, 0.29549138370648, 0.030952132652541, 0.90113367054536};
    EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
}

//---------------------------------------------------------------------------//

class TestEM3Test : public JsonOrangeTest
{
    std::string_view geometry_basename() const final { return "testem3"; }
};

// Test safety distance within a geometry that supports simple safety
TEST_F(TestEM3Test, safety)
{
    EXPECT_FALSE(this->params().supports_safety());

    auto geo = this->make_geo_track_view();

    // Initialize in innermost universe, near the universe boundary
    geo = Initializer_t{{19.99, 19.9, 19.9}, {0, 1, 0}};
    EXPECT_SOFT_EQ(0.01, geo.find_safety());

    // Initialize on the other side of the same volume
    geo = Initializer_t{{19.42, 19.9, 19.9}, {0, 1, 0}};
    EXPECT_SOFT_EQ(0.01, geo.find_safety());
}

//---------------------------------------------------------------------------//

TEST_F(InputBuilderTest, globalspheres)
{
    {
        auto result = this->track({0, 0, 0}, {0, 0, 1});

        static char const* const expected_volumes[] = {"inner", "shell"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {5, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {2.5, 2.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }

    OrangeParamsOutput out(this->geometry());
    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"orange","scalars":{"num_univ_levels":1,"max_faces":2,"max_intersections":4,"max_csg_levels":1,"tol":{"abs":1e-05,"rel":1e-05}},"sizes":{"bih":{"bboxes":3,"inner_nodes":0,"leaf_nodes":1,"local_volume_ids":3},"connectivity_records":2,"daughters":0,"fast_real3s":0,"local_surface_ids":4,"local_volume_ids":4,"logic_ints":7,"obz_records":0,"real_ids":2,"reals":2,"rect_arrays":0,"simple_units":1,"surface_types":2,"transforms":0,"universe_indexer":{"surfaces":2,"volumes":2},"univ_indices":1,"univ_types":1,"volume_ids":3,"volume_instance_ids":3,"volume_records":3}})json",
        to_string(out));
}

//---------------------------------------------------------------------------//

TEST_F(InputBuilderTest, lar_split_detector)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    {
        auto result = this->track({0, 0, -14}, {0, 0, 1});

        GenericGeoTrackingResult ref;
        ref.volumes = {
            "outer_region",
            "lower_shell",
            "inner",
            "upper_shell",
            "outer_region",
        };
        ref.volume_instances = {
            "outer_region@global",
            "lower_shell@global",
            "inner@global",
            "upper_shell@global",
            "outer_region@global",
        };
        ref.distances = {4, 5, 10, 5, 5};
        ref.halfway_safeties = {2, 2.5, inf, 2.5, 2.5};
        ref.bumps = {};
        auto tol = this->tracking_tol();
        EXPECT_REF_NEAR(ref, result, tol);
    }

    {
        SCOPED_TRACE("Initialize on irrelevant surface");
        auto geo = this->make_geo_track_view();
        EXPECT_NO_THROW((geo = Initializer_t{{1, 2, 0}, {0, 0, 1}}));
    }
}

//---------------------------------------------------------------------------//
TEST_F(InputBuilderTest, bgspheres)
{
    {
        SCOPED_TRACE("from background");
        auto result = this->track({0, 0, -9}, {0, 0, 1});

        static char const* const expected_volumes[]
            = {"global", "bottom", "global", "top", "global"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {3, 6, 1, 4, 5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("from inside top");
        auto result = this->track({0, 0, 3}, {0, 0, -1});

        static char const* const expected_volumes[]
            = {"top", "global", "bottom", "global"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {2, 1, 6, 4};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }

    OrangeParamsOutput out(this->geometry());
    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"orange","scalars":{"num_univ_levels":1,"max_faces":3,"max_intersections":2,"max_csg_levels":1,"tol":{"abs":1e-05,"rel":1e-05}},"sizes":{"bih":{"bboxes":4,"inner_nodes":1,"leaf_nodes":2,"local_volume_ids":3},"connectivity_records":3,"daughters":0,"fast_real3s":0,"local_surface_ids":6,"local_volume_ids":3,"logic_ints":5,"obz_records":0,"real_ids":3,"reals":9,"rect_arrays":0,"simple_units":1,"surface_types":3,"transforms":0,"universe_indexer":{"surfaces":2,"volumes":2},"univ_indices":1,"univ_types":1,"volume_ids":4,"volume_instance_ids":4,"volume_records":4}})json",
        to_string(out));
}

//---------------------------------------------------------------------------//
TEST_F(InputBuilderTest, universes)
{
    // NOTE: results should be identical to UniversesTest.tracking
    {
        SCOPED_TRACE("patty");
        auto result = this->track({-1.0, -3.75, 0.75}, {1, 0, 0});
        static char const* const expected_volumes[]
            = {"johnny", "patty", "c", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1, 0.5, 5.5, 2};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0.25, 0.25, 0.25, 0.25};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("inner +x");
        auto result = this->track({-1, -2, 1.0}, {1, 0, 0});
        static char const* const expected_volumes[]
            = {"johnny", "c", "a", "b", "c", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1, 1, 2, 2, 1, 2};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {0.5, 0, 0.5, 0.5, 0.5, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("inner +y");
        auto result = this->track({4, -5, 1.0}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"johnny", "c", "b", "c", "bobby", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {1, 1, 2, 1, 2, 2};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[]
            = {0.5, 0, 0.5, 0.5, 0.5, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("inner +z");
        auto result = this->track({4, -2, -0.75}, {0, 0, 1});
        static char const* const expected_volumes[]
            = {"johnny", "b", "b", "johnny"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {0.25, 1, 1, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {0.125, 0.5, 0.5, 0.25};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
}

//---------------------------------------------------------------------------//
TEST_F(InputBuilderTest, hierarchy)
{
    // FIXME: normal is inconsistent on transformed boundaries!
    supports_surface_normal_ = false;

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        auto geo = this->make_geo_track_view();
        geo = Initializer_t{{0, -5, -20}, {0, 1, 0}};

        EXPECT_JSON_EQ(
            R"json({"levels":[
{"dir":[0.0,1.0,0.0],"pos":[0.0,-5.0,-20.0],"universe":"global","volume":{"impl":"fd@global","local":3}},
{"dir":[0.0,1.0,0.0],"pos":[0.0,-5.0,0.0],"universe":"filled_daughter","volume":{"impl":"e@filled_daughter","local":2}},
{"dir":[0.0,0.0,-1.0],"pos":[0.0,0.0,0.0],"universe":"d2","volume":{"impl":"d2@bg","local":1}}
],"surface":null})json",
            StringSimplifier{3}(to_json_string(geo.track_view())));
    }
    {
        SCOPED_TRACE("py");
        auto result = this->track({0, -20, 0}, {0, 1, 0});
        static char const* const expected_volumes[]
            = {"interior", "d2", "interior", "d1", "interior"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {14, 2, 8, 2, 94};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("py_filled");
        auto result = this->track({0, -9, -20}, {0, 1, 0});
        static char const* const expected_volumes[] = {
            "filled_daughter",
            "d2",
            "filled_daughter",
            "d1",
            "filled_daughter",
            "interior",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {3, 2, 8, 2, 4, 87.979589711327};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        static real_type const expected_hw_safety[] = {1.5, 5, 4, 5, 2, 39};
        EXPECT_VEC_SOFT_EQ(expected_hw_safety, result.halfway_safeties);
    }
    {
        SCOPED_TRACE("pz");
        auto result = this->track({0, 0, -50}, {0, 0, 1});
        static char const* const expected_volumes[] = {
            "interior",
            "filled_daughter",
            "leaf1",
            "filled_daughter",
            "leaf2",
            "filled_daughter",
            "interior",
            "leaf1",
            "interior",
            "bottom",
            "top",
            "interior",
        };
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[]
            = {20, 4, 2, 8, 2, 4, 4, 2, 23, 1, 1, 79};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }

    OrangeParamsOutput out(this->geometry());
    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"orange","scalars":{"num_univ_levels":3,"max_faces":8,"max_intersections":14,"max_csg_levels":1,"tol":{"abs":1e-05,"rel":1e-05}},"sizes":{"bih":{"bboxes":24,"inner_nodes":9,"leaf_nodes":16,"local_volume_ids":13},"connectivity_records":13,"daughters":6,"fast_real3s":0,"local_surface_ids":20,"local_volume_ids":18,"logic_ints":31,"obz_records":0,"real_ids":13,"reals":46,"rect_arrays":0,"simple_units":7,"surface_types":13,"transforms":6,"universe_indexer":{"surfaces":8,"volumes":8},"univ_indices":7,"univ_types":7,"volume_ids":24,"volume_instance_ids":24,"volume_records":24}})json",
        to_string(out));
}

//---------------------------------------------------------------------------//
TEST_F(InputBuilderTest, incomplete_bb)
{
    OrangeParamsOutput out(this->geometry());
    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"orange","scalars":{"num_univ_levels":2,"max_faces":6,"max_intersections":6,"max_csg_levels":1,"tol":{"abs":1e-05,"rel":1e-05}},"sizes":{"bih":{"bboxes":6,"inner_nodes":1,"leaf_nodes":3,"local_volume_ids":5},"connectivity_records":8,"daughters":1,"fast_real3s":0,"local_surface_ids":10,"local_volume_ids":4,"logic_ints":37,"obz_records":0,"real_ids":8,"reals":26,"rect_arrays":0,"simple_units":2,"surface_types":8,"transforms":1,"universe_indexer":{"surfaces":3,"volumes":3},"univ_indices":2,"univ_types":2,"volume_ids":6,"volume_instance_ids":6,"volume_records":6}})json",
        to_string(out));
}

//---------------------------------------------------------------------------//
// TODO: see celeritas-project/celeritas#1260
TEST_F(InputBuilderTest, DISABLED_universe_union_boundary)
{
    {
        SCOPED_TRACE("pz");
        auto result = this->track({0, 0, -15}, {0, 0, 1});

        static char const* const expected_volumes[]
            = {"shell", "bottomsph", "bite", "shell"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {11.234, 10, 4, 9.766};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
    {
        SCOPED_TRACE("mz");
        auto result = this->track({0, 0, 15}, {0, 0, -1});
        result.print_expected();
    }
}

//---------------------------------------------------------------------------//
// TODO: see celeritas-project/celeritas#1342
TEST_F(InputBuilderTest, DISABLED_involute)
{
    {
        SCOPED_TRACE("involute");
        auto result = this->track({0, 2.1, 0}, {1, 0, 0});
        static char const* const expected_volumes[]
            = {"channel", "blade", "rest", "shell"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            static real_type const expected_distances[] = {
                0.531630657850489,
                0.660884355302089,
                2.21189389295726,
                1.13321161570553,
            };
            EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        }
        else
        {
            static real_type const expected_distances[]
                = {0.531625, 0.66089, 2.21189389295726, 1.13321161570553};
            EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        }
    }

    {
        SCOPED_TRACE("involute");
        auto result = this->track({0, 3.5, 0}, {0, -1, 0});
        static char const* const expected_volumes[]
            = {"rest", "blade", "channel", "center", "rest", "shell"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            0.528306129329604, 0.530097149547556, 0.441596721122841, 4, 2, 1};
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

TEST_F(InputBuilderTest, DISABLED_involute_cw)
{
    {
        SCOPED_TRACE("involute");
        auto result = this->track({-2, -1.5, 0}, {1, 0, 0});
        static char const* const expected_volumes[]
            = {"rest", "center", "rest", "blade", "rest", "shell"};
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
        static real_type const expected_distances[] = {
            0.6771243444677,
            2.6457513110646,
            0.17135844958615,
            0.50955324797963,
            1.7043118904498,
            1.0615967635369,
        };
        EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
    }
}

TEST_F(InputBuilderTest, DISABLED_involute_fuel)
{
    {
        SCOPED_TRACE("involute");
        auto result = this->track({1.75, 3.5, 0}, {0, -1, 0});
        static char const* const expected_volumes[] = {
            "clad2",
            "fuel2",
            "clad2",
            "rest2",
            "middle",
            "rest1",
            "clad1",
            "fuel1",
            "clad1",
            "rest1",
            "middle",
            "rest2",
            "shell",
        };
        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            static real_type const expected_distances[] = {
                0.12694500489541,
                0.16591646127344,
                0.46300647647806,
                0.30743347115084,
                0.65134147906653,
                2.1115149489926,
                0.30217303181663,
                0.70489828892381,
                0.39023902667286,
                0.06188891786551,
                0.65134147906653,
                1.1601750562823,
                1.0868748563143,
            };
            EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        }
        else
        {
            static real_type const expected_distances[] = {
                0.12694500489541,
                0.16591646127344,
                0.46300647647806,
                0.30743347115084,
                0.65134147906653,
                2.1115149489926,
                0.30217303181663,
                0.70489828892381,
                0.390229,
                0.0618988,
                0.65134147906653,
                1.1601750562823,
                1.0868748563143,
            };
            EXPECT_VEC_SOFT_EQ(expected_distances, result.distances);
        }
        EXPECT_VEC_EQ(expected_volumes, result.volumes);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
