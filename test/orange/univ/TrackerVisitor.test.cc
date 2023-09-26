//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/TrackerVisitor.test.cc
//---------------------------------------------------------------------------//

#include "orange/univ/TrackerVisitor.hh"

#include "orange/OrangeGeoTestBase.hh"
#include "orange/univ/detail/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class TrackerVisitorTest : public OrangeGeoTestBase
{
  protected:
    using LocalState = detail::LocalState;

    void SetUp() override { this->build_geometry("rect_array.org.json"); }

    LocalState make_state(Real3 pos, Real3 dir);
};

//---------------------------------------------------------------------------//
// TEST FIXTURE IMPLEMENTATION
//---------------------------------------------------------------------------//
/*!
 * Initialize without any logical state.
 */
detail::LocalState TrackerVisitorTest::make_state(Real3 pos, Real3 dir)
{
    normalize_direction(&dir);
    detail::LocalState state;
    state.pos = pos;
    state.dir = dir;
    state.volume = {};
    state.surface = {};

    auto const& hsref = this->host_state();
    auto face_storage = hsref.temp_face[AllItems<FaceId>{}];
    state.temp_sense = hsref.temp_sense[AllItems<Sense>{}];
    state.temp_next.face = face_storage.data();
    state.temp_next.distance
        = hsref.temp_distance[AllItems<real_type>{}].data();
    state.temp_next.isect = hsref.temp_isect[AllItems<size_type>{}].data();
    state.temp_next.size = face_storage.size();
    return state;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackerVisitorTest, initialize)
{
    TrackerVisitor visit_tracker{this->host_params()};

    auto local = this->make_state({0.5, 0.5, 0.5}, {1, 0, 0});

    auto init_functor = [&local](auto&& t) { return t.initialize(local); };

    auto init_simple = visit_tracker(init_functor, UniverseId{0});
    auto init_rect = visit_tracker(init_functor, UniverseId{2});
    auto init_simple2 = visit_tracker(init_functor, UniverseId{3});

    EXPECT_EQ("arrfill", this->id_to_label(UniverseId{0}, init_simple.volume));
    EXPECT_EQ("{0,0,0}", this->id_to_label(UniverseId{2}, init_rect.volume));
    EXPECT_EQ("Tfill", this->id_to_label(UniverseId{3}, init_simple2.volume));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
