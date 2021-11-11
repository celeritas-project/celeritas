//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CellInitializer.test.cc
//---------------------------------------------------------------------------//
#include "orange/universes/detail/CellInitializer.hh"

#include "orange/surfaces/Surfaces.hh"
#include "orange/universes/VolumeView.hh"

// Test includes
#include "celeritas_test.hh"
#include "orange/OrangeGeoTestBase.hh"

using celeritas::detail::CellInitializer;
using celeritas::detail::FoundFace;
using celeritas::detail::LocalState;
using celeritas::detail::OnSurface;
using namespace celeritas;

//---------------------------------------------------------------------------//
// DETAIL TESTS
//---------------------------------------------------------------------------//

TEST(Types, FoundFace)
{
    // Not found
    FoundFace not_found;
    EXPECT_FALSE(not_found);
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(not_found.face(), celeritas::DebugError);
    }
    EXPECT_NO_THROW(not_found.unchecked_face());

    // Found, not on a face
    FoundFace found{true};
    EXPECT_TRUE(found);
    EXPECT_FALSE(found.face());
    EXPECT_FALSE(found.unchecked_face());

    // Found, on a face
    FoundFace found_face{true, {FaceId{1}, Sense::outside}};
    EXPECT_TRUE(found_face);
    EXPECT_EQ(FaceId{1}, found_face.face().id());
    EXPECT_EQ(FaceId{1}, found_face.unchecked_face().id());
    EXPECT_EQ(Sense::outside, found_face.unchecked_face().sense());
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CellInitializerTest : public celeritas_test::OrangeGeoTestBase
{
  protected:
    using VolumeDataRef = VolumeView::VolumeDataRef;

    LocalState make_state(Real3 pos, OnSurface surface = {})
    {
        LocalState state;
        state.pos         = pos;
        state.dir         = {0, 0, 1}; // Doesn't matter to initializer
        state.cell        = {};
        state.surface     = surface;
        state.temp_senses = this->sense_storage();
        return state;
    }

    Surfaces make_surfaces() const
    {
        return Surfaces{this->params_host_ref().surfaces};
    }

    const VolumeDataRef& volume_ref() const
    {
        return this->params_host_ref().volumes;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CellInitializerTest, one_volume)
{
    // Construct geometry
    OneVolInput geo_inp;
    this->build_geometry(geo_inp);

    {
        // Test an arbitrary point
        CellInitializer try_init(this->make_surfaces(),
                                 this->make_state(Real3{123, 345, 567}));

        // Test volume
        auto result = try_init(VolumeView(this->volume_ref(), VolumeId{0}));
        EXPECT_TRUE(result);
        EXPECT_FALSE(result.face());
    }
}

TEST_F(CellInitializerTest, two_volumes)
{
    // Construct geometry
    TwoVolInput geo_inp;
    geo_inp.radius = 1.5;
    this->build_geometry(geo_inp);

    // Make volume accessors
    VolumeView        inner(this->volume_ref(), VolumeId{0});
    VolumeView        outer(this->volume_ref(), VolumeId{1});
    detail::FoundFace result;

    {
        // Test point is in the inner sphere
        CellInitializer try_init(this->make_surfaces(),
                                 this->make_state(Real3{0, 0.5, 0}));

        // Test sphere
        result = try_init(inner);
        EXPECT_TRUE(result);
        EXPECT_FALSE(result.face());

        // Test not-sphere
        result = try_init(outer);
        EXPECT_FALSE(result);
    }
    {
        // Point is in on the boundary, inside the face
        CellInitializer try_init(
            this->make_surfaces(),
            this->make_state(Real3{1.5, 0, 0}, {SurfaceId{0}, Sense::inside}));

        result = try_init(inner);
        EXPECT_TRUE(result);
        EXPECT_EQ(FaceId{0}, result.face().id());
        EXPECT_EQ(Sense::inside, result.face().sense());
    }
    {
        // Point is in on the boundary, face state unknown:
        // - It ends up inside the *outer* volume since `SignedSense::on ->
        // Sense::outside`
        // - Only one volume is allowed to claim the point in any valid
        // geometry!
        CellInitializer try_init(this->make_surfaces(),
                                 this->make_state(Real3{1.5, 0, 0}));

        result = try_init(inner);
        EXPECT_FALSE(result);

        result = try_init(outer);
        EXPECT_TRUE(result);
        EXPECT_EQ(FaceId{0}, result.face().id());
        EXPECT_EQ(Sense::outside, result.face().sense());
    }
    {
        // Point is in on the boundary, known to be outside
        CellInitializer try_init(
            this->make_surfaces(),
            this->make_state(Real3{1.5, 0, 0}, {SurfaceId{0}, Sense::outside}));

        result = try_init(outer);
        EXPECT_TRUE(result);
        EXPECT_EQ(FaceId{0}, result.face().id());
        EXPECT_EQ(Sense::outside, result.face().sense());
    }
    {
        // Point is outside
        CellInitializer try_init(this->make_surfaces(),
                                 this->make_state(Real3{2.5, 0, 0}));

        result = try_init(outer);
        EXPECT_TRUE(result);
        EXPECT_FALSE(result.face());
    }
}

TEST_F(CellInitializerTest, five_volumes)
{
    if (!CELERITAS_USE_JSON)
    {
        GTEST_SKIP() << "JSON is not enabled";
    }

    this->build_geometry("five-volumes.org.json");

    // Volume definitions
    VolumeView vol_c(this->volume_ref(), VolumeId{3});

    // Point is between spheres, on square edge
    CellInitializer try_init(this->make_surfaces(),
                             this->make_state(Real3{0.5, -0.25, 0},
                                              {SurfaceId{5}, Sense::outside}));

    auto result = try_init(vol_c);
    EXPECT_TRUE(result);
    EXPECT_FALSE(result.face());
}
