//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Raytracer.test.cc
//---------------------------------------------------------------------------//
#include "geocel/rasterize/Raytracer.hh"

#include <cmath>

#include "geocel/MockGeoTrackView.hh"
#include "geocel/rasterize/Image.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
int calc_id(MockGeoTrackView const& geo)
{
    if (geo.is_outside())
        return -1;
    return static_cast<int>(geo.pos()[2]);
}

//---------------------------------------------------------------------------//
TEST(MockGeo, tracking)
{
    MockGeoTrackView geo;
    geo = GeoTrackInitializer{{0, 0, 5.25}, {0, 0, 1}};
    EXPECT_EQ(5, geo.impl_volume_id().get());
    auto next_step = geo.find_next_step(2.0);
    EXPECT_TRUE(next_step.boundary);
    EXPECT_SOFT_EQ(0.75, next_step.distance);
    geo.move_to_boundary();
    EXPECT_EQ(5, geo.impl_volume_id().get());
    EXPECT_REAL_EQ(6.0, geo.pos()[2]);
    geo.cross_boundary();
    EXPECT_EQ(6, geo.impl_volume_id().get());
    next_step = geo.find_next_step(1.5);
    EXPECT_TRUE(next_step.boundary);
    EXPECT_SOFT_EQ(1.0, next_step.distance);
    geo.move_to_boundary();
    EXPECT_EQ(6, geo.impl_volume_id().get());
    geo.cross_boundary();
    EXPECT_EQ(7, geo.impl_volume_id().get());
    next_step = geo.find_next_step(0.5);
    EXPECT_FALSE(next_step.boundary);
    EXPECT_SOFT_EQ(0.5, next_step.distance);
    geo.move_internal(0.25);
    next_step = geo.find_next_step(1.0);
    EXPECT_TRUE(next_step.boundary);
    EXPECT_SOFT_EQ(0.75, next_step.distance);

    geo = GeoTrackInitializer{{0, 0, 4.75}, make_unit_vector(Real3{0, 0, -1})};
    EXPECT_EQ(4, geo.impl_volume_id().get());
    next_step = geo.find_next_step(100.0);
    EXPECT_TRUE(next_step.boundary);
    EXPECT_SOFT_EQ(0.75, next_step.distance);

    geo = GeoTrackInitializer{{0, 0, 9.75}, make_unit_vector(Real3{0, 4, -3})};
    EXPECT_EQ(9, geo.impl_volume_id().get());
    next_step = geo.find_next_step(100.0);
    EXPECT_TRUE(next_step.boundary);
    EXPECT_SOFT_EQ(0.75 * 5.0 / 3.0, next_step.distance);
}

//---------------------------------------------------------------------------//

class RaytracerTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(RaytracerTest, exact)
{
    ImageInput inp;
    inp.rightward = {0, 0, 1};
    inp.lower_left = {0, 0, 32};
    inp.upper_right = {0, 16, 160};
    inp.vertical_pixels = 16;
    inp.horizontal_divisor = 1;

    // Start at {0, 10.5, 32} and move along +z
    auto params = std::make_shared<ImageParams>(inp);
    Image<MemSpace::host> img(params);
    ImageLineView line{params->host_ref(), img.ref(), 5};

    MockGeoTrackView geo;
    EXPECT_EQ(0, geo.init_count());

    Raytracer trace{geo, calc_id, line};
    // Position doesn't change until the next pixel
    EXPECT_EQ(32, trace(0));
    EXPECT_EQ(1, geo.init_count());
    EXPECT_VEC_SOFT_EQ((Real3{0, 10.5, 32}), geo.pos());

    EXPECT_EQ(33, trace(1));
    EXPECT_VEC_SOFT_EQ((Real3{0, 10.5, 33}), geo.pos());
    EXPECT_EQ(1, geo.init_count());

    EXPECT_EQ(40, trace(8));
    EXPECT_EQ(2, geo.init_count());
}

TEST_F(RaytracerTest, offset)
{
    ImageInput inp;
    inp.rightward = {0, 0, 1};
    inp.lower_left = {0, 0, 1.75};
    inp.upper_right = {0, 1.0, 16.75};
    inp.vertical_pixels = 1;
    inp.horizontal_divisor = 1;

    auto params = std::make_shared<ImageParams>(inp);
    Image<MemSpace::host> img(params);
    ImageLineView line{params->host_ref(), img.ref(), 0};

    MockGeoTrackView geo;
    Raytracer trace{geo, calc_id, line};
    for (auto col : range(10))
    {
        EXPECT_EQ(col + 2, trace(col));
    }
    EXPECT_EQ(1, geo.init_count());
}

TEST_F(RaytracerTest, megapixels)
{
    ImageInput inp;
    inp.rightward = {0, 0, 1};
    inp.lower_left = {0, 0, 1};
    inp.upper_right = {0, 64, 64 * 8};
    inp.vertical_pixels = 1;
    inp.horizontal_divisor = 1;

    auto params = std::make_shared<ImageParams>(inp);
    Image<MemSpace::host> img(params);
    ImageLineView line{params->host_ref(), img.ref(), 0};

    MockGeoTrackView geo;
    Raytracer trace{geo, calc_id, line};
    EXPECT_EQ(1, trace(0));
    EXPECT_EQ(1, geo.init_count());
    EXPECT_EQ(65, trace(1));
    EXPECT_EQ(2, geo.init_count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
