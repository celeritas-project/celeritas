//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Image.test.cc
//---------------------------------------------------------------------------//
#include "geocel/rasterize/Image.hh"

#include "geocel/rasterize/ImageLineView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class ImageTest : public ::celeritas::test::Test
{
  protected:
    using Size2 = ImageParamsScalars::Size2;

    void SetUp() override {}
};

TEST_F(ImageTest, errors)
{
    ImageInput inp;
    inp.upper_right = {1, 1, 0};
    EXPECT_THROW(ImageParams{inp}, RuntimeError);
    inp.vertical_pixels = 128;
    inp.upper_right = inp.lower_left;
    EXPECT_THROW(ImageParams{inp}, RuntimeError);
    inp.upper_right = {1, 1, 0};
    inp.horizontal_divisor = 0;
    EXPECT_THROW(ImageParams{inp}, RuntimeError);
}

TEST_F(ImageTest, exact)
{
    ImageInput inp;
    inp.upper_right = {128, 32, 0};
    inp.vertical_pixels = 16;
    inp.horizontal_divisor = 16;

    auto params = std::make_shared<ImageParams>(inp);
    auto const& scalars = params->scalars();
    EXPECT_VEC_SOFT_EQ((Real3{0, 32, 0}), scalars.origin);
    EXPECT_VEC_SOFT_EQ((Real3{0, -1, 0}), scalars.down);
    EXPECT_SOFT_EQ(2, scalars.pixel_width);
    EXPECT_VEC_EQ((Size2{16, 16 * 4}), scalars.dims);
    EXPECT_SOFT_EQ(128, scalars.max_length);

    Image<MemSpace::host> img(params);
    {
        ImageLineView line{params->host_ref(), img.ref(), TrackSlotId{0}};
        EXPECT_EQ(64, line.max_index());
        line.set_pixel(0, 123);
        line.set_pixel(2, 345);
        EXPECT_VEC_SOFT_EQ((Real3{0, 31, 0}), line.start_pos());
    }
    {
        ImageLineView line{params->host_ref(), img.ref(), TrackSlotId{1}};
        line.set_pixel(1, 567);
        EXPECT_VEC_SOFT_EQ((Real3{0, 29, 0}), line.start_pos());
    }

    std::vector<int> result(params->num_pixels());
    img.copy_to_host(make_span(result));
    EXPECT_EQ(123, result[0]);
    EXPECT_EQ(-1, result[1]);
    EXPECT_EQ(345, result[2]);
    EXPECT_EQ(567, result[65]);
}

TEST_F(ImageTest, inexact)
{
    ImageInput inp;
    inp.lower_left = {-1, 2, -1};
    inp.upper_right = {6, 2, 4};
    inp.vertical_pixels = 256;
    inp.rightward = {0, 0, 1};  // +z
    inp.horizontal_divisor = 32;

    auto params = std::make_shared<ImageParams>(inp);
    auto const& scalars = params->scalars();
    EXPECT_VEC_SOFT_EQ((Real3{6, 2, -1}), scalars.origin);
    EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), scalars.down);
    EXPECT_SOFT_EQ(real_type{7} / 256, scalars.pixel_width);
    EXPECT_VEC_EQ((Size2{256, 192}), scalars.dims);
    EXPECT_SOFT_EQ(real_type{5}, scalars.max_length);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
