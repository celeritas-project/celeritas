//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/ImageWriter.test.cc
//---------------------------------------------------------------------------//
#include "geocel/rasterize/ImageWriter.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
constexpr std::size_t test_img_width{29};
constexpr std::size_t test_img_height{15};
char const test_img[] = R"txt(
+---------------------------+
|     e               e     |
|    e                 e    |
|   e                   e   |
|  e  aaaaaa     aaaaaa  e  |
|     aaaabb     aaaabb     |
|     aaaabb     aaaabb     |
|                           |
|              c            |
|           cccc            |
|                           |
|   dddddddddddddddddddddd  |
|   deeeeeeeeeeeeeeeeeeeed  |
|   dddddddddddddddddddddd  |
+---------------------------+
)txt";

Color get_color(char c)
{
    // clang-format off
    switch (c)
    {
        case ' ': return Color{};            // black transparent
        case '+': return Color{0xff000080u}; // half-opacity red
        case '-': return Color{0x00ff0080u}; // half-opacity green
        case '|': return Color{0x0000ff80u}; // half-opacity blue
        case 'a': return Color{0xeeeeeeeeu}; // translucent near-white
        case 'b': return Color{0x000000ffu}; // black
        case 'c': return Color{0x10c030ffu}; // greenish
        case 'd': return Color{0xe00830ffu}; // reddish
        case 'e': return Color{0x303030ffu}; // gray
        default: CELER_ASSERT_UNREACHABLE();
    }
    // clang-format on
}

using VecVecColor = std::vector<std::vector<Color>>;

class ImageWriterTest : public ::celeritas::test::Test
{
  protected:
    static void SetUpTestCase()
    {
        for (char c : test_img)
        {
            if (c == '\n')
            {
                lines_.push_back({});
            }
            else if (c == '\0') {}
            else
            {
                CELER_ASSERT(!lines_.empty());
                lines_.back().push_back(get_color(c));
            }
        }
        ASSERT_EQ(test_img_height + 1, lines_.size());
        CELER_ASSERT(lines_.back().empty());
        lines_.pop_back();

        for (auto const& line : lines_)
        {
            ASSERT_EQ(test_img_width, line.size());
        }
    }

    static Span<Color const> img_line(std::size_t i)
    {
        CELER_EXPECT(i < lines_.size());
        auto result = make_span(lines_[i]);
        CELER_ENSURE(result.size() == test_img_width);
        return result;
    }

  private:
    static VecVecColor lines_;
};

VecVecColor ImageWriterTest::lines_;

//---------------------------------------------------------------------------//

TEST_F(ImageWriterTest, write)
{
    ImageWriter write_line(this->make_unique_filename(".png"),
                           {2 * test_img_height, test_img_width});
    for (auto i : range(test_img_height))
    {
        // Write the same line twice to get better proportions
        write_line(img_line(i));
        write_line(img_line(i));
    }
}

TEST_F(ImageWriterTest, bad_open)
{
    EXPECT_THROW(ImageWriter("/celeritas/doesnt-exist/bad.png", {1, 2}),
                 RuntimeError);
    EXPECT_THROW(ImageWriter(this->make_unique_filename(".png"), {0, 10}),
                 RuntimeError);
    EXPECT_THROW(ImageWriter(this->make_unique_filename(".png"), {10, 0}),
                 RuntimeError);
}

TEST_F(ImageWriterTest, write_incomplete)
{
    ImageWriter write_line(this->make_unique_filename(".png"),
                           {2 * test_img_height, test_img_width});
    EXPECT_TRUE(write_line);
    for (auto i : range(test_img_height))
    {
        write_line(img_line(i));
    }

    ScopedLogStorer scoped_log_{&celeritas::world_logger()};
    EXPECT_THROW(write_line.close(), RuntimeError);
    EXPECT_FALSE(write_line);

    static char const* const expected_log_messages[] = {
        "PNG file received only 15 of 30 lines", "No IDATs written into file"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"error", "error"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
