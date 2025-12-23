//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/RaytraceImager.test.cc
//---------------------------------------------------------------------------//
#include "orange/RaytraceImager.hh"

#include "corecel/cont/Span.hh"
#include "geocel/rasterize/Image.hh"

#include "OrangeGeoTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
std::string to_ascii(ImageInterface const& image, Span<char const> id_to_char)
{
    ImageParams const& params = *image.params();

    // Allocate destination space
    std::vector<int> pixels(params.num_pixels(), -1);
    std::string result(params.num_pixels() + 2 * params.num_lines() + 1, ' ');

    // Copy image to our array
    image.copy_to_host(make_span(pixels));

    // Convert image to pixels
    auto iter = result.begin();
    *iter++ = '\n';
    int num_cols = params.scalars().dims[1];
    for (int vol_id : pixels)
    {
        *iter++ = [&] {
            if (vol_id < 0)
                return ' ';
            if (vol_id < static_cast<int>(id_to_char.size()))
                return id_to_char[vol_id];
            return 'x';
        }();

        CELER_ASSERT(iter != result.end());
        if (--num_cols == 0)
        {
            *iter++ = '|';
            CELER_ASSERT(iter != result.end());
            *iter++ = '\n';
            num_cols = params.scalars().dims[1];
        }
    }
    CELER_ASSERT(iter == result.end());
    return result;
}

//---------------------------------------------------------------------------//
struct TwoVolumeTest
{
    //! Name of the test
    static inline char const* name = "two_volume";
    //! Geo params arguments
    static auto geometry_input()
    {
        return OrangeGeoTestBase::TwoVolInput{1.0};
    }
    //! Image params input
    static ImageInput image_input()
    {
        ImageInput result;
        result.lower_left = {0, 0, 0};
        result.upper_right = {1, 1, 0};
        result.rightward = {1, 0, 0};
        result.vertical_pixels = 8;
        result.horizontal_divisor = 1;
        return result;
    }
    //! Mapping of volume ID to character
    static constexpr char const id_to_char[] = {' ', 'x'};
    //! Expected image
    static constexpr char const expected_image[] = R"(
xxx     |
xxxxx   |
xxxxxx  |
xxxxxxx |
xxxxxxx |
xxxxxxxx|
xxxxxxxx|
xxxxxxxx|
)";
};

//---------------------------------------------------------------------------//
struct TwoVolumeTestBackward : public TwoVolumeTest
{
    static inline char const* name = "two_volume backward";
    static ImageInput image_input()
    {
        ImageInput result;
        result.lower_left = {1, 0, 0};
        result.upper_right = {0, 1, 0};
        result.rightward = {-1, 0, 0};
        result.vertical_pixels = 8;
        result.horizontal_divisor = 1;
        return result;
    }
    // NOTE: image is different because the trace starts outside the geometry
    static constexpr char const expected_image[] = R"(
      xx|
    xxxx|
   xxxxx|
  xxxxxx|
 xxxxxxx|
 xxxxxxx|
 xxxxxxx|
 xxxxxxx|
)";
};

//---------------------------------------------------------------------------//
struct UniversesTest
{
    //! Name of the test
    static inline char const* name = "universes";
    //! Geo params arguments
    static auto geometry_input() { return "universes.org.json"; }
    //! Image params input
    static ImageInput image_input()
    {
        // NOTE: due to axis-aligned boundaries, we can't start the raytrace on
        // the outer box
        ImageInput result;
        result.lower_left = {-1.9, -5.9, 0.25};
        result.upper_right = {8.1, 4.1, 0.25};
        result.rightward = {1, 0, 0};
        result.vertical_pixels = 20;
        result.horizontal_divisor = 2;
        return result;
    }
    /*!
     * Mapping of volume ID to character.
     *
     * - ` `: `[EXTERIOR]` volume (should not appear since ID should be -1 when
     *   outside)
     * - `-`: volume replaced by child universe (should never be "final"
     *   volume)
     * - `abc`: universes named as such
     * - `BJP`: bobby, johnny, patty
     */
    static constexpr char const id_to_char[] = " --BJ -abc P";
    //! Expected image
    static constexpr char const expected_image[] = R"(
JJJJJJJJJJJJJJJJJJJJ|
JJJJJJJJJJJJJJJJJJJJ|
JJJJJJJJJJJJJJJJJJJJ|
JJJJJJJJJJJJJJJJJJJJ|
JJJJBBBBBBBBBBBBJJJJ|
JJJJBBBBBBBBBBBBJJJJ|
JJJJBBBBBBBBBBBBJJJJ|
JJJJBBBBBBBBBBBBJJJJ|
JJJJccccccccccccJJJJ|
JJJJccccccccccccJJJJ|
JJJJccaaaabbbbccJJJJ|
JJJJccaaaabbbbccJJJJ|
JJJJccaaaabbbbccJJJJ|
JJJJccaaaabbbbccJJJJ|
JJJJccccccccccccJJJJ|
JJJJPcccccccccccJJJJ|
JJJJJJJJJJJJJJJJJJJJ|
JJJJJJJJJJJJJJJJJJJJ|
JJJJJJJJJJJJJJJJJJJJ|
JJJJJJJJJJJJJJJJJJJJ|
)";
};

//---------------------------------------------------------------------------//
template<class P>
class RaytraceImagerTest : public OrangeGeoTestBase
{
  protected:
    //! Length scale is hardcoded into JSON input
    UnitLength unit_length() const override { return {Constant{1}, "length"}; }

    void SetUp() override
    {
        this->build_geometry(P::geometry_input());
        img_params_ = std::make_shared<ImageParams>(P::image_input());
    }

    ImageParams const& img_params() const { return *img_params_; }

    template<MemSpace M>
    void test_image() const;

  private:
    std::shared_ptr<ImageParams> img_params_;
};

template<class P>
template<MemSpace M>
void RaytraceImagerTest<P>::test_image() const
{
    RaytraceImager raytrace_image{this->geometry()};

    Image<M> image(img_params_);

    // Raytrace
    raytrace_image(&image);
    auto actual = to_ascii(image, make_span(P::id_to_char));
    EXPECT_EQ(P::expected_image, actual)
        << "static constexpr char const expected_image[] = R\"(" << actual
        << ")\";\n";

    // Raytrace again, hopefully reusing geometry cache
    raytrace_image(&image);
    auto again = to_ascii(image, make_span(P::id_to_char));
    EXPECT_EQ(actual, again);
}

//---------------------------------------------------------------------------//
using RaytraceTypes
    = ::testing::Types<TwoVolumeTest, TwoVolumeTestBackward, UniversesTest>;

struct TestToString
{
    template<class U>
    static std::string GetName(int)
    {
        return U::name;
    }
};

TYPED_TEST_SUITE(RaytraceImagerTest, RaytraceTypes, TestToString);

TYPED_TEST(RaytraceImagerTest, host)
{
    this->template test_image<MemSpace::host>();
}

#if CELER_USE_DEVICE
TYPED_TEST(RaytraceImagerTest, device)
#else
TYPED_TEST(RaytraceImagerTest, DISABLED_device)
#endif
{
    this->template test_image<MemSpace::device>();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
