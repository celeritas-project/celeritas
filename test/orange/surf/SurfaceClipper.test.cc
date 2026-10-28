//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipper.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SurfaceClipper.hh"

#include "geocel/RefUtils.hh"
#include "orange/surf/CylAligned.hh"
#include "orange/surf/CylCentered.hh"
#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SphereCentered.hh"
#include "orange/surf/VariantSurface.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct ClipResult
{
    static constexpr char class_str[] = "ClipResult";

    BBox i = BBox::from_infinite();
    BBox x = BBox::from_infinite();

    void print_expected() const;

    friend std::ostream& operator<<(std::ostream& os, ClipResult const& result)
    {
        return os << '{' << result.i << ", " << result.x << '}';
    }

    friend void check_ref_eq(AssertionHelper& helper,
                             ClipResult const& expected,
                             ClipResult const& actual,
                             std::nullptr_t)
    {
        is_ref_eq(helper, "i", expected.i, actual.i, nullptr);
        is_ref_eq(helper, "x", expected.x, actual.x, nullptr);
    }
};

class SurfaceClipperTest : public ::celeritas::test::Test
{
  protected:
    template<class S>
    ClipResult test_clip(S const& surf)
    {
        ClipResult cr;
        SurfaceClipper clip{&cr.i, &cr.x};
        clip(surf);
        return cr;
    }
};

void ClipResult::print_expected() const
{
    ::celeritas::testdetail::print_expected(*this);
}

TEST_F(SurfaceClipperTest, inside)
{
    auto inf = std::numeric_limits<real_type>::infinity();

    ClipResult cr;
    ClipResult ref;
    cr = this->test_clip(PlaneX{4});
    print_expected(cr);
    ref.i = BBox{Real3{-inf, -inf, -inf}, Real3{4, inf, inf}};
    ref.x = BBox{Real3{-inf, -inf, -inf}, Real3{4, inf, inf}};
    EXPECT_REF_EQ(ref, cr);

    cr = this->test_clip(CCylZ{3});
    ref.i = BBox{Real3{-2.1213203435596, -2.1213203435596, -inf},
                 Real3{2.1213203435596, 2.1213203435596, inf}};
    ref.x = BBox{Real3{-3, -3, -inf}, Real3{3, 3, inf}};
    EXPECT_REF_EQ(ref, cr);

    cr = this->test_clip(SphereCentered{3});
    ref.i = BBox{Real3{-2.5980762113533, -2.5980762113533, -2.5980762113533},
                 Real3{2.5980762113533, 2.5980762113533, 2.5980762113533}};
    ref.x = BBox{Real3{-3, -3, -3}, Real3{3, 3, 3}};
    EXPECT_REF_EQ(ref, cr);

    cr = this->test_clip(CylZ{{1, 2, 3}, 0.25});
    ref.i = BBox{Real3{0.82322330470336, 1.8232233047034, -inf},
                 Real3{1.1767766952966, 2.1767766952966, inf}};
    ref.x = BBox{Real3{0.75, 1.75, -inf}, Real3{1.25, 2.25, inf}};
    EXPECT_REF_EQ(ref, cr);

    cr = this->test_clip(Sphere{{1, 2, 3}, 0.25});
    ref.i = BBox{Real3{0.78349364905389, 1.7834936490539, 2.7834936490539},
                 Real3{1.2165063509461, 2.2165063509461, 3.2165063509461}};
    ref.x = BBox{Real3{0.75, 1.75, 2.75}, Real3{1.25, 2.25, 3.25}};
    EXPECT_REF_EQ(ref, cr);

    // Test with variant
    VariantSurface v{CCylY{4}};
    cr = this->test_clip(v);
    ref.i = BBox{Real3{-2.8284271247462, -inf, -2.8284271247462},
                 Real3{2.8284271247462, inf, 2.8284271247462}};
    ref.x = BBox{Real3{-4, -inf, -4}, Real3{4, inf, 4}};
    EXPECT_REF_EQ(ref, cr);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
