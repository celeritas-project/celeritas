//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipper.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SurfaceClipper.hh"

#include "orange/BoundingBoxUtils.hh"
#include "orange/surf/CylAligned.hh"
#include "orange/surf/CylCentered.hh"
#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SphereCentered.hh"
#include "orange/surf/VariantSurface.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct ClipResult
{
    BBox i = BBox::from_infinite();
    BBox x = BBox::from_infinite();

    void print_expected() const;
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
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n";
#define CR_PRINT(BOX, BOUND)                                       \
    cout << "EXPECT_VEC_SOFT_EQ((Real3" << repr(this->BOX.BOUND()) \
         << "), result." #BOX "." #BOUND "());\n"
    CR_PRINT(i, lower);
    CR_PRINT(i, upper);
    CR_PRINT(x, lower);
    CR_PRINT(x, upper);
    cout << "/*** END CODE ***/\n";
}

TEST_F(SurfaceClipperTest, inside)
{
    auto inf = std::numeric_limits<real_type>::infinity();

    ClipResult cr;
    cr = this->test_clip(PlaneX{4});
    EXPECT_VEC_SOFT_EQ((Real3{-inf, -inf, -inf}), cr.i.lower());
    EXPECT_VEC_SOFT_EQ((Real3{4, inf, inf}), cr.i.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-inf, -inf, -inf}), cr.x.lower());
    EXPECT_VEC_SOFT_EQ((Real3{4, inf, inf}), cr.x.upper());
    cr = this->test_clip(CCylZ{3});
    EXPECT_VEC_SOFT_EQ((Real3{-2.1213203435596, -2.1213203435596, -inf}),
                       cr.i.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2.1213203435596, 2.1213203435596, inf}),
                       cr.i.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-3, -3, -inf}), cr.x.lower());
    EXPECT_VEC_SOFT_EQ((Real3{3, 3, inf}), cr.x.upper());
    cr = this->test_clip(SphereCentered{3});
    EXPECT_VEC_SOFT_EQ(
        (Real3{-2.5980762113533, -2.5980762113533, -2.5980762113533}),
        cr.i.lower());
    EXPECT_VEC_SOFT_EQ(
        (Real3{2.5980762113533, 2.5980762113533, 2.5980762113533}),
        cr.i.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-3, -3, -3}), cr.x.lower());
    EXPECT_VEC_SOFT_EQ((Real3{3, 3, 3}), cr.x.upper());
    cr = this->test_clip(CylZ{{1, 2, 3}, 0.25});
    EXPECT_VEC_SOFT_EQ((Real3{0.82322330470336, 1.8232233047034, -inf}),
                       cr.i.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.1767766952966, 2.1767766952966, inf}),
                       cr.i.upper());
    EXPECT_VEC_SOFT_EQ((Real3{0.75, 1.75, -inf}), cr.x.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.25, 2.25, inf}), cr.x.upper());
    cr = this->test_clip(Sphere{{1, 2, 3}, 0.25});
    EXPECT_VEC_SOFT_EQ(
        (Real3{0.78349364905389, 1.7834936490539, 2.7834936490539}),
        cr.i.lower());
    EXPECT_VEC_SOFT_EQ(
        (Real3{1.2165063509461, 2.2165063509461, 3.2165063509461}),
        cr.i.upper());
    EXPECT_VEC_SOFT_EQ((Real3{0.75, 1.75, 2.75}), cr.x.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.25, 2.25, 3.25}), cr.x.upper());

    // Test with variant
    VariantSurface v{CCylY{4}};
    cr = this->test_clip(v);
    EXPECT_VEC_SOFT_EQ((Real3{-2.8284271247462, -inf, -2.8284271247462}),
                       cr.i.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2.8284271247462, inf, 2.8284271247462}),
                       cr.i.upper());
    EXPECT_VEC_SOFT_EQ((Real3{-4, -inf, -4}), cr.x.lower());
    EXPECT_VEC_SOFT_EQ((Real3{4, inf, 4}), cr.x.upper());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
