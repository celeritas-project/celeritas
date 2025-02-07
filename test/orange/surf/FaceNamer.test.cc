//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/FaceNamer.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/FaceNamer.hh"

#include "corecel/math/ArrayUtils.hh"
#include "orange/surf/VariantSurface.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class FaceNamerTest : public ::celeritas::test::Test
{
  protected:
    static constexpr auto in = Sense::inside;
    static constexpr auto out = Sense::outside;
};

TEST_F(FaceNamerTest, typed)
{
    FaceNamer name_face;

    EXPECT_EQ("px", name_face(in, PlaneX(1)));
    EXPECT_EQ("mx", name_face(out, PlaneX(1)));
    EXPECT_EQ("px", name_face(in, PlaneX(2)));

    EXPECT_EQ("p0", name_face(in, Plane(make_unit_vector(Real3{1, 2, 3}), 1)));
    EXPECT_EQ("p1", name_face(in, Plane(make_unit_vector(Real3{1, 2, 3}), -1)));

    EXPECT_EQ("cx", name_face(in, CCylX(5)));
    EXPECT_EQ("cy", name_face(in, CCylY(6)));
    EXPECT_EQ("cz", name_face(in, CCylZ(7)));
    EXPECT_EQ("s", name_face(in, SphereCentered(1.0)));
    EXPECT_EQ("cx", name_face(in, CylX({1, 2, 3}, 0.5)));
    EXPECT_EQ("cy", name_face(in, CylY({1, 2, 3}, 0.6)));
    EXPECT_EQ("cz", name_face(in, CylZ({1, 2, 3}, 0.7)));
    EXPECT_EQ("s", name_face(in, Sphere({1, 2, 3}, 1.5)));
    EXPECT_EQ("kx", name_face(in, ConeX({1, 2, 3}, 0.2)));
    EXPECT_EQ("ky", name_face(in, ConeY({1, 2, 3}, 0.4)));
    EXPECT_EQ("kz", name_face(in, ConeZ({1, 2, 3}, 0.6)));
    EXPECT_EQ("sq", name_face(in, SimpleQuadric({0, 1, 2}, {6, 7, 8}, 9)));
    EXPECT_EQ(
        "gq",
        name_face(in, GeneralQuadric({0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 9)));
}

TEST_F(FaceNamerTest, prefix)
{
    FaceNamer name_face{"obox"};

    EXPECT_EQ("obox.px", name_face(in, PlaneX(1)));
    EXPECT_EQ("obox.mx", name_face(out, PlaneX(1)));
    EXPECT_EQ("obox.px", name_face(in, PlaneX(2)));
}

TEST_F(FaceNamerTest, variant)
{
    VariantSurface cyl{std::in_place_type_t<CCylY>(), 6};

    EXPECT_EQ("cy", FaceNamer{}(in, cyl));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
