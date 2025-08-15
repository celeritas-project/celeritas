//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Truncated.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/Truncated.hh"

#include "corecel/Assert.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "orange/orangeinp/IntersectRegion.hh"
#include "orange/orangeinp/Shape.hh"

#include "CsgTestUtils.hh"
#include "ObjectTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//

class TruncatedTest : public ObjectTestBase
{
  protected:
    using Plane = InfPlane;

    void SetUp() override {}
    Tol tolerance() const override { return Tol::from_relative(1e-4); }
};

TEST_F(TruncatedTest, errors)
{
    // No planes
    EXPECT_THROW(Truncated("el", std::make_unique<Sphere>(1.0), {}),
                 RuntimeError);

    // Redundant truncating planes
    EXPECT_THROW(Truncated("el",
                           std::make_unique<Sphere>(1.0),
                           {Plane{Sense::inside, Axis::z, 1.25},
                            Plane{Sense::inside, Axis::z, 0.25}}),
                 RuntimeError);
}

TEST_F(TruncatedTest, ellipsoid)
{
    Real3 radii{1.5, 0.5, 2.0};
    this->build_volume(Truncated("el",
                                 std::make_unique<Ellipsoid>(radii),
                                 {Plane{Sense::inside, Axis::z, 1.25},
                                  Plane{Sense::outside, Axis::z, -0.5}}));

    static char const* const expected_surface_strings[]
        = {"SQuadric: {0.44444,4,0.25} {0,0,0} -1",
           "Plane: z=1.25",
           "Plane: z=-0.5"};
    static char const* const expected_md_strings[] = {
        "",
        "",
        "el@interior.sq",
        "el@interior",
        "el@trunc.pz",
        "",
        "el@trunc.mz",
        "el@trunc",
        "el",
    };
    static char const* const expected_volume_strings[] = {"all(-0, -1, +2)"};

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
}

TEST_F(TruncatedTest, or_shape)
{
    Real3 radii{1, 2, 3};

    TypeDemangler<ObjectInterface> demangle_shape;
    {
        auto shape = Truncated::or_shape("el", Ellipsoid{radii}, {});
        EXPECT_TRUE(shape);
        EXPECT_TRUE(dynamic_cast<EllipsoidShape const*>(shape.get()))
            << "actual shape: " << demangle_shape(*shape);
    }
    {
        auto trunc = Truncated::or_shape(
            "el", Ellipsoid{radii}, {InfPlane{Sense::inside, Axis::x, 1.25}});
        EXPECT_TRUE(trunc);
        EXPECT_TRUE(dynamic_cast<Truncated const*>(trunc.get()))
            << demangle_shape(*trunc);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
