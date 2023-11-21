//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SoftSurfaceEqual.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SoftSurfaceEqual.hh"

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/surf/detail/AllSurfaces.hh"
#include "orange/surf/detail/SurfaceTranslator.hh"

#include "celeritas_test.hh"

using celeritas::detail::SurfaceTranslator;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class SoftSurfaceEqualTest : public ::celeritas::test::Test
{
  protected:
    static constexpr real_type small = 1e-5;  // small < eps
    static constexpr real_type eps = 1e-4;
    static constexpr real_type large = 1e-3;  // eps < large < sqrt(eps)
    SoftSurfaceEqual softeq_{eps};

    //! Check surfaces with a sphere-like constructor
    template<class S>
    void
    check_equality_s(Real3 const& pt, real_type r, Axis skip = Axis::size_) const
    {
        auto ref = S(pt, r);
        EXPECT_TRUE(softeq_(ref, S(pt + eps / 4 * norm(pt), r)));
        if (skip != Axis::z)
        {
            EXPECT_TRUE(softeq_(ref, S(pt + Real3{0, 0, small}, r)));
        }
        EXPECT_TRUE(softeq_(ref, S(pt, r - small)));
        EXPECT_TRUE(softeq_(ref, S(pt, r + small)));
        if (skip != Axis::x)
        {
            EXPECT_FALSE(softeq_(ref, S(pt + Real3{large, 0, 0}, r)));
        }
        EXPECT_FALSE(softeq_(ref, S(pt, r - large)));
        EXPECT_FALSE(softeq_(ref, S(pt, r + large)));
    }
};

constexpr real_type SoftSurfaceEqualTest::small;
constexpr real_type SoftSurfaceEqualTest::eps;
constexpr real_type SoftSurfaceEqualTest::large;

TEST_F(SoftSurfaceEqualTest, plane_aligned)
{
    EXPECT_TRUE(softeq_(PlaneX{4.0}, PlaneX{4.0 - small}));
    EXPECT_FALSE(softeq_(PlaneX{4.0}, PlaneX{4.0 + large}));
}

TEST_F(SoftSurfaceEqualTest, cyl_centered)
{
    EXPECT_TRUE(softeq_(CCylX{2.0}, CCylX{2.0 + small}));
    EXPECT_FALSE(softeq_(CCylX{2.0}, CCylX{2.0 + large}));
}

TEST_F(SoftSurfaceEqualTest, sphere_centered)
{
    EXPECT_TRUE(softeq_(SphereCentered{10}, SphereCentered{10 - 10 * small}));
    EXPECT_FALSE(softeq_(SphereCentered{10}, SphereCentered{10 + 10 * large}));
}

TEST_F(SoftSurfaceEqualTest, cyl_aligned)
{
    this->check_equality_s<CylX>({1, 2, 3}, 0.5, Axis::x);
    this->check_equality_s<CylY>({1, 2, 3}, 0.5, Axis::y);
    this->check_equality_s<CylZ>({1, 2, 3}, 0.5, Axis::z);
}

TEST_F(SoftSurfaceEqualTest, plane)
{
    Real3 const p{1, 0, 0};
    Real3 const n = make_unit_vector(Real3{1, 1, 0});
    Plane const ref{n, p};

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_TRUE(softeq_(ref, Plane{n, p + Real3{small, 0, 0}}));
    }
    EXPECT_FALSE(softeq_(ref, Plane{n, p + Real3{large, 0, 0}}));

    Real3 const npert = make_unit_vector(n + Real3{small, 0, 0});
    EXPECT_TRUE(softeq_(ref, Plane{npert, p}));

    Real3 const ndiff = make_unit_vector(n + Real3{0, large, 0});
    EXPECT_FALSE(softeq_(ref, Plane{ndiff, p}));
    EXPECT_FALSE(softeq_(ref, Plane{make_unit_vector(Real3{-1, 1, 0}), p}));
    EXPECT_FALSE(softeq_(ref, Plane{make_unit_vector(Real3{1, -1, 0}), p}));
}

TEST_F(SoftSurfaceEqualTest, sphere)
{
    this->check_equality_s<Sphere>({0, 1, 2}, 1);
    this->check_equality_s<Sphere>({-0.4, 0.6, 0.5}, 0.9);
}

TEST_F(SoftSurfaceEqualTest, cone_aligned)
{
    this->check_equality_s<ConeX>({1, -1, 0}, 0.7);
    this->check_equality_s<ConeY>({1, -1, 0}, 0.7);
    this->check_equality_s<ConeZ>({1, -1, 0}, 0.7);
}

TEST_F(SoftSurfaceEqualTest, simple_quadric)
{
    auto ellipsoid = [](Real3 const& radii) {
        const Real3 second{ipow<2>(radii[1]) * ipow<2>(radii[2]),
            ipow<2>(radii[2]) * ipow<2>(radii[0]),
            ipow<2>(radii[0]) * ipow<2>(radii[1])};
        const real_type zeroth = -ipow<2>(radii[0]) * ipow<2>(radii[1])
            * ipow<2>(radii[2]);
        return SimpleQuadric{second, Real3{0, 0, 0}, zeroth};
    };
    auto translated = [](auto&& s, Real3 const& center) {
            SurfaceTranslator translate{Translation{center}};
            return translate(s);
    };

    {
        SCOPED_TRACE("ellipsoid");
        Real3 const origin{0, 0, 0};
        Real3 const radii{1, 2.5, .3};
        SimpleQuadric const ref = ellipsoid(radii);
        // Perturb a single dimension
        EXPECT_TRUE(softeq_(ref, ellipsoid({1 + small, 2.5, .3 - small})));
        EXPECT_TRUE(softeq_(ref, ellipsoid({1 + small, 2.5 + small, .3})));
        EXPECT_FALSE(softeq_(ref, ellipsoid({1 + large, 2.5, .3 - large})));
        EXPECT_FALSE(softeq_(ref, ellipsoid({1 + large, 2.5 + large, .3})));

        // Translate and scale
        EXPECT_TRUE(softeq_(ref, translated(ref, {0, small/2, 0})));
        EXPECT_TRUE(softeq_(
            ref, translated(ellipsoid(radii * (1 + small)), origin)));
        EXPECT_FALSE(softeq_(ref, translated(ref, {0, 0, large})));
        EXPECT_FALSE(
            softeq_(ref, translated(ellipsoid(radii * (1 + large)), origin)));
    }
    {
        Real3 const origin{10, 0, 0};
        Real3 const radii{1, 2.5, 0.75};
        auto ref = translated(ellipsoid(radii), origin);

        EXPECT_TRUE(softeq_(ref, translated(ref, {0, small/2, 0})));
        EXPECT_TRUE(softeq_(
            ref, translated(ellipsoid(radii * (1 + small)), origin)));
        EXPECT_FALSE(softeq_(ref, translated(ref, {0, 0, large})));
        EXPECT_FALSE(
            softeq_(ref, translated(ellipsoid(radii * (1 + large)), origin)));
    }
}

TEST_F(SoftSurfaceEqualTest, general_quadric)
{
    GeneralQuadric ref{{10.3125, 22.9375, 15.75},
                       {-21.867141445557, -20.25, 11.69134295109},
                       {-11.964745962156, -9.1328585544429, -65.69134295109},
                       77.652245962156};

    EXPECT_TRUE(
        softeq_(ref, SurfaceTranslator(Translation{{small, 0, small}})(ref)));
    EXPECT_FALSE(
        softeq_(ref, SurfaceTranslator(Translation{{large, 0, 0}})(ref)));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
