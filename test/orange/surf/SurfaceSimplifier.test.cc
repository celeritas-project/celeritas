//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceSimplifier.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SurfaceSimplifier.hh"

#include <iomanip>

#include "corecel/Constants.hh"
#include "corecel/cont/ArrayIO.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/surf/detail/AllSurfaces.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using constants::sqrt_three;
using constants::sqrt_two;

//---------------------------------------------------------------------------//
//! Return the string form of a surface
struct ToString
{
    template<class S>
    std::string operator()(S const& surf) const
    {
        std::ostringstream os;
        os << std::setprecision(13) << surf;
        return os.str();
    }

    std::string operator()(std::monostate) const { return "{unsimplified}"; }
};

//---------------------------------------------------------------------------//

class SurfaceSimplifierTest : public ::celeritas::test::Test
{
  protected:
    //! Promote from S to T, then simplify
    template<class T, class S>
    void check_round_trip(S const& s)
    {
        SCOPED_TRACE(s);

        // Promote to more general class
        T promoted{s};

        // Simplify to a variant
        this->sense = Sense::inside;
        auto result = this->simplify(promoted);

        if (S const* s2 = std::get_if<S>(&result))
        {
            // The resulting surface is the same type: check data
            EXPECT_VEC_SOFT_EQ(s.data(), s2->data());
        }
        else
        {
            ADD_FAILURE() << "Actual: " << std::visit(ToString{}, result);
        }

        // Sense should not have changed
        EXPECT_EQ(Sense::inside, this->sense);
    }

    //! Check that a surface simplification doesn't alter the state
    template<class S>
    void check_unchanged(S const& s)
    {
        SCOPED_TRACE(s);

        // Simplify to a variant
        auto result = this->simplify(s);

        if (!std::holds_alternative<std::monostate>(result))
        {
            ADD_FAILURE() << "Actual: " << std::visit(ToString{}, result);
        }
    }

    //! Check that one surface simplifies to another
    template<class S, class T>
    void
    check_simplifies_to(S const& s, T const& t, Sense new_sense = Sense::inside)
    {
        SCOPED_TRACE(s);

        // Simplify to a variant
        this->sense = Sense::inside;
        auto result = this->simplify(s);

        if (T const* s2 = std::get_if<T>(&result))
        {
            // The resulting surface is the same type: check data
            EXPECT_VEC_SOFT_EQ(t.data(), s2->data());
        }
        else
        {
            ADD_FAILURE() << "Actual: " << std::visit(ToString{}, result);
        }

        // Check for sense flip
        EXPECT_EQ(new_sense, this->sense);
    }

    Sense sense{Sense::inside};
    SurfaceSimplifier simplify{&sense, 1e-6};
};

//---------------------------------------------------------------------------//

TEST_F(SurfaceSimplifierTest, plane_aligned)
{
    this->check_unchanged(PlaneX{4.0});

    this->check_simplifies_to(PlaneX{1e-8}, PlaneX{0});
}

TEST_F(SurfaceSimplifierTest, cyl_centered)
{
    this->check_unchanged(CCylX{1e-8});
}

TEST_F(SurfaceSimplifierTest, sphere_centered)
{
    this->check_unchanged(SphereCentered{1e-8});
}

TEST_F(SurfaceSimplifierTest, cyl_aligned)
{
    this->check_unchanged(CylZ{{1, 2, 3}, 0.5});

    this->check_round_trip<CylX>(CCylX{4.0});
    this->check_round_trip<CylY>(CCylY{1.0});
    this->check_round_trip<CylZ>(CCylZ{0.1});

    {
        SCOPED_TRACE("near center");
        this->check_simplifies_to(CylZ{{1e-8, -1e-9, 1e-10}, 0.5}, CCylZ{0.5});
    }
}

TEST_F(SurfaceSimplifierTest, plane)
{
    this->check_unchanged(Plane{{1 / sqrt_two, 0, 1 / sqrt_two}, 2.0});
    this->check_unchanged(Plane{{1 / sqrt_two, 0, 1 / sqrt_two}, 0.0});

    this->check_round_trip<Plane>(PlaneX{4.0});
    this->check_round_trip<Plane>(PlaneY{-1.0});
    this->check_round_trip<Plane>(PlaneZ{1.0});

    this->check_simplifies_to(
        Plane{{-1 / sqrt_two, -1 / sqrt_two, 0.0}, -2 * sqrt_two},
        Plane{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two},
        Sense::outside);

    this->check_simplifies_to(Plane{{sqrt_three / 2, 0.5, 0.0}, 1e-15},
                              Plane{{sqrt_three / 2, 0.5, 0.0}, 0});

    // Check vector/displacement normalization
    Real3 n = make_unit_vector(Real3{1, 0, 1e-4});
    this->check_simplifies_to(Plane{n, {5.0, 0, 0}}, PlaneX{5.0});

    // First pass should clip zeros and normalize
    n = make_unit_vector(Real3{-1, 0, 1e-7});
    this->check_simplifies_to(
        Plane{n, {5.0, 0, 0}}, Plane{{1, 0, 0}, {5, 0, 0}}, Sense::outside);
}

TEST_F(SurfaceSimplifierTest, sphere)
{
    this->check_unchanged(Sphere{{1, 2, 3}, 0.5});

    this->check_round_trip<Sphere>(SphereCentered{0.2});

    this->check_simplifies_to(Sphere{{1e-7, -1e-7, 1e-9}, 0.5},
                              SphereCentered{0.5});
}

TEST_F(SurfaceSimplifierTest, cone_aligned)
{
    this->check_unchanged(ConeX{{1, 2, 3}, 0.5});
    this->check_simplifies_to(ConeX{{1e-7, -1e-7, 1e-9}, 0.5},
                              ConeX{{0, 0, 0}, 0.5});
    this->check_simplifies_to(ConeY{{10, -1e-7, 1}, 0.5},
                              ConeY{{10, 0, 1}, 0.5});
    this->check_unchanged(ConeX{{0, 0, 0}, 0.5});
}

TEST_F(SurfaceSimplifierTest, simple_quadric)
{
    {
        SCOPED_TRACE("plane");
        this->check_round_trip<SimpleQuadric>(
            Plane{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two});
    }
    {
        SCOPED_TRACE("cylinder");
        this->check_round_trip<SimpleQuadric>(CylX{{4, 5, -1}, 4.0});
        this->check_round_trip<SimpleQuadric>(CylY{{4, 5, -1}, 1.0});

        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            this->check_round_trip<SimpleQuadric>(CylZ{{4, 5, -1}, 0.1});
        }
    }

    {
        SCOPED_TRACE("cone");
        this->check_round_trip<SimpleQuadric>(ConeX{{2, 5, -1}, 10.0});
        this->check_round_trip<SimpleQuadric>(ConeY{{2, 5, -1}, 1.0});
        this->check_round_trip<SimpleQuadric>(ConeZ{{2, 5, -1}, 0.1});
    }

    {
        SCOPED_TRACE("sphere");
        // {1,1,1} {-2,-4,-6} 13.75
        this->check_round_trip<SimpleQuadric>(Sphere{{1, 2, 3}, 0.5});
        this->check_simplifies_to(SimpleQuadric{{4, 4, 4}, {-8, -16, -24}, 55},
                                  Sphere{{1, 2, 3}, 0.5});
        // Complex radius
        this->check_unchanged(SimpleQuadric{{2, 2, 2}, {0, 0, 0}, 10});
    }

    {
        SCOPED_TRACE("inverted ellipsoid");
        this->check_simplifies_to(
            SimpleQuadric{{-0.5625, -0.09, -6.25}, {0, 0, 0}, 0.5625},
            SimpleQuadric{{0.5625, 0.09, 6.25}, {0, 0, 0}, -0.5625},
            Sense::outside);
        this->check_unchanged(
            SimpleQuadric{{0.5625, 0.09, 6.25}, {0, 0, 0}, -0.5625});
    }

    {
        SCOPED_TRACE("inverted scaled cone");
        // {-0.5625,1,1} {1.6875,-4.5,-7.5} 17.859375
        this->check_simplifies_to(
            SimpleQuadric{{2.25, -4, -4}, {-6.75, 18.0, 30.0}, -71.4375},
            SimpleQuadric{{-2.25, 4, 4}, {6.75, -18.0, -30.0}, 71.4375},
            Sense::outside);
        this->check_simplifies_to(
            SimpleQuadric{{-2.25, 4, 4}, {6.75, -18.0, -30.0}, 71.4375},
            ConeX{{1.5, 2.25, 3.75}, 0.75});
    }

    {
        SCOPED_TRACE("scaled near-cylinder");
        // CylY{{1,2,3}, 2.5} -> SQ{{1,0,1}, {-2,0,-6}, 3.75}
        this->check_simplifies_to(
            SimpleQuadric{{1, 1e-9, 1}, {-2, 0, -6}, 3.75},
            CylY{{1, 2, 3}, 2.5});
    }

    {
        SCOPED_TRACE("scaled negative cylinder");
        this->check_simplifies_to(
            SimpleQuadric{{-2, 0, -2}, {4, 0, 12}, -2 * 3.75},
            SimpleQuadric{{2, 0, 2}, {-4, 0, -12}, 2 * 3.75},
            Sense::outside);

        this->check_simplifies_to(
            SimpleQuadric{{2, 0, 2}, {-4, 0, -12}, 2 * 3.75},
            CylY{{1, 2, 3}, 2.5});
    }

    {
        SCOPED_TRACE("hyperboloid");  // (not a cone!)
        this->check_unchanged(
            SimpleQuadric{{0.5625, 0.09, -6.25}, {0, 0, 0}, -0.5625});
    }
}

TEST_F(SurfaceSimplifierTest, general_quadric)
{
    this->check_unchanged(
        GeneralQuadric{{10.3125, 22.9375, 15.75},
                       {-21.867141445557, -20.25, 11.69134295109},
                       {-11.964745962156, -9.1328585544429, -65.69134295109},
                       77.652245962156});

    this->check_simplifies_to(
        GeneralQuadric{{-2, 0, -2}, {0, 0, 0}, {4, 0, 12}, -2 * 3.75},
        SimpleQuadric{{-2, 0, -2}, {4, 0, 12}, -2 * 3.75});

    this->check_simplifies_to(
        GeneralQuadric{{1, 2, 3}, {0, 0, 0}, {-1, -1, 1}, 6},
        SimpleQuadric{{1, 2, 3}, {-1, -1, 1}, 6});
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
