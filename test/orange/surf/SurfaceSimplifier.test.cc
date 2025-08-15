//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
constexpr real_type sqrt_three{constants::sqrt_three};
constexpr real_type sqrt_two{constants::sqrt_two};

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
// PLANE ALIGNED
//---------------------------------------------------------------------------//
using PlaneAlignedTest = SurfaceSimplifierTest;

TEST_F(PlaneAlignedTest, unchanged)
{
    this->check_unchanged(PlaneX{4.0});
}

TEST_F(PlaneAlignedTest, simplifies_to_zero)
{
    this->check_simplifies_to(PlaneX{1e-8}, PlaneX{0});
}

//---------------------------------------------------------------------------//
// CYLINDER CENTERED
//---------------------------------------------------------------------------//
using CylCenteredTest = SurfaceSimplifierTest;

TEST_F(CylCenteredTest, unchanged)
{
    this->check_unchanged(CCylX{1e-8});
}

//---------------------------------------------------------------------------//
// SPHERE CENTERED
//---------------------------------------------------------------------------//
using SphereCenteredTest = SurfaceSimplifierTest;

TEST_F(SphereCenteredTest, unchanged)
{
    this->check_unchanged(SphereCentered{1e-8});
}

//---------------------------------------------------------------------------//
// CYLINDER ALIGNED
//---------------------------------------------------------------------------//
using CylAlignedTest = SurfaceSimplifierTest;

TEST_F(CylAlignedTest, unchanged)
{
    this->check_unchanged(CylZ{{1, 2, 3}, 0.5});
}

TEST_F(CylAlignedTest, round_trip_x)
{
    this->check_round_trip<CylX>(CCylX{4.0});
}

TEST_F(CylAlignedTest, round_trip_y)
{
    this->check_round_trip<CylY>(CCylY{1.0});
}

TEST_F(CylAlignedTest, round_trip_z)
{
    this->check_round_trip<CylZ>(CCylZ{0.1});
}

TEST_F(CylAlignedTest, near_center)
{
    this->check_simplifies_to(CylZ{{1e-8, -1e-9, 1e-10}, 0.5}, CCylZ{0.5});
}

//---------------------------------------------------------------------------//
// PLANE
//---------------------------------------------------------------------------//
using PlaneTest = SurfaceSimplifierTest;

TEST_F(PlaneTest, unchanged)
{
    this->check_unchanged(Plane{{1 / sqrt_two, 0, 1 / sqrt_two}, 2.0});
    this->check_unchanged(Plane{{1 / sqrt_two, 0, 1 / sqrt_two}, 0.0});
    this->check_unchanged(
        Plane{{1 / sqrt_three, -1 / sqrt_three, 1 / sqrt_three}, 0.0});
    this->check_unchanged(
        Plane{{-1 / sqrt_three, 1 / sqrt_three, 1 / sqrt_three}, 0.0});
}

TEST_F(PlaneTest, round_trip_aligned)
{
    this->check_round_trip<Plane>(PlaneX{4.0});
    this->check_round_trip<Plane>(PlaneY{-1.0});
    this->check_round_trip<Plane>(PlaneZ{1.0});
}

TEST_F(PlaneTest, simplifies_with_sense_flip)
{
    this->check_simplifies_to(
        Plane{{-1 / sqrt_two, -1 / sqrt_two, 0.0}, -2 * sqrt_two},
        Plane{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two},
        Sense::outside);
    this->check_simplifies_to(
        Plane{{1 / sqrt_three, -1 / sqrt_three, -1 / sqrt_three},
              -2 * sqrt_three},
        Plane{{-1 / sqrt_three, 1 / sqrt_three, 1 / sqrt_three}, 2 * sqrt_three},
        Sense::outside);
}

TEST_F(PlaneTest, zero_displacement)
{
    this->check_simplifies_to(Plane{{sqrt_three / 2, 0.5, 0.0}, 1e-15},
                              Plane{{sqrt_three / 2, 0.5, 0.0}, 0});
}

TEST_F(PlaneTest, vector_normalization)
{
    // Check vector/displacement normalization
    Real3 n = make_unit_vector(Real3{1, 0, 1e-7});
    this->check_simplifies_to(Plane{n, {5.0, 0, 0}}, PlaneX{5.0});
}

TEST_F(PlaneTest, first_pass_normalization)
{
    // First pass should normalize
    Real3 n = make_unit_vector(Real3{-1, 0, 1e-7});
    this->check_simplifies_to(
        Plane{n, {5.0, 0, 0}}, Plane{{1, 0, -1e-7}, {5, 0, 0}}, Sense::outside);
}

//---------------------------------------------------------------------------//
// SPHERE
//---------------------------------------------------------------------------//
using SphereTest = SurfaceSimplifierTest;

TEST_F(SphereTest, unchanged)
{
    this->check_unchanged(Sphere{{1, 2, 3}, 0.5});
}

TEST_F(SphereTest, round_trip)
{
    this->check_round_trip<Sphere>(SphereCentered{0.2});
}

TEST_F(SphereTest, simplifies_to_centered)
{
    this->check_simplifies_to(Sphere{{1e-7, -1e-7, 1e-9}, 0.5},
                              SphereCentered{0.5});
}

//---------------------------------------------------------------------------//
// CONE ALIGNED
//---------------------------------------------------------------------------//
using ConeAlignedTest = SurfaceSimplifierTest;

TEST_F(ConeAlignedTest, unchanged)
{
    this->check_unchanged(ConeX{{1, 2, 3}, 0.5});
    this->check_unchanged(ConeX{{0, 0, 0}, 0.5});
}

TEST_F(ConeAlignedTest, simplifies_to_origin)
{
    this->check_simplifies_to(ConeX{{1e-7, -1e-7, 1e-9}, 0.5},
                              ConeX{{0, 0, 0}, 0.5});
}

TEST_F(ConeAlignedTest, simplifies_y_coordinate)
{
    this->check_simplifies_to(ConeY{{10, -1e-7, 1}, 0.5},
                              ConeY{{10, 0, 1}, 0.5});
}

//---------------------------------------------------------------------------//
// SIMPLE QUADRIC
//---------------------------------------------------------------------------//
using SimpleQuadricTest = SurfaceSimplifierTest;

TEST_F(SimpleQuadricTest, plane)
{
    this->check_round_trip<SimpleQuadric>(
        Plane{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two});
}

TEST_F(SimpleQuadricTest, cylinder)
{
    this->check_round_trip<SimpleQuadric>(CylX{{4, 5, -1}, 4.0});
    this->check_round_trip<SimpleQuadric>(CylY{{4, 5, -1}, 1.0});

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        this->check_round_trip<SimpleQuadric>(CylZ{{4, 5, -1}, 0.1});
    }

    // Inverted
    this->check_simplifies_to(SimpleQuadric{{-2, 0, -2}, {4, 0, 12}, -2 * 3.75},
                              SimpleQuadric{{2, 0, 2}, {-4, 0, -12}, 2 * 3.75},
                              Sense::outside);

    // CylY{{1,2,3}, 2.5} -> SQ{{1,0,1}, {-2,0,-6}, 3.75}
    this->check_simplifies_to(SimpleQuadric{{1, 1e-9, 1}, {-2, 0, -6}, 3.75},
                              CylY{{1, 2, 3}, 2.5});
    this->check_simplifies_to(SimpleQuadric{{2, 0, 2}, {-4, 0, -12}, 2 * 3.75},
                              CylY{{1, 2, 3}, 2.5});
}

TEST_F(SimpleQuadricTest, cone)
{
    this->check_round_trip<SimpleQuadric>(ConeX{{2, 5, -1}, 10.0});
    this->check_round_trip<SimpleQuadric>(ConeY{{2, 5, -1}, 1.0});
    this->check_round_trip<SimpleQuadric>(ConeZ{{2, 5, -1}, 0.1});

    // Inverted
    this->check_simplifies_to(
        SimpleQuadric{{2.25, -4, -4}, {-6.75, 18.0, 30.0}, -71.4375},
        SimpleQuadric{{-2.25, 4, 4}, {6.75, -18.0, -30.0}, 71.4375},
        Sense::outside);
    // Then just a cone
    this->check_simplifies_to(
        SimpleQuadric{{-2.25, 4, 4}, {6.75, -18.0, -30.0}, 71.4375},
        ConeX{{1.5, 2.25, 3.75}, 0.75});
}

TEST_F(SimpleQuadricTest, sphere)
{
    // {1,1,1} {-2,-4,-6} 13.75
    this->check_round_trip<SimpleQuadric>(Sphere{{1, 2, 3}, 0.5});
    this->check_simplifies_to(SimpleQuadric{{4, 4, 4}, {-8, -16, -24}, 55},
                              Sphere{{1, 2, 3}, 0.5});
    // Complex radius
    this->check_unchanged(SimpleQuadric{{2, 2, 2}, {0, 0, 0}, 10});
}

TEST_F(SimpleQuadricTest, ellipsoid)
{
    // Inverted
    this->check_simplifies_to(
        SimpleQuadric{{-0.5625, -0.09, -6.25}, {0, 0, 0}, 0.5625},
        SimpleQuadric{{0.5625, 0.09, 6.25}, {0, 0, 0}, -0.5625},
        Sense::outside);

    // Unchanged
    this->check_unchanged(
        SimpleQuadric{{0.5625, 0.09, 6.25}, {0, 0, 0}, -0.5625});

    // Small (from Ellipsoid({0.008, 0.004, 0.005}))
    this->check_unchanged(SimpleQuadric{{0.5, 2, 1.28}, {0, 0, 0}, -3.2e-05});
}

TEST_F(SimpleQuadricTest, hyperboloid)
{
    // (not a cone!)
    this->check_unchanged(
        SimpleQuadric{{0.5625, 0.09, -6.25}, {0, 0, 0}, -0.5625});
}

//---------------------------------------------------------------------------//
// GENERAL QUADRIC
//---------------------------------------------------------------------------//
using GeneralQuadricTest = SurfaceSimplifierTest;

TEST_F(GeneralQuadricTest, rotated_ellipsoid)
{
    this->check_unchanged(
        GeneralQuadric{{10.3125, 22.9375, 15.75},
                       {-21.867141445557, -20.25, 11.69134295109},
                       {-11.964745962156, -9.1328585544429, -65.69134295109},
                       77.652245962156});
}

TEST_F(GeneralQuadricTest, axis_aligned_ellipsoid)
{
    this->check_simplifies_to(
        GeneralQuadric{{-2, 0, -2}, {0, 0, 0}, {4, 0, 12}, -2 * 3.75},
        SimpleQuadric{{-2, 0, -2}, {4, 0, 12}, -2 * 3.75});
}

TEST_F(GeneralQuadricTest, general_surface)
{
    this->check_unchanged(
        GeneralQuadric{{0, 0, 0}, {0, 0.5, 0}, {2, 0.5, 0}, 0});
    this->check_simplifies_to(
        GeneralQuadric{{0, 0, 0}, {0, -0.5, 0}, {-2, -0.5, 0}, 0},
        GeneralQuadric{{0, 0, 0}, {0, 0.5, 0}, {2, 0.5, 0}, 0},
        Sense::outside);
    this->check_simplifies_to(
        GeneralQuadric{{0, 0, 0}, {-1, 1, 0}, {1, 1, 0}, 0},
        GeneralQuadric{{0, 0, 0}, {1, -1, 0}, {-1, -1, 0}, 0},
        Sense::outside);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
