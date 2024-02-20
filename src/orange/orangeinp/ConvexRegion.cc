//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexRegion.cc
//---------------------------------------------------------------------------//
#include "ConvexRegion.hh"

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/cont/Range.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/Types.hh"
#include "orange/surf/ConeAligned.hh"
#include "orange/surf/CylCentered.hh"
#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/SimpleQuadric.hh"
#include "orange/surf/SphereCentered.hh"

#include "ConvexSurfaceBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Create a z-aligned bounding box infinite along z and symmetric in r.
 */
BBox make_radial_bbox(real_type r)
{
    CELER_EXPECT(r > 0);
    constexpr auto inf = numeric_limits<real_type>::infinity();
    return BBox::from_unchecked({-r, -r, -inf}, {r, r, inf});
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// BOX
//---------------------------------------------------------------------------//
/*!
 * Construct with half-widths.
 */
Box::Box(Real3 const& halfwidths) : hw_{halfwidths}
{
    for (auto ax : range(Axis::size_))
    {
        CELER_VALIDATE(hw_[to_int(ax)] > 0,
                       << "nonpositive halfwidth along " << to_char(ax)
                       << " axis: " << hw_[to_int(ax)]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Box::build(ConvexSurfaceBuilder& insert_surface) const
{
    constexpr auto X = to_int(Axis::x);
    constexpr auto Y = to_int(Axis::y);
    constexpr auto Z = to_int(Axis::z);

    insert_surface(Sense::outside, PlaneX{-hw_[X]});
    insert_surface(Sense::inside, PlaneX{hw_[X]});
    insert_surface(Sense::outside, PlaneY{-hw_[Y]});
    insert_surface(Sense::inside, PlaneY{hw_[Y]});
    insert_surface(Sense::outside, PlaneZ{-hw_[Z]});
    insert_surface(Sense::inside, PlaneZ{hw_[Z]});
}

//---------------------------------------------------------------------------//
// CONE
//---------------------------------------------------------------------------//
/*!
 * Construct with Z halfwidth and lo, hi radii.
 */
Cone::Cone(Real2 const& radii, real_type halfheight)
    : radii_{radii}, hh_{halfheight}
{
    for (auto i : range(2))
    {
        CELER_VALIDATE(radii_[i] >= 0, << "negative radius: " << radii_[i]);
    }
    CELER_VALIDATE(radii_[0] != radii_[1], << "radii cannot be equal");
    CELER_VALIDATE(hh_ > 0, << "nonpositive halfheight: " << hh_);
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Cone::build(ConvexSurfaceBuilder& insert_surface) const
{
    // Build the bottom and top planes
    insert_surface(Sense::outside, PlaneZ{-hh_});
    insert_surface(Sense::inside, PlaneZ{hh_});

    // Calculate the cone using lo and hi radii
    real_type const lo{radii_[0]};
    real_type const hi{radii_[1]};

    // Arctangent of the opening angle of the cone (opposite / adjacent)
    real_type const tangent = std::fabs(lo - hi) / (2 * hh_);

    // Calculate vanishing point (origin)
    real_type vanish_z = 0;
    if (hi < lo)
    {
        // Cone opens downward (base is on bottom)
        vanish_z = -hh_ + lo / tangent;
        CELER_ASSERT(vanish_z > 0);
    }
    else
    {
        // Cone opens upward
        vanish_z = hh_ - hi / tangent;
        CELER_ASSERT(vanish_z < 0);
    }

    // Build the cone surface along the given axis
    insert_surface(ConeZ{Real3{0, 0, vanish_z}, tangent});

    // Set radial extents of exterior bbox
    insert_surface(Sense::inside, make_radial_bbox(std::fmax(lo, hi)));
    // TODO: set radial extents of interior bbox
}

//---------------------------------------------------------------------------//
// CYLINDER
//---------------------------------------------------------------------------//
/*!
 * Construct with radius.
 */
Cylinder::Cylinder(real_type radius, real_type halfheight)
    : radius_{radius}, hh_{halfheight}
{
    CELER_VALIDATE(radius_ > 0, << "nonpositive radius: " << radius_);
    CELER_VALIDATE(hh_ > 0, << "nonpositive half-height: " << hh_);
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Cylinder::build(ConvexSurfaceBuilder& insert_surface) const
{
    insert_surface(Sense::outside, PlaneZ{-hh_});
    insert_surface(Sense::inside, PlaneZ{hh_});
    insert_surface(CCylZ{radius_});
}

//---------------------------------------------------------------------------//
// ELLIPSOID
//---------------------------------------------------------------------------//
/*!
 * Construct with half-widths.
 */
Ellipsoid::Ellipsoid(Real3 const& radii) : radii_{radii}
{
    for (auto ax : range(Axis::size_))
    {
        CELER_VALIDATE(radii_[to_int(ax)] > 0,
                       << "nonpositive radius " << to_char(ax)
                       << " axis: " << radii_[to_int(ax)]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Ellipsoid::build(ConvexSurfaceBuilder& insert_surface) const
{
    // Second-order coefficients are product of the other two squared radii;
    // Zeroth-order coefficient is the product of all three squared radii
    Real3 rsq;
    for (auto ax : range(to_int(Axis::size_)))
    {
        rsq[ax] = ipow<2>(radii_[ax]);
    }

    Real3 abc{1, 1, 1};
    real_type g = -1;
    for (auto ax : range(to_int(Axis::size_)))
    {
        g *= rsq[ax];
        for (auto nax : range(to_int(Axis::size_)))
        {
            if (ax != nax)
            {
                abc[ax] *= rsq[nax];
            }
        }
    }

    insert_surface(SimpleQuadric{abc, Real3{0, 0, 0}, g});

    // Set exterior bbox
    insert_surface(Sense::inside, BBox{-radii_, radii_});
    // TODO: set radial extents of interior bbox
}

//---------------------------------------------------------------------------//
// PRISM
//---------------------------------------------------------------------------//
/*!
 * Construct with inner radius (apothem), half height, and orientation.
 */
Prism::Prism(int num_sides,
             real_type apothem,
             real_type halfheight,
             real_type orientation)
    : num_sides_{num_sides}
    , apothem_{apothem}
    , hh_{halfheight}
    , orientation_{orientation}
{
    CELER_VALIDATE(num_sides_ >= 3,
                   << "degenerate prism (num_sides = " << num_sides_ << ')');
    CELER_VALIDATE(apothem_ > 0, << "nonpositive apothem: " << apothem_);
    CELER_VALIDATE(hh_ > 0, << "nonpositive half-height " << hh_);
    CELER_VALIDATE(orientation_ >= 0 && orientation_ < 1,
                   << "orientation is out of bounds [0, 1): " << orientation_);
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Prism::build(ConvexSurfaceBuilder& insert_surface) const
{
    using constants::pi;

    // Build top and bottom
    insert_surface(Sense::outside, PlaneZ{-hh_});
    insert_surface(Sense::inside, PlaneZ{hh_});

    // Offset (if user offset is zero) is calculated to put a plane on the
    // -y face (sitting upright as visualized). An offset of 1 produces a
    // shape congruent with an offset of zero, except that every face has
    // an index that's decremented by 1.
    real_type const offset = std::fmod(num_sides_ * 3 + 4 * orientation_, 4)
                             / 4;
    CELER_ASSERT(offset >= 0 && offset < 1);

    // Change of angle in radians per side
    real_type const delta_rad = 2 * pi / static_cast<real_type>(num_sides_);

    // Build prismatic sides
    for (auto n : range(num_sides_))
    {
        real_type const theta = delta_rad * (n + offset);

        // Create a normal vector along the X axis, then rotate it through
        // the angle theta
        Real3 normal{0, 0, 0};
        normal[to_int(Axis::x)] = std::cos(theta);
        normal[to_int(Axis::y)] = std::sin(theta);

        insert_surface(Plane{normal, apothem_});
    }

    // Apothem is interior, circumradius exterior
    insert_surface(Sense::inside,
                   make_radial_bbox(apothem_ / std::cos(pi / num_sides_)));

    auto interior_bbox = make_radial_bbox(apothem_);
    interior_bbox.shrink(Bound::lo, Axis::z, -hh_);
    interior_bbox.shrink(Bound::hi, Axis::z, hh_);
    insert_surface(Sense::outside, interior_bbox);
}

//---------------------------------------------------------------------------//
// SPHERE
//---------------------------------------------------------------------------//
/*!
 * Construct with radius.
 */
Sphere::Sphere(real_type radius) : radius_{radius}
{
    CELER_VALIDATE(radius_ > 0, << "nonpositive radius: " << radius_);
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Sphere::build(ConvexSurfaceBuilder& insert_surface) const
{
    insert_surface(SphereCentered{radius_});
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
