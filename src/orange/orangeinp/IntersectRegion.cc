//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectRegion.cc
//---------------------------------------------------------------------------//
#include "IntersectRegion.hh"

#include <cmath>

#include "corecel/Constants.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/Types.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/detail/PolygonUtils.hh"
#include "orange/surf/ConeAligned.hh"
#include "orange/surf/CylCentered.hh"
#include "orange/surf/Involute.hh"
#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/SimpleQuadric.hh"
#include "orange/surf/SphereCentered.hh"

#include "IntersectSurfaceBuilder.hh"
#include "ObjectIO.json.hh"

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
BBox make_xyradial_bbox(real_type r)
{
    CELER_EXPECT(r > 0);
    constexpr auto inf = numeric_limits<real_type>::infinity();
    return BBox::from_unchecked({-r, -r, -inf}, {r, r, inf});
}

//! Convenience enumeration for implementations in this file
enum
{
    X = 0,
    Y = 1,
    Z = 2
};

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
void Box::build(IntersectSurfaceBuilder& insert_surface) const
{
    insert_surface(Sense::outside, PlaneX{-hw_[X]});
    insert_surface(Sense::inside, PlaneX{hw_[X]});
    insert_surface(Sense::outside, PlaneY{-hw_[Y]});
    insert_surface(Sense::inside, PlaneY{hw_[Y]});
    insert_surface(Sense::outside, PlaneZ{-hw_[Z]});
    insert_surface(Sense::inside, PlaneZ{hw_[Z]});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Box::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// CONE
//---------------------------------------------------------------------------//
/*!
 * Construct with Z half-height and lo, hi radii.
 */
Cone::Cone(Real2 const& radii, real_type halfheight)
    : radii_{radii}, hh_{halfheight}
{
    for (auto i : range(2))
    {
        CELER_VALIDATE(radii_[i] >= 0, << "negative radius: " << radii_[i]);
    }
    CELER_VALIDATE(hh_ > 0, << "nonpositive halfheight: " << hh_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether this encloses another cone.
 */
bool Cone::encloses(Cone const& other) const
{
    return radii_[0] >= other.radii_[0] && radii_[1] >= other.radii_[1]
           && hh_ >= other.hh_;
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 *
 * The inner bounding box of a cone is determined with the following procedure:
 * - Represent a radial slice of the cone as a right triangle with base b
 *   (aka the higher radius) and height h (translated vanishing point)
 * - An interior bounding box (along the xy diagonal cut!) will satisfy
 *   r = b - tangent * z
 * - Maximize the area of that box to obtain r = b / 2, i.e. z = h / 2
 * - Truncate z so that it's not outside of the half-height
 * - Project that radial slice onto the xz plane by multiplying 1/sqrt(2)
 */
void Cone::build(IntersectSurfaceBuilder& insert_surface) const
{
    if (CELER_UNLIKELY(
            SoftEqual{insert_surface.tol().rel}(radii_[0], radii_[1])))
    {
        // Degenerate cone: build a cylinder instead
        Cylinder cyl{real_type{0.5} * (radii_[0] + radii_[1]), hh_};
        return cyl.build(insert_surface);
    }

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
    if (lo > hi)
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
    ConeZ cone{Real3{0, 0, vanish_z}, tangent};
    insert_surface(cone);

    // Set radial extents of exterior bbox
    insert_surface(Sense::inside, make_xyradial_bbox(std::fmax(lo, hi)));

    // Calculate the interior bounding box:
    real_type const b = std::fmax(lo, hi);
    real_type const h = b / tangent;
    real_type const z = std::fmin(h / 2, 2 * hh_);
    real_type const r = b - tangent * z;

    // Now convert from "triangle above z=0" to "cone centered on z=0"
    real_type zmin = -hh_;
    real_type zmax = zmin + z;
    if (lo < hi)
    {
        // Base is on top
        zmax = hh_;
        zmin = zmax - z;
    }
    CELER_ASSERT(zmin < zmax);
    real_type const rbox = (constants::sqrt_two / 2) * r;
    BBox const interior_bbox{{-rbox, -rbox, zmin}, {rbox, rbox, zmax}};

    // Check that the corners are actually inside the cone
    CELER_ASSERT(cone.calc_sense(interior_bbox.lower() * real_type(1 - 1e-5))
                 == SignedSense::inside);
    CELER_ASSERT(cone.calc_sense(interior_bbox.upper() * real_type(1 - 1e-5))
                 == SignedSense::inside);
    insert_surface(Sense::outside, interior_bbox);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Cone::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
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
 * Whether this encloses another cylinder.
 */
bool Cylinder::encloses(Cylinder const& other) const
{
    return radius_ >= other.radius_ && hh_ >= other.hh_;
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Cylinder::build(IntersectSurfaceBuilder& insert_surface) const
{
    insert_surface(Sense::outside, PlaneZ{-hh_});
    insert_surface(Sense::inside, PlaneZ{hh_});
    insert_surface(CCylZ{radius_});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Cylinder::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// ELLIPSOID
//---------------------------------------------------------------------------//
/*!
 * Construct with radii.
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
 * Whether this encloses another ellipsoid.
 */
bool Ellipsoid::encloses(Ellipsoid const& other) const
{
    for (auto ax : range(Axis::size_))
    {
        if (this->radii_[to_int(ax)] < other.radii_[to_int(ax)])
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Ellipsoid::build(IntersectSurfaceBuilder& insert_surface) const
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

    // Set an interior bbox with maximum volume: a scaled inscribed cuboid
    Real3 inner_radii = radii_;
    for (real_type& r : inner_radii)
    {
        r *= 1 / constants::sqrt_three;
    }
    insert_surface(Sense::outside, BBox{-inner_radii, inner_radii});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Ellipsoid::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// ELLIPTICAL CYLINDER
//---------------------------------------------------------------------------//
/*!
 * Construct with x- and y-radii and half-height in z.
 */
EllipticalCylinder::EllipticalCylinder(Real2 const& radii, real_type halfheight)
    : radii_{radii}, hh_{halfheight}
{
    for (auto i : range(2))
    {
        CELER_VALIDATE(radii_[i] >= 0, << "negative radius: " << radii_[i]);
    }
    CELER_VALIDATE(hh_ > 0, << "nonpositive halfheight: " << hh_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether this encloses another elliptical cylinder.
 */
bool EllipticalCylinder::encloses(EllipticalCylinder const& other) const
{
    for (auto ax : range(2))
    {
        if (this->radii_[ax] < other.radii_[ax])
        {
            return false;
        }
    }
    return hh_ >= other.hh_;
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void EllipticalCylinder::build(IntersectSurfaceBuilder& insert_surface) const
{
    insert_surface(Sense::outside, PlaneZ{-hh_});
    insert_surface(Sense::inside, PlaneZ{hh_});

    // Insert elliptical cylinder surface last, as a simple quadric with
    // equation:
    // r_y^2 x^2 + r_x^2 y^2 - r_x^2 r_y^2 = 0
    real_type rx_sq = ipow<2>(radii_[to_int(Axis::x)]);
    real_type ry_sq = ipow<2>(radii_[to_int(Axis::y)]);
    real_type g = -rx_sq * ry_sq;

    Real3 abc{ry_sq, rx_sq, 0};
    insert_surface(SimpleQuadric{abc, Real3{0, 0, 0}, g});

    // Set exterior bbox
    Real3 ex_halves{radii_[to_int(Axis::x)], radii_[to_int(Axis::y)], hh_};
    insert_surface(Sense::inside, BBox{-ex_halves, ex_halves});

    // Set an interior bbox (inscribed cuboid)
    auto inv_sqrt_two = 1 / constants::sqrt_two;
    Real3 in_halves{radii_[to_int(Axis::x)] * inv_sqrt_two,
                    radii_[to_int(Axis::y)] * inv_sqrt_two,
                    hh_};
    insert_surface(Sense::outside, BBox{-in_halves, in_halves});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void EllipticalCylinder::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// ELLIPTICAL CONE
//---------------------------------------------------------------------------//
/*!
 * Construct with lower/upper x- and y-radii and half-height in z.
 */
EllipticalCone::EllipticalCone(Real2 const& lower_radii,
                               Real2 const& upper_radii,
                               real_type halfheight)
    : lower_radii_{lower_radii}, upper_radii_{upper_radii}, hh_{halfheight}
{
    // True if either radius is negative
    auto has_negative
        = [](Real2 const& radii) { return radii[X] < 0 || radii[Y] < 0; };

    // True if radii is (0, 0)
    auto is_vertex = [](Real2 const& radii) {
        return soft_zero(radii[X]) && soft_zero(radii[Y]);
    };

    // True if radii is (0, x) || (x, 0), where x != 0
    auto is_partial_zero = [](Real2 const& radii) {
        return (soft_zero(radii[X]) != soft_zero(radii[Y]));
    };

    // Check for negatives
    CELER_VALIDATE(!has_negative(lower_radii_),
                   << "negative lower radii: " << lower_radii_[X] << ", "
                   << lower_radii_[Y]);
    CELER_VALIDATE(!has_negative(upper_radii_),
                   << "negative upper radii: " << upper_radii_[X] << ", "
                   << upper_radii_[Y]);

    // Check for partial zeros
    CELER_VALIDATE(!is_partial_zero(lower_radii_),
                   << "mismatched zero lower radii: " << lower_radii_[X]
                   << ", " << lower_radii_[Y]);
    CELER_VALIDATE(!is_partial_zero(upper_radii_),
                   << "mismatched zero upper radii: " << upper_radii_[X]
                   << ", " << upper_radii_[Y]);

    // Check aspect ratios
    if (!is_vertex(lower_radii_) && !is_vertex(upper_radii_))
    {
        CELER_VALIDATE(soft_equal(lower_radii_[X] / lower_radii_[Y],
                                  upper_radii_[X] / upper_radii_[Y]),
                       << "differing aspect ratios for upper and lower radii");
    }

    // Check for elliptical cylinders. Since we have already validated the
    // aspect ratio, we only need to test the x-values here.
    CELER_VALIDATE(!soft_equal(lower_radii_[X], upper_radii_[X]),
                   << "equal and lower and upper radii (use cylinder "
                      "instead)");

    // Check positivity of half-height
    CELER_VALIDATE(hh_ > 0, << "nonpositive halfheight: " << hh_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether this encloses another elliptical cone.
 */
bool EllipticalCone::encloses(EllipticalCone const& other) const
{
    for (auto ax : range(2))
    {
        if (this->lower_radii_[ax] < other.lower_radii_[ax]
            || this->upper_radii_[ax] < other.upper_radii_[ax])
        {
            return false;
        }
    }
    return hh_ >= other.hh_;
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void EllipticalCone::build(IntersectSurfaceBuilder& insert_surface) const
{
    insert_surface(Sense::outside, PlaneZ{-hh_});
    insert_surface(Sense::inside, PlaneZ{hh_});

    constexpr auto X = to_int(Axis::x);
    constexpr auto Y = to_int(Axis::y);

    real_type a = ipow<2>((2 * hh_) / (lower_radii_[X] - upper_radii_[X]));

    real_type b = ipow<2>((2 * hh_) / (lower_radii_[Y] - upper_radii_[Y]));

    real_type v = hh_ * (lower_radii_[X] + upper_radii_[X])
                  / (lower_radii_[X] - upper_radii_[X]);

    insert_surface(
        SimpleQuadric{Real3{a, b, -1}, Real3{0, 0, 2 * v}, -ipow<2>(v)});

    // Set an exterior bbox
    real_type x_max = std::fmax(lower_radii_[X], upper_radii_[X]);
    real_type y_max = std::fmax(lower_radii_[Y], upper_radii_[Y]);
    Real3 ex_halves{x_max, y_max, hh_};
    insert_surface(Sense::inside, BBox{-ex_halves, ex_halves});

    // TODO: interior bbox
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void EllipticalCone::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// GENPRISM
//---------------------------------------------------------------------------//
/*!
 * Construct from two simple, centered trapezoids.
 */
GenPrism GenPrism::from_trd(real_type halfz, Real2 const& lo, Real2 const& hi)
{
    CELER_VALIDATE(lo[0] > 0, << "nonpositive lower x half-edge: " << lo[0]);
    CELER_VALIDATE(hi[0] > 0, << "nonpositive upper x half-edge: " << hi[0]);
    CELER_VALIDATE(lo[1] > 0, << "nonpositive lower y half-edge: " << lo[1]);
    CELER_VALIDATE(hi[1] > 0, << "nonpositive upper y half-edge: " << hi[1]);
    CELER_VALIDATE(halfz > 0, << "nonpositive half-height: " << halfz);

    // Construct points like prism: lower right is first
    VecReal2 lower
        = {{lo[0], -lo[1]}, {lo[0], lo[1]}, {-lo[0], lo[1]}, {-lo[0], -lo[1]}};
    VecReal2 upper
        = {{hi[0], -hi[1]}, {hi[0], hi[1]}, {-hi[0], hi[1]}, {-hi[0], -hi[1]}};

    return GenPrism{halfz, std::move(lower), std::move(upper)};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from skewed trapezoids.
 *
 * For details on construction, see:
 * https://geant4-userdoc.web.cern.ch/UsersGuides/ForApplicationDeveloper/html/Detector/Geometry/geomSolids.html#constructed-solid-geometry-csg-solids
 *
 * \arg hz Half the distance between the faces
 * \arg theta Polar angle of line between center of bases
 * \arg phi Azimuthal angle of line between center of bases
 * \arg lo Trapezoidal face at -hz
 * \arg hi Trapezoidal face at +hz
 */
GenPrism GenPrism::from_trap(
    real_type hz, Turn theta, Turn phi, TrapFace const& lo, TrapFace const& hi)
{
    CELER_VALIDATE(hz > 0, << "nonpositive half-height: " << hz);
    CELER_VALIDATE(theta >= zero_quantity() && theta < Turn{0.25},
                   << "invalid angle " << theta.value()
                   << " [turns]: must be in the range [0, 0.25)");

    // Calculate offset of faces from z axis
    auto [dxdz_hz, dydz_hz] = [&]() -> std::pair<real_type, real_type> {
        real_type cos_phi{}, sin_phi{};
        sincos(phi, &sin_phi, &cos_phi);
        real_type const tan_theta = std::tan(native_value_from(theta));
        return {hz * tan_theta * cos_phi, hz * tan_theta * sin_phi};
    }();

    // Construct points on faces
    TrapFace const* const faces[] = {&lo, &hi};
    Array<VecReal2, 2> points;
    for (auto i : range(2))
    {
        TrapFace const& face = *faces[i];
        CELER_VALIDATE(face.hx_lo > 0,
                       << "nonpositive lower x half-edge: " << face.hx_lo);
        CELER_VALIDATE(face.hx_hi > 0,
                       << "nonpositive upper x half-edge: " << face.hx_hi);
        CELER_VALIDATE(face.hy > 0,
                       << "nonpositive y half-distance: " << face.hy);
        CELER_VALIDATE(face.alpha > Turn{-0.25} && face.alpha < Turn{0.25},
                       << "invalid trapezoidal shear: " << face.alpha.value()
                       << " [turns]: must be in the range (-0.25, -0.25)");

        real_type const xoff = (i == 0 ? -dxdz_hz : dxdz_hz);
        real_type const yoff = (i == 0 ? -dydz_hz : dydz_hz);
        real_type const shear = std::tan(native_value_from(face.alpha))
                                * face.hy;

        // Construct points counterclockwise from lower right
        points[i] = {{xoff - shear + face.hx_lo, yoff - face.hy},
                     {xoff + shear + face.hx_hi, yoff + face.hy},
                     {xoff + shear - face.hx_hi, yoff + face.hy},
                     {xoff - shear - face.hx_lo, yoff - face.hy}};
    }

    return GenPrism{hz, std::move(points[0]), std::move(points[1])};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from half Z height and 1-4 vertices for top and bottom planes.
 */
GenPrism::GenPrism(real_type halfz, VecReal2 const& lo, VecReal2 const& hi)
    : hz_{halfz}, lo_{lo}, hi_{hi}
{
    CELER_VALIDATE(hz_ > 0, << "nonpositive halfheight: " << hz_);
    CELER_VALIDATE(lo_.size() >= 3,
                   << "insufficient number of vertices (" << lo_.size()
                   << ") for -z polygon");
    CELER_VALIDATE(hi_.size() == lo_.size(),
                   << "incompatible number of vertices (" << hi_.size()
                   << ") for +z polygon: expected " << lo_.size());

    // Input vertices must be arranged in the same counter/clockwise order
    // and be convex
    using detail::calc_orientation;
    constexpr auto cw = detail::Orientation::clockwise;
    constexpr auto col = detail::Orientation::collinear;
    constexpr bool allow_degen = true;
    CELER_VALIDATE(detail::is_convex(make_span(lo_), allow_degen),
                   << "-z polygon is not convex");
    CELER_VALIDATE(detail::is_convex(make_span(hi_), allow_degen),
                   << "+z polygon is not convex");

    auto lo_orient = calc_orientation(lo_[0], lo_[1], lo_[2]);
    auto hi_orient = calc_orientation(hi_[0], hi_[1], hi_[2]);
    CELER_VALIDATE(is_same_orientation(lo_orient, hi_orient, allow_degen),
                   << "-z and +z polygons have different orientations");

    if (lo_orient == col && hi_orient != col)
    {
        degen_ = Degenerate::lo;
    }
    else if (lo_orient != col && hi_orient == col)
    {
        degen_ = Degenerate::hi;
    }
    else
    {
        CELER_VALIDATE(lo_orient != col || hi_orient != col,
                       << "-z and +z polygons are both degenerate");
    }
    if (lo_orient == cw || hi_orient == cw)
    {
        // Reverse point orders so it's counterclockwise, needed for vectors to
        // point outward
        std::reverse(lo_.begin(), lo_.end());
        std::reverse(hi_.begin(), hi_.end());
    }

    // Check that sides aren't rotated more than 90 degrees
    for (auto i : range<size_type>(lo_.size()))
    {
        real_type twist_angle_cosine = this->calc_twist_cosine(i);
        auto j = (i + 1) % lo_.size();
        CELER_VALIDATE(
            twist_angle_cosine > 0,
            << "twist angle between lo (" << lo_[i] << "->" << lo_[j]
            << ") and hi (" << hi_[i] << "->" << hi_[j]
            << ") is not less than a quarter turn (actual angle: "
            << native_value_to<Turn>(std::acos(twist_angle_cosine)).value()
            << " turns)");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cosine of the twist angle for a given side.
 *
 * The index \c i is the lower left point on the face when looking from the
 * outside. The result is the dot product between the
 * rightward direction vector of the lower and upper edges. If one edge is
 * degenerate, the twist angle is zero (cosine of 1).
 */
real_type GenPrism::calc_twist_cosine(size_type i) const
{
    CELER_EXPECT(i < lo_.size());

    auto j = (i + 1) % lo_.size();
    if (lo_[i] == lo_[j] || hi_[i] == hi_[j])
    {
        // Degenerate face: top or bottom is a single point
        return 1;
    }

    auto lo = make_unit_vector(lo_[j] - lo_[i]);
    auto hi = make_unit_vector(hi_[j] - hi_[i]);

    return dot_product(lo, hi);
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void GenPrism::build(IntersectSurfaceBuilder& insert_surface) const
{
    // Build the bottom and top planes
    if (degen_ != Degenerate::lo)
    {
        insert_surface(Sense::outside, PlaneZ{-hz_});
    }
    if (degen_ != Degenerate::hi)
    {
        insert_surface(Sense::inside, PlaneZ{hz_});
    }

    /*! \todo Use plane normal equality from SoftSurfaceEqual, or maybe soft
     * equivalence on twist angle cosine?
     */
    SoftEqual soft_equal{insert_surface.tol().rel};

    // Build the side planes
    for (auto i : range(lo_.size()))
    {
        auto j = (i + 1) % lo_.size();

        Real3 const ilo{lo_[i][X], lo_[i][Y], -hz_};
        Real3 const jlo{lo_[j][X], lo_[j][Y], -hz_};
        Real3 const jhi{hi_[j][X], hi_[j][Y], hz_};
        Real3 const ihi{hi_[i][X], hi_[i][Y], hz_};

        // Calculate outward normal by taking the cross product of the edges
        auto lo_normal = make_unit_vector(cross_product(jlo - ilo, ihi - ilo));
        auto hi_normal = make_unit_vector(cross_product(ihi - jhi, jlo - jhi));

        if (soft_equal(dot_product(lo_normal, hi_normal), real_type{1})
            || ihi == jhi)
        {
            // Insert a planar face
            insert_surface(
                Sense::inside, Plane{lo_normal, ilo}, "p" + std::to_string(i));
        }
        else if (ilo == jlo)
        {
            // Insert a degenerate planar face
            insert_surface(
                Sense::inside, Plane{hi_normal, ihi}, "p" + std::to_string(i));
        }
        else
        {
            // Insert a "twisted" face
            // x,y-'slopes' of i,j vertical edges in terms of z
            auto aux = 0.5 / hz_;
            auto txi = aux * (ihi[X] - ilo[X]);
            auto tyi = aux * (ihi[Y] - ilo[Y]);
            auto txj = aux * (jhi[X] - jlo[X]);
            auto tyj = aux * (jhi[Y] - jlo[Y]);

            // half-way coordinates of i,j vertical edges
            auto mxi = 0.5 * (ilo[X] + ihi[X]);
            auto myi = 0.5 * (ilo[Y] + ihi[Y]);
            auto mxj = 0.5 * (jlo[X] + jhi[X]);
            auto myj = 0.5 * (jlo[Y] + jhi[Y]);

            // coefficients for the quadric
            real_type czz = txj * tyi - txi * tyj;
            real_type eyz = txi - txj;
            real_type fzx = tyj - tyi;
            real_type gx = myj - myi;
            real_type hy = mxi - mxj;
            real_type iz = txj * myi - txi * myj + tyi * mxj - tyj * mxi;
            real_type js = mxj * myi - mxi * myj;

            insert_surface(
                Sense::inside,
                GeneralQuadric{{0, 0, czz}, {0, eyz, fzx}, {gx, hy, iz}, js},
                "t" + std::to_string(i));
        }
    }

    // Construct exterior bounding box
    BBox exterior_bbox;
    for (VecReal2 const* p : {&lo_, &hi_})
    {
        for (Real2 const& xy : *p)
        {
            for (auto ax : {Axis::x, Axis::y})
            {
                exterior_bbox.grow(ax, xy[to_int(ax)]);
            }
        }
    }
    exterior_bbox.grow(Bound::lo, Axis::z, -hz_);
    exterior_bbox.grow(Bound::hi, Axis::z, hz_);
    insert_surface(Sense::inside, exterior_bbox);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void GenPrism::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// INFSLAB
//---------------------------------------------------------------------------//
/*!
 * Construct from lower and upper z-planes.
 */
InfSlab::InfSlab(real_type lower, real_type upper)
    : lower_{lower}, upper_{upper}
{
    CELER_VALIDATE(lower_ < upper_,
                   << "invalid z planes, lower plane z value " << lower_
                   << " must be less than upper plane z value" << upper_);
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void InfSlab::build(IntersectSurfaceBuilder& insert_surface) const
{
    insert_surface(Sense::outside, PlaneZ{lower_});
    insert_surface(Sense::inside, PlaneZ{upper_});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void InfSlab::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// INFWEDGE
//---------------------------------------------------------------------------//
/*!
 * Construct from a starting angle and interior angle.
 */
InfWedge::InfWedge(Turn start, Turn interior)
    : start_{start}, interior_{interior}
{
    CELER_VALIDATE(start_ >= zero_quantity() && start_ < Turn{1},
                   << "invalid start angle " << start_.value()
                   << " [turns]: must be in the range [0, 1)");
    CELER_VALIDATE(interior_ > zero_quantity() && interior_ <= Turn{0.5},
                   << "invalid interior wedge angle " << interior.value()
                   << " [turns]: must be in the range (0, 0.5]");
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 *
 * Both planes should point "outward" to the wedge. In the degenerate case of
 * interior = 0.5 we rely on CSG object deduplication.
 */
void InfWedge::build(IntersectSurfaceBuilder& insert_surface) const
{
    real_type sinstart, cosstart, sinend, cosend;
    sincos(start_, &sinstart, &cosstart);
    sincos(start_ + interior_, &sinend, &cosend);

    insert_surface(Sense::inside, Plane{Real3{sinstart, -cosstart, 0}, 0.0});
    insert_surface(Sense::outside, Plane{Real3{sinend, -cosend, 0}, 0.0});

    //! \todo Restrict bounding boxes, at least eliminating two quadrants...
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void InfWedge::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// Involute
//---------------------------------------------------------------------------//
/*!
 * Construct with prarameters and half height.
 */
Involute::Involute(Real3 const& radii,
                   Real2 const& displacement,
                   Chirality sign,
                   real_type halfheight)
    : radii_(radii), a_(displacement), t_bounds_(), sign_(sign), hh_{halfheight}
{
    CELER_VALIDATE(radii_[0] > 0,
                   << "nonpositive involute radius: " << radii_[0]);
    CELER_VALIDATE(radii_[1] > radii_[0],
                   << "inner cylinder radius " << radii_[1]
                   << " is not greater than involute radius " << radii_[0]);
    CELER_VALIDATE(radii_[2] > radii_[1],
                   << "outer cylinder radius " << radii_[2]
                   << " is not greater than inner cyl radius " << radii_[1]);

    CELER_VALIDATE(a_[1] > a_[0],
                   << "nonpositive delta displacment: " << a_[1] - a_[0]);
    CELER_VALIDATE(hh_ > 0, << "nonpositive half-height: " << hh_);

    for (auto i : range(2))
    {
        t_bounds_[i] = std::sqrt(
            clamp_to_nonneg(ipow<2>(radii_[i + 1] / radii_[0]) - 1));
    }
    auto outer_isect = t_bounds_[0] + 2 * constants::pi - (a_[1] - a_[0]);
    CELER_VALIDATE(t_bounds_[1] < outer_isect,
                   << "radial bounds result in angular overlap: "
                   << outer_isect - t_bounds_[1]);
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Involute::build(IntersectSurfaceBuilder& insert_surface) const
{
    using InvSurf = ::celeritas::Involute;

    insert_surface(Sense::outside, PlaneZ{-hh_});
    insert_surface(Sense::inside, PlaneZ{hh_});
    insert_surface(Sense::outside, CCylZ{radii_[1]});
    insert_surface(Sense::inside, CCylZ{radii_[2]});
    // Make an inside and outside involute
    Real2 const xy{0, 0};
    auto sense = (sign_ == Chirality::right ? Sense::outside : Sense::inside);
    static char const* names[] = {"invl", "invr"};

    for (auto i : range(2))
    {
        insert_surface(sense,
                       InvSurf{xy,
                               radii_[0],
                               eumod(a_[i], real_type(2 * constants::pi)),
                               sign_,
                               t_bounds_[0],
                               t_bounds_[1] + a_[1] - a_[0]},
                       std::string{names[i]});
        sense = flip_sense(sense);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Involute::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
// PARALLELEPIPED
//---------------------------------------------------------------------------//
/*!
 * Construct with a 3-vector of half-edges and three angles.
 */
Parallelepiped::Parallelepiped(Real3 const& half_projs,
                               Turn alpha,
                               Turn theta,
                               Turn phi)
    : hpr_{half_projs}, alpha_{alpha}, theta_{theta}, phi_{phi}
{
    for (auto ax : range(Axis::size_))
    {
        CELER_VALIDATE(hpr_[to_int(ax)] > 0,
                       << "nonpositive half-edge - roughly along "
                       << to_char(ax) << " axis: " << hpr_[to_int(ax)]);
    }

    CELER_VALIDATE(alpha_ > -Turn{0.25} && alpha_ < Turn{0.25},
                   << "invalid angle " << alpha_.value()
                   << " [turns]: must be in the range (-0.25, 0.25)");
    CELER_VALIDATE(theta_ >= zero_quantity() && theta_ < Turn{0.25},
                   << "invalid angle " << theta_.value()
                   << " [turns]: must be in the range [0, 0.25)");
    CELER_VALIDATE(phi_ >= zero_quantity() && phi_ < Turn{1.},
                   << "invalid angle " << phi_.value()
                   << " [turns]: must be in the range [0, 1)");
}

//---------------------------------------------------------------------------//
/*!
 * Build surfaces.
 */
void Parallelepiped::build(IntersectSurfaceBuilder& insert_surface) const
{
    // Cache trigonometric values
    real_type sinth, costh, sinphi, cosphi, sinal, cosal;
    sincos(theta_, &sinth, &costh);
    sincos(phi_, &sinphi, &cosphi);
    sincos(alpha_, &sinal, &cosal);

    // Base vectors
    auto a = hpr_[X] * Real3{1, 0, 0};
    auto b = hpr_[Y] * Real3{sinal, cosal, 0};
    auto c = hpr_[Z] * Real3{sinth * cosphi, sinth * sinphi, costh};

    // Position the planes
    auto xnorm = make_unit_vector(cross_product(b, c));
    auto ynorm = make_unit_vector(cross_product(c, a));
    auto xoffset = dot_product(a, xnorm);
    auto yoffset = dot_product(b, ynorm);

    // Build top and bottom planes
    insert_surface(Sense::outside, PlaneZ{-hpr_[Z]});
    insert_surface(Sense::inside, PlaneZ{hpr_[Z]});

    // Build the side planes roughly perpendicular to y-axis
    insert_surface(Sense::outside, Plane{ynorm, -yoffset});
    insert_surface(Sense::inside, Plane{ynorm, yoffset});

    // Build the side planes roughly perpendicular to x-axis
    insert_surface(Sense::outside, Plane{xnorm, -xoffset});
    insert_surface(Sense::inside, Plane{xnorm, xoffset});

    // Add an exterior bounding box
    auto half_diagonal = a + b + c;
    insert_surface(Sense::inside, BBox{-half_diagonal, half_diagonal});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Parallelepiped::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
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
void Prism::build(IntersectSurfaceBuilder& insert_surface) const
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
    real_type const delta_rad = 2 * pi / real_type(num_sides_);

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
                   make_xyradial_bbox(apothem_ / std::cos(delta_rad / 2)));

    auto interior_bbox = make_xyradial_bbox(apothem_);
    interior_bbox.shrink(Bound::lo, Axis::z, -hh_);
    interior_bbox.shrink(Bound::hi, Axis::z, hh_);
    insert_surface(Sense::outside, interior_bbox);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Prism::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
/*!
 * Whether this encloses another sphere.
 */
bool Prism::encloses(Prism const& other) const
{
    if (num_sides_ != other.num_sides_ || orientation_ != other.orientation_)
    {
        CELER_NOT_IMPLEMENTED(
            "hollow prism unless number of sides and orientation are "
            "identical");
    }
    return apothem_ >= other.apothem() && hh_ >= other.halfheight();
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
void Sphere::build(IntersectSurfaceBuilder& insert_surface) const
{
    insert_surface(SphereCentered{radius_});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Sphere::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
/*!
 * Whether this encloses another sphere.
 */
bool Sphere::encloses(Sphere const& other) const
{
    return radius_ >= other.radius();
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
