//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectRegion.hh
//! \brief Contains IntersectRegionInterface and concrete daughters
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Turn.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
struct JsonPimpl;

namespace orangeinp
{
class IntersectSurfaceBuilder;

//---------------------------------------------------------------------------//
/*!
 * Interface class for building non-reentrant spatial regions.
 *
 * This is a building block for constructing more complex objects out of
 * smaller spatial regions. A \c shape object will have a single intersect
 * region, and a \c solid object region may have multiple adjacent intersect
 * regions.
 *
 * Convex regions should be as minimal as possible and rely on transformations
 * to change axes, displacement, etc. As a general rule, the exterior bounding
 * box of a intersect region should be <em>centered on the origin</em>, and
 * objects should be aligned along the \em z axis.
 *
 * When implementing this class, prefer to build simpler surfaces (planes)
 * before complex ones (cones) in case we implement short-circuiting logic,
 * since expressions are currently sorted.
 *
 * \note Additional methods such as volume calculation may be added here later.
 */
class IntersectRegionInterface
{
  public:
    //! Construct surfaces that are AND-ed into this region
    virtual void build(IntersectSurfaceBuilder&) const = 0;

    //! Write the region to a JSON object
    virtual void output(JsonPimpl*) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    IntersectRegionInterface() = default;
    virtual ~IntersectRegionInterface() = default;
    CELER_DEFAULT_COPY_MOVE(IntersectRegionInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * A rectangular parallelepiped/cuboid centered on the origin.
 */
class Box final : public IntersectRegionInterface
{
  public:
    // Construct with half-widths
    explicit Box(Real3 const& halfwidths);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Half-width for each axis
    Real3 const& halfwidths() const { return hw_; }

  private:
    Real3 hw_;
};

//---------------------------------------------------------------------------//
/*!
 * A closed cone along the Z axis centered on the origin.
 *
 * A quadric cone technically defines two opposing cones that touch at a single
 * vanishing point, but this cone is required to be truncated so that the
 * vanishing point is on our outside the cone.
 *
 * The midpoint along the Z axis of the cone is the origin. A cone is \em not
 * allowed to have equal radii: for that, use a cylinder. However, it \em may
 * have a single radius of zero, which puts the vanishing point on one end of
 * the cone.
 *
 * This intersect region, along with the Cylinder, is a base component of the
 * G4Polycone (PCON).
 */
class Cone final : public IntersectRegionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Real2 = Array<real_type, 2>;
    //!@}

  public:
    // Construct with Z half-height and lo, hi radii
    Cone(Real2 const& radii, real_type halfheight);

    //// INTERFACE ////

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another cone
    bool encloses(Cone const& other) const;

    //// ACCESSORS ////

    //! Lower and upper radii
    Real2 const& radii() const { return radii_; }

    //! Half-height along Z
    real_type halfheight() const { return hh_; }

  private:
    Real2 radii_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * A Z-aligned cylinder centered on the origin.
 */
class Cylinder final : public IntersectRegionInterface
{
  public:
    // Construct with radius
    Cylinder(real_type radius, real_type halfheight);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another cylinder
    bool encloses(Cylinder const& other) const;

    //// ACCESSORS ////

    //! Radius
    real_type radius() const { return radius_; }

    //! Half-height along Z
    real_type halfheight() const { return hh_; }

  private:
    real_type radius_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * An axis-alligned ellipsoid centered at the origin.
 */
class Ellipsoid final : public IntersectRegionInterface
{
  public:
    // Construct with radius
    explicit Ellipsoid(Real3 const& radii);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Radius along each axis
    Real3 const& radii() const { return radii_; }

  private:
    Real3 radii_;
};

//---------------------------------------------------------------------------//
/*!
 * A generalized polygon with flat faces along the z axis.
 *
 * A GenTrap, like VecGeom's equivalent and ROOT's Arb8, represents a general
 * trapezoidal volume with up to eight vertices, or two 4-point sets, sitting
 * on two parallel planes perpendicular to Z axis.
 *
 * TODO: Add proper treatment for degenerate cases.
 */
class GenTrap final : public IntersectRegionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Real2 = Array<real_type, 2>;
    using VecReal2 = std::vector<Real2>;
    //!@}

    //! Regular trapezoidal top/bottom face
    struct TrapFace
    {
        //! Half the vertical distance between horizontal edges
        real_type hy{};
        //! Top horizontal edge half-length
        real_type hx_lo{};
        //! Bottom horizontal edge half-length
        real_type hx_hi{};
        //! Tangent of shear angle, between horizontal line centers and Y axis
        real_type tan_alpha{};
    };

  public:
    // Helper function to construct a Trd shape from hz and two rectangles,
    // one for each z-face
    static GenTrap from_trd(real_type halfz, Real2 const& lo, Real2 const& hi);

    // Helper function to construct a general trap from its half-height and
    // the two trapezoids defining its lower and upper faces
    static GenTrap from_trap(real_type hz,
                             Turn theta,
                             Turn phi,
                             TrapFace const& lo,
                             TrapFace const& hi);

    // Construct from half Z height and 4 vertices for top and bottom planes
    GenTrap(real_type halfz, VecReal2 const& lo, VecReal2 const& hi);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Half-height along Z
    real_type halfheight() const { return hz_; }

    //! Polygon on -z face
    VecReal2 const& lower() const { return lo_; }
    //! Polygon on +z face
    VecReal2 const& upper() const { return hi_; }

  private:
    real_type hz_;  //!< half-height
    VecReal2 lo_;  //!< corners of the -z face
    VecReal2 hi_;  //!< corners of the +z face
};

//---------------------------------------------------------------------------//
/*!
 * An open wedge shape from the Z axis.
 *
 * The wedge is defined by an interior angle that *must* be less than or equal
 * to 180 degrees (half a turn) and *must* be more than zero. It can be
 * subtracted, or its negation can be subtracted. The start angle is mapped
 * onto [0, 1) on construction.
 */
class InfWedge final : public IntersectRegionInterface
{
  public:
    // Construct from a starting angle and interior angle
    InfWedge(Turn start, Turn interior);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Starting angle
    Turn start() const { return start_; }

    //! Interior angle
    Turn interior() const { return interior_; }

  private:
    Turn start_;
    Turn interior_;
};

//---------------------------------------------------------------------------//
/*!
 * A general parallelepiped centered on the origin.
 *
 * A parallelepiped is a shape having 3 pairs of parallel faces out of
 * which one is parallel with the XY plane (Z faces). All faces are
 * parallelograms in the general case. The Z faces have 2 edges parallel
 * with the X-axis. Note that all angle parameters are expressed in terms
 * of fractions of a 360deg turn.
 *
 * The shape has the center in the origin and it is defined by:
 *
 *   - `halfedges:` a 3-vector (dY, dY, dZ) with half-lengths of the
 *     projections of the edges on X, Y, Z. The lower Z face is positioned at
 *     `-dZ`, and the upper one at `+dZ`.
 *   - `alpha:` angle between the segment defined by the centers of the
 *     X-parallel edges and Y axis. Validity range is `(-1/4, 1/4)`;
 *   - `theta:` polar angle of the shape's main axis, e.g. the segment defined
 *     by the centers of the Z faces. Validity range is `[0, 1/4)`;
 *   - `phi:` azimuthal angle of the shape's main axis (as explained above).
 *     Validity range is `[0, 1)`.
 */
class Parallelepiped final : public IntersectRegionInterface
{
  public:
    // Construct with half widths and 3 angles
    Parallelepiped(Real3 const& halfedges, Turn alpha, Turn theta, Turn phi);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Half-lengths of edge projections along each axis
    Real3 const& halfedges() const { return hpr_; }
    //! Angle between slanted y-edges and the y-axis (in turns)
    Turn alpha() const { return alpha_; }
    //! Polar angle of main axis (in turns)
    Turn theta() const { return theta_; }
    //! Azimuthal angle of main axis (in turns)
    Turn phi() const { return phi_; }

  private:
    // half-lengths
    Real3 hpr_;
    // angle between slanted y-edges and the y-axis
    Turn alpha_;
    // polar angle of main axis
    Turn theta_;
    // azimuthal angle of main axis
    Turn phi_;
};

//---------------------------------------------------------------------------//
/*!
 * A regular, z-extruded polygon centered on the origin.
 *
 * This is the base component of a G4Polyhedra (PGON). The default rotation is
 * to put a y-aligned plane on the bottom of the shape, so looking at an x-y
 * slice given an apothem \em a, every shape has a surface at \f$ y = -a \f$:
 * - n=3 is a triangle with a flat bottom, point up
 * - n=4 is a square with axis-aligned sides
 * - n=6 is a flat-top hexagon
 *
 * The "orientation" parameter is a scaled counterclockwise rotation on
 * \f$[0, 1)\f$, where zero preserves the orientation described above, and
 * unity replicates the original shape but with the "p0" face being where the
 * "p1" originally was. With a value of 0.5:
 * - n=3 is a downward-pointing triangle
 * - n=4 is a diamond
 * - n=6 is a pointy-top hexagon
 */
class Prism final : public IntersectRegionInterface
{
  public:
    // Construct with inner radius (apothem), half height, and orientation
    Prism(int num_sides,
          real_type apothem,
          real_type halfheight,
          real_type orientation);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Number of sides
    int num_sides() const { return num_sides_; }
    //! Inner radius
    real_type apothem() const { return apothem_; }
    //! Half the Z height
    real_type halfheight() const { return hh_; }
    //! Rotation factor
    real_type orientation() const { return orientation_; }

  private:
    // Number of sides
    int num_sides_;

    // Distance from center to midpoint of its side
    real_type apothem_;

    // Half the z height
    real_type hh_;

    // Rotational offset (0 has bottom face at -Y, 1 is congruent)
    real_type orientation_;
};

//---------------------------------------------------------------------------//
/*!
 * A sphere centered on the origin.
 *
 * \note Be aware there's also a sphere *surface* at orange/surf/Sphere.hh in a
 * different namespace.
 */
class Sphere final : public IntersectRegionInterface
{
  public:
    // Construct with radius
    explicit Sphere(real_type radius);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another sphere
    bool encloses(Sphere const& other) const;

    //// ACCESSORS ////

    //! Radius
    real_type radius() const { return radius_; }

  private:
    real_type radius_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
