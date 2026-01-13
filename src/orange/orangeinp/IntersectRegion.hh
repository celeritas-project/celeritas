//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectRegion.hh
//! \brief Contains IntersectRegionInterface and concrete daughters
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/grid/GridTypes.hh"
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

    virtual ~IntersectRegionInterface() = default;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    IntersectRegionInterface() = default;
    CELER_DEFAULT_COPY_MOVE(IntersectRegionInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * A rectangular parallelepiped/cuboid centered on the origin.
 *
 * The box is constructed with half-widths.
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
 * A closed truncated cone along the *z* axis centered on the origin.
 *
 * A quadric cone technically defines two opposing cones that touch at a single
 * vanishing point, but this cone is required to be truncated so that the
 * vanishing point is on our outside the cone.
 *
 * The midpoint along the \em z axis of the cone is the origin. A cone is \em
 * not allowed to have equal radii: for that, use a cylinder. However, it \em
 * may have a single radius of zero, which puts the vanishing point on one end
 * of the cone.
 *
 * This intersect region, along with the Cylinder, is a base component of the
 * G4Polycone (PCON).
 *
 * \note The Cone is allowed to be "degenerate" in the sense of
 * having nearly equal lower and upper radii. It will construct a cylinder with
 * an average of the two radii.
 */
class Cone final : public IntersectRegionInterface
{
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
 * A *z*-aligned cylinder centered on the origin, with top/bottom cuts.
 *
 * The shape is defined with a radius, half-height, and the outward-facing
 * normals of the cutting planes, passing through \f$(0, 0, \pm
 * \mathrm{hh})\f$.
 */
class CutCylinder final : public IntersectRegionInterface
{
  public:
    // Construct with radius, half-height, and bottom/top cut plane normals
    CutCylinder(real_type radius,
                real_type halfheight,
                Real3 const& bottom_normal,
                Real3 const& top_normal);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another cut cylinder
    bool encloses(CutCylinder const& other) const;

    //// ACCESSORS ////

    //! Radius
    real_type radius() const { return radius_; }

    //! Half-height along Z
    real_type halfheight() const { return hh_; }

    //! Outward-facing normal of the bottom cuting plane
    Real3 const& bottom_normal() const { return bot_normal_; }

    //! Outward-facing normal of the top cuting plane
    Real3 const& top_normal() const { return top_normal_; }

  private:
    real_type radius_;
    real_type hh_;
    Real3 bot_normal_;
    Real3 top_normal_;
};

//---------------------------------------------------------------------------//
/*!
 * A *z*-aligned cylinder centered on the origin.
 *
 * The cylinder is defined with a radius and half-height.
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
 * An axis-aligned ellipsoid centered at the origin.
 *
 * The ellipsoid is constructed with the three radial lengths. For a length
 * scale \em L , the quadric it creates has second-order terms that are \f$
 * O(1) \f$ and a zeroth order term that's \f$ O(L^2) \f$. Translations on that
 * length scale will preserve the accuracy of the quadratic solution.
 *
 * There are many scalings of the quadric equation that produce unitary
 * second-order terms if the ellipsoid's radii are identical: \f[
  \frac{k}{r_x^2} x^2 +  \frac{k}{r_y^2} y^2 +  \frac{k}{r_z^2} z^2 = k
 * \f]
 * but we make the ad hoc decision to choose \f$ k = \min r_i \max r_i \f$
 * to avoid irrational normalization constants, which makes unit tests and
 * output easier to read.
 */
class Ellipsoid final : public IntersectRegionInterface
{
  public:
    // Construct with radius along each Cartesian axis
    explicit Ellipsoid(Real3 const& radii);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another ellipsoid
    bool encloses(Ellipsoid const& other) const;

    //// ACCESSORS ////

    //! Radius along each axis
    Real3 const& radii() const { return radii_; }

    //! Get the radius along a single axis
    real_type radius(Axis ax) const { return radii_[to_int(ax)]; }

  private:
    Real3 radii_;
};

//---------------------------------------------------------------------------//
/*!
 * A *z*-aligned cylinder with an elliptical cross section.
 *
 * The elliptical cylinder is defined with a two radii and a half-height,
 * such that the centroid of the bounding box is origin. The quadric
 * coefficient of the cylindrical component, \f[
  x^2 / r_x^2 + y^2 / r_y^2 = 1 \,
  \f]
 * are scaled based on the radii of the cylinder, reducing to
 * \f$ x^2 + y^2 = R^2 \f$ when the radii are equal.
 */
class EllipticalCylinder final : public IntersectRegionInterface
{
  public:
    // Construct with x- and y-radii and half-height in z
    EllipticalCylinder(Real2 const& radii, real_type halfheight);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another ellipsoid
    bool encloses(EllipticalCylinder const& other) const;

    //// ACCESSORS ////

    //! Radius along each axis
    Real2 const& radii() const { return radii_; }

    //! Half-height along Z
    real_type halfheight() const { return hh_; }

    // Get the radius along the x/y axis
    real_type radius(Axis ax) const;

  private:
    Real2 radii_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * A finite \em z-aligned cone with an elliptical cross section.
 *
 * The elliptical cone is defined in an analogous fashion to the regular
 * (i.e., circular) cone. A half-height (hh) defines the \em z extents, such
 * that the centroid of the outer bounding box is the origin. The lower radii
 * are the \em x  and \em y radii at the plane \f$ z = -\mathrm{hh} \f$.
 * The upper radii are the \em x and \em y radii at the plane
 * \f$ z = \mathrm{hh} \f$. There are several restrictions on these radii:
 *
 * - Either the lower or upper radii may be \f$(0, 0)\f$; this is the only
 *   permitted way for the elliptical cone to include the vertex.
 * - The aspect ratio of the elliptical cross sections is constant. Thus, the
 *   upper radii must be a constant scalar times the upper radii.
 * - Degenerate elliptical cones (lower_radii == upper_radii, i.e.,
 *   elliptical cylinders) are not permitted.
 * - Degenerate elliptical cones where lower or upper radii are equal to
 *   \f$(0, x)\f$ or \f$(x, 0)\f$, where \em x is non-zero, are not permitted.
 *
 * The elliptical surface can be expressed as
 * \f[
   (x/r_x)^2 + (y/r_y)^2 = (v-z)^2,
 * \f]
 * where \em v is the location of the vertex.
 *
 * The \f$ r_x \f$, \f$r_y \f$, and \f$v\f$ can be calculated from the lower
 * and upper radii as given by \c G4EllipticalCone:
 * \verbatim
   r_x = (lower_radii[X] - upper_radii[X])/(2 hh),
   r_y = (lower_radii[Y] - upper_radii[Y])/(2 hh),
     v = hh (lower_radii[X] + upper_radii[X])
         / (lower_radii[X] - upper_radii[X]).
   \endverbatim
 */
class EllipticalCone final : public IntersectRegionInterface
{
  public:
    // Construct with x- and y-radii and half-height in z
    EllipticalCone(Real2 const& lower_radii,
                   Real2 const& upper_radii,
                   real_type halfheight);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another elliptical cone
    bool encloses(EllipticalCone const& other) const;

    //// ACCESSORS ////

    //! Radii along the x- and y-axes at z=-hh
    Real2 const& lower_radii() const { return lower_radii_; }

    //! Radii along the x- and y-axes at z=-hh
    Real2 const& upper_radii() const { return upper_radii_; }

    //! Half-height along Z
    real_type halfheight() const { return hh_; }

    // Get the bottom/top radius along the x/y axis
    real_type radius(Bound b, Axis ax) const;

  private:
    Real2 lower_radii_;
    Real2 upper_radii_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * Region formed by extruding + scaling a convex polygon along a line segment.
 *
 * The convex polygon is supplied as a set of points on the \em xy plane in
 * counterclockwise order. The line segment and scaling factors are specified
 * by providing a line segment point and scaling factor for the top and bottom
 * polygon faces of the region. The line segment point of the top face must
 * have a \em z value greater than that of the bottom face. Along the line
 * segment,
 * the size of the polygon is linearly scaled in accordance with scaling
 * factors.
 *
 * As is done in Geant4, construction is done by first applying scaling factors
 * to the upper and lower polygons via scalar multiplication with each polygon
 * point, then the points on the line are used to offset the upper and lower
 * polygons.
 */
class ExtrudedPolygon final : public IntersectRegionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecReal2 = std::vector<Real2>;

    //! Specifies the top or bottom face of the ExtrudedPolygon
    struct PolygonFace
    {
        //! Start or end point of the line segment the polygon is extruded
        //! along
        Real3 line_segment_point{};

        //! The fractional amount this face is scaled
        real_type scaling_factor{};
    };
    //!@}

  public:
    // Construct from a convex polygon and bottom/top faces
    ExtrudedPolygon(VecReal2 const& polygon,
                    PolygonFace const& bot_face,
                    PolygonFace const& top_face);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Polygon points (2D)
    VecReal2 polygon() const { return polygon_; }

    //! Bottom point of the line segment
    Real3 bot_line_segment_point() const { return line_segment_[Bound::lo]; }

    //! Top point of the line segment
    Real3 top_line_segment_point() const { return line_segment_[Bound::hi]; }

    //! Bottom scaling factor
    real_type bot_scaling_factor() const
    {
        return scaling_factors_[Bound::lo];
    }

    //! Top scaling factor
    real_type top_scaling_factor() const
    {
        return scaling_factors_[Bound::hi];
    }

  private:
    //// TYPES ////
    using Range = celeritas::Array<real_type, 2>;

    //// DATA ////

    VecReal2 polygon_;

    EnumArray<Bound, Real3> line_segment_;
    EnumArray<Bound, real_type> scaling_factors_;

    Range x_range_;
    Range y_range_;

    //// HELPER FUNCTIONS ////

    // Calculate the min/max x or y values of the extruded region
    Range calc_range(VecReal2 const& polygon, size_type dim);
};

//---------------------------------------------------------------------------//
/*!
 * A generalized polygon with parallel flat faces along the *z* axis.
 *
 * A GenPrism, like VecGeom's GenTrap, ROOT's Arb8, and Geant4's
 * G4GenericTrap, represents a generalized volume with polyhedral faces on two
 * parallel planes perpendicular to the \em z axis. Unlike those other codes,
 * the number of faces can be arbitrary in number.
 *
 * The faces have an orientation and ordering so that \em twisted faces can be
 * constructed by joining corresponding points using straight-line "vertical"
 * edges, directly matching the G4GenericTrap definition, but directly
 * generating a hyperbolic paraboloid for each twisted face.
 *
 * Trapezoids constructed from the helper functions will have sides that are
 * same ordering as a prism: the rightward face is first (normal is along the
 * \em +x axis), then the others follow counterclockwise.
 */
class GenPrism final : public IntersectRegionInterface
{
  public:
    //!@{
    //! \name Type aliases
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
        //! Shear angle between horizontal line centers and Y axis
        Turn alpha;
    };

  public:
    // Helper function to construct a Trd shape from hz and two rectangles,
    // one for each z-face
    static GenPrism from_trd(real_type halfz, Real2 const& lo, Real2 const& hi);

    // Helper function to construct a general trap from its half-height and
    // the two trapezoids defining its lower and upper faces
    static GenPrism from_trap(real_type hz,
                              Turn theta,
                              Turn phi,
                              TrapFace const& lo,
                              TrapFace const& hi);

    // Construct from half Z height and 4 vertices for top and bottom planes
    GenPrism(real_type halfz, VecReal2 const& lo, VecReal2 const& hi);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Half-height along Z
    real_type halfheight() const { return hh_; }

    //! Polygon on -z face
    VecReal2 const& lower() const { return lo_; }

    //! Polygon on +z face
    VecReal2 const& upper() const { return hi_; }

    //! Number of sides (points on the Z face)
    size_type num_sides() const { return lo_.size(); }

    // Calculate the cosine of the twist angle for a given side
    real_type calc_twist_cosine(size_type size_idx) const;

  private:
    enum class Degenerate
    {
        none,
        lo,
        hi
    };

    real_type hh_;  //!< half-height
    VecReal2 lo_;  //!< corners of the -z face (CCW)
    VecReal2 hi_;  //!< corners of the +z face (CCW)
    Degenerate degen_{Degenerate::none};  //!< no plane on this z axis
    real_type length_scale_{};
};
// ...existing code...

//---------------------------------------------------------------------------//
/*!
 * A *z*-aligned hyperboloid of revolution centered on the origin.
 *
 * A hyperboloid is defined by rotating a hyperbola around the z-axis. This
 * implementation uses a minimum radius (at \f$ z=0 \f$) and a maximum radius
 * (at \f$z=\pm \textrm{hh}\f$).
 *
 * The hyperboloid surface is defined by the equation:
 * \f[
 *   x^2 + y^2 - r_{min}^2 \left(1 + z^2 t^2\right) = 0
 * \f]
 * where \f$ r_{min} \f$ is the minimum radius (at z=0) and
 * \f$ t^2 = (r_{max}^2 - r_{min}^2) / h^2 \f$, with \f$ r_{max} \f$
 * being the maximum radius and \f$ h \f$ the half-height.
 *
 * The maximum radius must be greater than the minimum radius, ensuring a valid
 * hyperboloid shape. The minimum radius must be positive, as a radius of zero
 * would produce a two-sheeted cone.
 */
class Hyperboloid final : public IntersectRegionInterface
{
  public:
    // Construct with radius at midpoint (min) and end (max), and half-height
    Hyperboloid(real_type min_radius,
                real_type max_radius,
                real_type halfheight);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another hyperboloid
    bool encloses(Hyperboloid const& other) const;

    //// ACCESSORS ////

    //! Minimum radius at z=0
    real_type min_radius() const { return r_min_; }

    //! Maximum radius at |z|=hh
    real_type max_radius() const { return r_max_; }

    //! Half-height along z
    real_type halfheight() const { return hh_; }

  private:
    real_type r_min_;  //!< Minimum radius at z=0
    real_type r_max_;  //!< Maximum radius at |z|=hh
    real_type hh_;  //!< Half-height along z
};

//---------------------------------------------------------------------------//
/*!
 * An axis-aligned infinite half-space to use for truncation operations.
 *
 * An "inside" sense means to include everything *below* the position on the
 * axis, and an "outside" sense means to include only what's *above* the
 * position.
 */
class InfPlane : public IntersectRegionInterface
{
  public:
    // Construct with sense, axis, and position
    InfPlane(Sense sense, Axis axis, real_type position);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Get the sense (inside or outside)
    Sense sense() const { return sense_; }

    //! Get the axis (x, y, or z)
    Axis axis() const { return axis_; }

    //! Get the position along the axis
    real_type position() const { return position_; }

  private:
    Sense sense_;
    Axis axis_;
    real_type position_;
};

//---------------------------------------------------------------------------//
/*!
 * An open wedge shape from the *z* axis.
 *
 * The wedge is defined by an interior angle that \em must be less than or
 * equal to 180 degrees (half a turn) and \em must be more than zero. It can be
 * subtracted, or its negation can be subtracted. The start angle is mapped
 * onto \f$[0, 1)\f$ on construction.
 */
class InfAziWedge final : public IntersectRegionInterface
{
  public:
    // Construct from an angular range less than 180
    InfAziWedge(Turn start, Turn stop);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Starting angle
    Turn start() const { return start_; }

    //! stop angle
    Turn stop() const { return stop_; }

  private:
    Turn start_;
    Turn stop_;
};

//---------------------------------------------------------------------------//
/*!
 * Select a polar (latitudinal) region.
 *
 * This uses an equatorial plane and up to two cones to slice a
 * polar-coordinate region from the origin.  A polar wedge always defines a
 * region in a single hemisphere: either \f$ z >= 0 \f$ or \f$ z <= 0 \f$,
 * corresponding to an stop range of [0, .25] turns or [0.25, 0.5] turns.
 */
class InfPolarWedge final : public IntersectRegionInterface
{
  public:
    // Construct from a starting angle and stop angle
    InfPolarWedge(Turn start, Turn stop);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Starting angle
    Turn start() const { return start_; }

    //! stop angle
    Turn stop() const { return stop_; }

  private:
    Turn start_;
    Turn stop_;

    static constexpr Turn north_pole{0};
    static constexpr Turn equator{0.25};
    static constexpr Turn south_pole{0.5};
};

//---------------------------------------------------------------------------//
/*!
 * An involute "blade" centered on the origin.
 *
 * This is the intersection of two parallel involutes with a cylindrical shell.
 * The three radii, which must be in ascending order, are that of the involute,
 * the inner cylinder, and the outer cylinder.
 *
 * The "chirality" of the involute is viewed from the \em +z axis looking down:
 * whether it spirals to the right or left.
 */
class Involute final : public IntersectRegionInterface
{
  public:
    // Construct with radius
    explicit Involute(Real3 const& radii,
                      Real2 const& displacement,
                      Chirality chirality,
                      real_type halfheight);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Radii: Rdius of involute, minimum radius, maximum radius
    Real3 const& radii() const { return radii_; }
    //! Displacement angle
    Real2 const& displacement_angle() const { return a_; }
    //!  Angular bounds of involute
    Real2 const& t_bounds() const { return t_bounds_; }
    //! Chirality of involute: turning left or right
    Chirality chirality() const { return sign_; }
    //! Halfheight
    real_type halfheight() const { return hh_; }

  private:
    Real3 radii_;
    Real2 a_;
    Real2 t_bounds_;
    Chirality sign_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * A finite *z*-aligned parabolid.
 *
 * The paraboloid is defined in an analogous fashion to the cone. A half-height
 * (hh) defines the z-extents, such that the centroid of the outer bounding box
 * is the origin. The lower and upper radii correspond to the radii at
 * \f$ z = \pm h \f$. Either the lower or upper radii may be 0, i.e.,
 * the solid may include the vertex. Degenerate cases where the lower and upper
 * radii are equal are not permitted: a cylinder should be used instead.
 *
 * A paraboloid with these properties is expressed in SimpleQuadric form as:
 * \f[
    x^2 + y^2 + \frac{(R_{\mathrm{lo}}^2 - R_{\mathrm{hi}}^2)}{h} z
    - \frac{R_{\mathrm{lo}}^2 + R_{\mathrm{hi}}^2}{2} = 0,
 * \f]
 * where \f$R_{\mathrm{lo}}\f$ and \f$R_\mathrm{hi}\f$ correspond to the lower
 * and upper radii, respectively, and \f$h\f$ is the full height. Note that the
 * scaling is such that as \f$ R_{\mathrm{lo}} \to R_{\mathrm{hi}} \f$ this
 * approaches the cylindrical equation \f$ x^2 + y^2 = R^2 \f$.
 *
 */
class Paraboloid final : public IntersectRegionInterface
{
  public:
    // Construct with lower/upper radii and the half-height
    Paraboloid(real_type lower_radius,
               real_type upper_radius,
               real_type halfheight);

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another paraboloid
    bool encloses(Paraboloid const& other) const;

    //// ACCESSORS ////

    //! Radius at z=-hh
    real_type lower_radius() const { return r_lo_; }

    //! Radius at z=hh
    real_type upper_radius() const { return r_hi_; }

    //! Half-height along Z
    real_type halfheight() const { return hh_; }

  private:
    real_type r_lo_;
    real_type r_hi_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * A general parallelepiped centered on the origin.
 *
 * A parallelepiped is a shape having 3 pairs of parallel faces out of
 * which one is parallel with the \em x-y plane (\em z faces). All faces are
 * parallelograms in the general case. The \em z faces have 2 edges parallel
 * with the \em x axis. Note that all angle parameters are expressed in terms
 * of fractions of a 360-degree turn.
 *
 * The shape has the center in the origin and it is defined by:
 *
 *   - \c halfedges: a 3-vector (dY, dY, dZ) with half-lengths of the
 *     projections of the edges on X, Y, Z. The lower Z face is positioned at
 *     `-dZ`, and the upper one at `+dZ`.
 *   - \c alpha angle between the segment defined by the centers of the
 *     X-parallel edges and Y axis. Validity range is `(-1/4, 1/4)`;
 *   - \c theta polar angle of the shape's main axis, e.g. the segment defined
 *     by the centers of the Z faces. Validity range is `[0, 1/4)`;
 *   - \c phi azimuthal angle of the shape's main axis (as explained above).
 *     Validity range is `[0, 1)`.
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
    //! Angle between slanted *y* edges and the *y* axis (in turns)
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
 * to put the first point at \f$ y = 0 \f$. Looking at an x-y
 * slice for a prism with apothem \em a, odd-numbered prisms have a plane at
 * \f$ x=-a \f$:
 * - \f$ n=3 \f$ is a triangle pointing right,
 * - \f$ n=4 \f$ is a diamond,
 * - \f$ n=5 \f$ is a tilted pentagon (flat on the left), and
 * - \f$ n=6 \f$ is a flat-top hexagon.
 *
 *
 * The "orientation" parameter is a scaled counterclockwise rotation on
 * \f$[0, 1)\f$, where zero preserves the orientation described above, and
 * unity would replicate the original shape but with the "p0" face being where
 * the "p1" originally was. With a value of 0.5:
 * - \f$ n=3 \f$ is a triangle pointing left
 * - \f$ n=4 \f$ is an upright square,
 * - \f$ n=5 \f$ is a tilted pentagon (flat on the right), and
 * - \f$ n=6 \f$ is a pointy-top hexagon.
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

    //// TEMPLATE INTERFACE ////

    // Whether this encloses another cylinder
    bool encloses(Prism const& other) const;

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

    // Rotational offset: 0 has point at (r, 0), 1 is congruent with 0
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
/*!
 * A tetrahedron defined by four vertices.
 *
 * A tetrahedron is a polyhedron with four triangular faces. The four vertices
 * should not be coplanar, and they should be ordered such that when viewed
 * from outside the tetrahedron, each triangular face has vertices in
 * counterclockwise order (following the right-hand rule for outward normals).
 *
 * The tetrahedron is constructed by defining four planes, one for each face.
 * Each plane is determined by three vertices. Face \em i uses all the vertices
 * except for the one at index \em i .
 */
class Tet final : public IntersectRegionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using ArrReal3 = Array<Real3, 4>;
    //!@}

  public:
    // Construct from an array of four vertices
    explicit Tet(ArrReal3 const&);

    //! Construct from four vertices
    Tet(Real3 const& v0, Real3 const& v1, Real3 const& v2, Real3 const& v3)
        : Tet(ArrReal3{v0, v1, v2, v3})
    {
    }

    // Build surfaces
    void build(IntersectSurfaceBuilder&) const final;

    // Output to JSON
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    // Get a vertex by index
    Real3 const& vertex(size_type i) const;

    //! Get all vertices
    ArrReal3 const& vertices() const { return v_; }

  private:
    ArrReal3 v_;  //!< Four vertices defining the tetrahedron
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
