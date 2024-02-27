//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexRegion.hh
//! \brief Contains ConvexRegionInterface and concrete daughters
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class ConvexSurfaceBuilder;

//---------------------------------------------------------------------------//
/*!
 * Interface class for building non-reentrant spatial regions.
 *
 * This is a building block for constructing more complex objects out of
 * smaller spatial regions. A \c shape object will have a single convex region,
 * and a \c solid object region may have multiple adjacent convex regions.
 *
 * Convex regions should be as minimal as possible and rely on transformations
 * to change axes, displacement, etc. As a general rule, the exterior bounding
 * box of a convex region should be <em>centered on the origin</em>, and
 * objects should be aligned along the \em z axis.
 *
 * When implementing this class, prefer to build simpler surfaces (planes)
 * before complex ones (cones) in case we implement short-circuiting logic,
 * since expressions are currently sorted.
 *
 * \note Additional methods such as volume calculation may be added here later.
 */
class ConvexRegionInterface
{
  public:
    //! Construct surfaces that are AND-ed into this region
    virtual void build(ConvexSurfaceBuilder&) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    ConvexRegionInterface() = default;
    virtual ~ConvexRegionInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ConvexRegionInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * A rectangular parallelepiped/cuboid centered on the origin.
 */
class Box final : public ConvexRegionInterface
{
  public:
    // Construct with half-widths
    explicit Box(Real3 const& halfwidths);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

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
 * This convex region, along with the Cylinder, is a base component of the
 * G4Polycone (PCON).
 */
class Cone final : public ConvexRegionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Real2 = Array<real_type, 2>;
    //!@}

  public:
    // Construct with Z half-height and lo, hi radii
    Cone(Real2 const& radii, real_type halfheight);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

  private:
    Real2 radii_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * A Z-aligned cylinder centered on the origin.
 */
class Cylinder final : public ConvexRegionInterface
{
  public:
    // Construct with radius
    Cylinder(real_type radius, real_type halfheight);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

  private:
    real_type radius_;
    real_type hh_;
};

//---------------------------------------------------------------------------//
/*!
 * An axis-alligned ellipsoid centered at the origin.
 */
class Ellipsoid final : public ConvexRegionInterface
{
  public:
    // Construct with radius
    explicit Ellipsoid(Real3 const& radii);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

  private:
    Real3 radii_;
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
class Prism final : public ConvexRegionInterface
{
  public:
    // Construct with inner radius (apothem), half height, and orientation
    Prism(int num_sides,
          real_type apothem,
          real_type halfheight,
          real_type orientation);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

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
 */
class Sphere final : public ConvexRegionInterface
{
  public:
    // Construct with radius
    explicit Sphere(real_type radius);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

  private:
    real_type radius_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
