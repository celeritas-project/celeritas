//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexRegion.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class ConvexSurfaceBuilder;

//---------------------------------------------------------------------------//
/*!
 * A non-reentrant volume of space for building more complex objects.
 */
class ConvexRegion
{
  public:
    //! Construct surfaces that are AND-ed into this region
    virtual void build(ConvexSurfaceBuilder&) const = 0;

    // TODO: additional (optional?) bounding box constraints/update
    // TODO: volume calculation?

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    ConvexRegion() = default;
    virtual ~ConvexRegion() = default;
    CELER_DEFAULT_COPY_MOVE(ConvexRegion);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * A rectangular parallelepiped/cuboid centered on the origin.
 */
class Box final : public ConvexRegion
{
  public:
    // Construct with half-widths
    explicit Box(Real3 halfwidths);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

  private:
    Real3 hw_;
};

//---------------------------------------------------------------------------//
/*!
 * A truncated cone along the Z axis centered on the origin.
 *
 * The midpoint along the Z axis of the cone is the origin. A cone is *not*
 * allowed to have equal radii: for that, use a cylinder. This, along with the
 * Cylinder, is a base component of the G4Polycone (PCON).
 */
class Cone final : public ConvexRegion
{
  public:
    //!@{
    //! \name Type aliases
    using Real2 = Array<real_type, 2>;
    //!@}

  public:
    // Construct with Z halfwidth and lo, hi radii
    Cone(Real2 radii, real_type halfheight);

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
class Cylinder final : public ConvexRegion
{
  public:
    // Construct with radius
    explicit Cylinder(real_type radius, real_type halfheight);

    // Build surfaces
    void build(ConvexSurfaceBuilder&) const final;

  private:
    real_type radius_;
    real_type hh_;
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
 * unity replicates the original shape but with faces indexed by one. The usual
 * value for this is 0.5:
 * - n=4 is a diamond
 * - n=6 is a pointy-top hexagon
 */
class Prism final : public ConvexRegion
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
class Sphere final : public ConvexRegion
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
