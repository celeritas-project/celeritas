//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

#include "corecel/Assert.hh"
#include "orange/OrangeTypes.hh"

#include "SurfaceFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a simplified, regularized version of a surface/sense pair.
 *
 * This class takes a general surface with an associated sense and will
 * simplify (e.g., turning a general plane into an axis-aligned one) and
 * regularize (e.g., flipping normals so that the plane points in a positive
 * direction) it, modifying the sense as needed.
 *
 * It is meant to be used with \c VariantSurface to visit a surface type.
 *
 * The result of each simplification type should be a \c std::variant of
 * possible simplified class forms, or a \c std::monostate if no simplification
 * was applied.
 *
 * \internal \todo Use a \c Tolerance object instead of a single tolerance? (Or
 * use local length scale with \c SoftClose.) Also, compare implementations
 * with \c SoftSurfaceEqual for consistency.
 */
class SurfaceSimplifier
{
  private:
    template<class... T>
    using Optional = std::variant<std::monostate, T...>;

  public:
    // Construct with snapping tolerance and reference to sense
    inline SurfaceSimplifier(Sense* s, real_type tol);

    //! Construct with reference to sense that may be flipped
    explicit inline SurfaceSimplifier(Sense* s) : SurfaceSimplifier{s, 1e-10}
    {
    }

    // Plane may be snapped to origin
    template<Axis T>
    Optional<PlaneAligned<T>> operator()(PlaneAligned<T> const&) const;

    // Cylinder at origin will be simplified
    template<Axis T>
    Optional<CylCentered<T>> operator()(CylAligned<T> const&) const;

    // Cone near origin will be snapped
    template<Axis T>
    Optional<ConeAligned<T>> operator()(ConeAligned<T> const&) const;

    // Plane may be flipped, adjusted, or become axis-aligned
    Optional<PlaneAligned<Axis::x>, PlaneAligned<Axis::y>, PlaneAligned<Axis::z>, Plane>
    operator()(Plane const&) const;

    // Sphere near center can be snapped
    Optional<SphereCentered> operator()(Sphere const&) const;

    // Simple quadric can be normalized or simplified
    Optional<Plane,
             Sphere,
             CylAligned<Axis::x>,
             CylAligned<Axis::y>,
             CylAligned<Axis::z>,
             ConeAligned<Axis::x>,
             ConeAligned<Axis::y>,
             ConeAligned<Axis::z>,
             SimpleQuadric>
    operator()(SimpleQuadric const&) const;

    // Quadric can be normalized or simplified
    Optional<SimpleQuadric, GeneralQuadric>
    operator()(GeneralQuadric const&) const;

    //! Default: no simplification
    template<class S>
    std::variant<std::monostate> operator()(S const&) const
    {
        return {};
    }

  private:
    Sense* sense_;
    real_type tol_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with snapping tolerance and reference to sense.
 */
SurfaceSimplifier::SurfaceSimplifier(Sense* s, real_type tol)
    : sense_{s}, tol_{tol}
{
    CELER_EXPECT(sense_);
    CELER_EXPECT(tol_ >= 0);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
