//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 * Return a simplified version of a surface, possibly flipping associated
 * sense.
 *
 * This class is meant for using SurfaceAction or std::variant to visit a
 * surface type.
 *
 * The result of each simplification type should be a \c std::variant of
 * possible simplified class forms, or a \c std::monostate if no simplification
 * was applied.
 *
 * The embedded sense may be flipped as part of the simplifications.
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
    operator()(Plane const&);

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
    operator()(SimpleQuadric const&);

    // Quadric with no cross terms is simple
    Optional<SimpleQuadric> operator()(GeneralQuadric const&);

    //! Default: no simplifcation
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
