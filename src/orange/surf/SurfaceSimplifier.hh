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
    inline SurfaceSimplifier(real_type tol, Sense* s);

    //! Construct with reference to sense that may be flipped
    explicit inline SurfaceSimplifier(Sense* s) : SurfaceSimplifier{1e-10, s}
    {
    }

    // Plane may be snapped to origin
    template<Axis T>
    Optional<PlaneAligned<T>> operator()(PlaneAligned<T> const&) const;

    // Cylinder at origin will be simplified
    template<Axis T>
    Optional<CylCentered<T>> operator()(CylAligned<T> const&) const;

    // Plane may be flipped, adjusted, or become axis-aligned
    Optional<Plane> operator()(Plane const&);

    // Sphere near center can be snapped
    Optional<SphereCentered> operator()(Sphere const&) const;

    // Simple quadric with near-zero terms can be another second-order surface
    Optional<Sphere,
             ConeAligned<Axis::x>,
             ConeAligned<Axis::y>,
             ConeAligned<Axis::z>,
             CylAligned<Axis::x>,
             CylAligned<Axis::y>,
             CylAligned<Axis::z>>
    operator()(SimpleQuadric const&);

    // Quadric with no cross terms is simple
    Optional<SimpleQuadric> operator()(GeneralQuadric const&);

    //! Default: no simplifcation
    template<class S>
    std::variant<std::monostate> operator()(S const&)
    {
        return {};
    }

  private:
    real_type tol_;
    Sense* sense_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with snapping tolerance and reference to sense.
 */
SurfaceSimplifier::SurfaceSimplifier(real_type tol, Sense* s)
    : tol_{tol}, sense_{s}
{
    CELER_EXPECT(tol_ >= 0);
    CELER_EXPECT(sense_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
