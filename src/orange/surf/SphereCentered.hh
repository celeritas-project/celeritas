//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SphereCentered.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/cont/Span.hh"
#include "corecel/Types.hh"

#include "detail/QuadraticSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sphere centered at the origin.
 */
class SphereCentered
{
  public:
    //@{
    //! Type aliases
    using Intersections = Array<real_type, 2>;
    using Storage       = Span<const real_type, 1>;
    //@}

    //// CLASS ATTRIBUTES ////

    //! Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::sc;
    }

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return true; }

  public:
    //// CONSTRUCTORS ////

    // Construct with origin and radius
    explicit inline CELER_FUNCTION SphereCentered(real_type radius);

    // Construct from raw data
    explicit inline CELER_FUNCTION SphereCentered(Storage);

    //// ACCESSORS ////

    //! Square of the radius
    CELER_FUNCTION real_type radius_sq() const { return radius_sq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION Storage data() const { return {&radius_sq_, 1}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(const Real3& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        const Real3& pos, const Real3& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(const Real3& pos) const;

  private:
    // Square of the radius
    real_type radius_sq_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with sphere radius.
 */
CELER_FUNCTION SphereCentered::SphereCentered(real_type radius)
    : radius_sq_(ipow<2>(radius))
{
    CELER_EXPECT(radius > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
CELER_FUNCTION SphereCentered::SphereCentered(Storage data)
    : radius_sq_{data[0]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
CELER_FUNCTION SignedSense SphereCentered::calc_sense(const Real3& pos) const
{
    return real_to_sense(dot_product(pos, pos) - radius_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto
SphereCentered::calc_intersections(const Real3& pos,
                                   const Real3& dir,
                                   SurfaceState on_surface) const
    -> Intersections
{
    detail::QuadraticSolver solve_quadric(real_type(1), dot_product(pos, dir));
    if (on_surface == SurfaceState::off)
    {
        return solve_quadric(dot_product(pos, pos) - radius_sq_);
    }
    else
    {
        return solve_quadric();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
CELER_FUNCTION Real3 SphereCentered::calc_normal(const Real3& pos) const
{
    Real3 result{pos};
    normalize_direction(&result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
