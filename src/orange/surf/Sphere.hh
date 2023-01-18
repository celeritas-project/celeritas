//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Sphere.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

#include "detail/QuadraticSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sphere centered at an arbitrary point.
 */
class Sphere
{
  public:
    //@{
    //! Type aliases
    using Intersections = Array<real_type, 2>;
    using Storage = Span<const real_type, 4>;
    //@}

    //// CLASS ATTRIBUTES ////

    //! Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::s;
    }

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return true; }

  public:
    //// CONSTRUCTORS ////

    // Construct with origin and radius
    inline CELER_FUNCTION Sphere(Real3 const& origin, real_type radius);

    // Construct from raw data
    explicit inline CELER_FUNCTION Sphere(Storage);

    //// ACCESSORS ////

    //! Center position
    CELER_FUNCTION Real3 const& origin() const { return origin_; }

    //! Square of the radius
    CELER_FUNCTION real_type radius_sq() const { return radius_sq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION Storage data() const { return {origin_.data(), 4}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    // Spatial position
    Real3 origin_;
    // Square of the radius
    real_type radius_sq_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with sphere origin and radius.
 */
CELER_FUNCTION Sphere::Sphere(Real3 const& origin, real_type radius)
    : origin_(origin), radius_sq_(ipow<2>(radius))
{
    CELER_EXPECT(radius > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
CELER_FUNCTION Sphere::Sphere(Storage data)
    : origin_{data[0], data[1], data[2]}, radius_sq_{data[3]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
CELER_FUNCTION SignedSense Sphere::calc_sense(Real3 const& pos) const
{
    Real3 tpos{pos[0] - origin_[0], pos[1] - origin_[1], pos[2] - origin_[2]};

    return real_to_sense(dot_product(tpos, tpos) - radius_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto Sphere::calc_intersections(Real3 const& pos,
                                               Real3 const& dir,
                                               SurfaceState on_surface) const
    -> Intersections
{
    Real3 tpos{pos[0] - origin_[0], pos[1] - origin_[1], pos[2] - origin_[2]};

    detail::QuadraticSolver solve_quadric(real_type(1), dot_product(tpos, dir));
    if (on_surface == SurfaceState::off)
    {
        return solve_quadric(dot_product(tpos, tpos) - radius_sq_);
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
CELER_FUNCTION Real3 Sphere::calc_normal(Real3 const& pos) const
{
    Real3 tpos{pos[0] - origin_[0], pos[1] - origin_[1], pos[2] - origin_[2]};
    normalize_direction(&tpos);
    return tpos;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
