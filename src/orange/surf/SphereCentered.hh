//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SphereCentered.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/SenseUtils.hh"

#include "detail/IntersectionSolver.hh"

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
    //! \name Type aliases
    using Intersections = Array<real_type, 2>;
    using StorageSpan = Span<real_type const, 1>;
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

    // Construct with square of radius for simplification
    static inline SphereCentered from_radius_sq(real_type rsq);

    // Construct with origin and radius
    explicit inline CELER_FUNCTION SphereCentered(real_type radius);

    // Construct from raw data
    template<class R>
    explicit inline
        CELER_FUNCTION SphereCentered(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! Square of the radius
    CELER_FUNCTION real_type radius_sq() const { return radius_sq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&radius_sq_, 1}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate the sphere expression at a point
    inline CELER_FUNCTION real_type calc_value(Real3 const& pos) const;

    // Prepare intersection inputs for the shared quadratic solver
    inline CELER_FUNCTION detail::QuadraticIntersectionInput
    calc_intersection_quadratic(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

    // Evaluate the gradient at a point
    inline CELER_FUNCTION Real3 calc_gradient(Real3 const& pos) const;

  private:
    // Square of the radius
    real_type radius_sq_;

    //! Private default constructor for manual construction
    SphereCentered() = default;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from the square of the radius.
 *
 * This is used for surface simplification.
 */
SphereCentered SphereCentered::from_radius_sq(real_type rsq)
{
    CELER_EXPECT(rsq > 0);
    SphereCentered result;
    result.radius_sq_ = rsq;
    return result;
}

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
template<class R>
CELER_FUNCTION SphereCentered::SphereCentered(Span<R, StorageSpan::extent> data)
    : radius_sq_{data[0]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
CELER_FUNCTION SignedSense SphereCentered::calc_sense(Real3 const& pos) const
{
    return real_to_sense(this->calc_value(pos));
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the sphere expression at a point.
 */
CELER_FUNCTION real_type SphereCentered::calc_value(Real3 const& pos) const
{
    return dot_product(pos, pos) - radius_sq_;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto SphereCentered::calc_intersections(
    Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const
    -> Intersections
{
    return detail::solve_intersections(
        this->calc_intersection_quadratic(pos, dir, on_surface));
}

//---------------------------------------------------------------------------//
/*!
 * Prepare input values for the shared quadratic intersection solver.
 */
CELER_FUNCTION auto SphereCentered::calc_intersection_quadratic(
    Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const
    -> detail::QuadraticIntersectionInput
{
    real_type const half_b = dot_product(pos, dir);
    real_type const c = dot_product(pos, pos) - radius_sq_;
    return {1, half_b, c, on_surface};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
CELER_FUNCTION Real3 SphereCentered::calc_normal(Real3 const& pos) const
{
    return make_unit_vector(this->calc_gradient(pos));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the gradient at a point.
 */
CELER_FUNCTION Real3 SphereCentered::calc_gradient(Real3 const& pos) const
{
    return {2 * pos[0], 2 * pos[1], 2 * pos[2]};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
