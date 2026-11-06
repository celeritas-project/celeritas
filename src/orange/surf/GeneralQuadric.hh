//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/GeneralQuadric.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"

#include "detail/IntersectionSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class SimpleQuadric;

//---------------------------------------------------------------------------//
/*!
 * General quadric surface.
 *
 * General quadrics that cannot be simplified to other ORANGE surfaces include
 * hyperboloids and paraboloids; and non-axis-aligned cylinders, ellipsoids,
 * and cones.
 *
 * \f[
    ax^2 + by^2 + cz^2 + dxy + eyz + fzx + gx + hy + iz + j = 0
   \f]
 *
 * Note that some formulations of a general quadric include a factor of 2 for
 * the g/h/i terms, or use \f$ \dots + iz = j \f$ (reversing the sign of \em
 j).
 */
class GeneralQuadric
{
  public:
    //@{
    //! Type aliases
    using Intersections = Array<real_type, 2>;
    using StorageSpan = Span<real_type const, 10>;
    using SpanConstReal3 = Span<real_type const, 3>;
    //@}

    //// CLASS ATTRIBUTES ////

    //! Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::gq;
    }

    //! Safety is *not* the nearest intersection along the surface "normal"
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

  public:
    //// CONSTRUCTORS ////

    // Construct from coefficients
    explicit GeneralQuadric(
        Real3 const& abc, Real3 const& def, Real3 const& ghi, real_type j);

    // Construct from raw data
    template<class R>
    explicit inline
        CELER_FUNCTION GeneralQuadric(Span<R, StorageSpan::extent>);

    // Promote from a simple quadric
    explicit GeneralQuadric(SimpleQuadric const& other) noexcept(
        !CELERITAS_DEBUG);

    //// ACCESSORS ////

    //! Second-order terms
    CELER_FUNCTION SpanConstReal3 second() const { return {&a_, 3}; }

    //! Cross terms (xy, yz, zx)
    CELER_FUNCTION SpanConstReal3 cross() const { return {&d_, 3}; }

    //! First-order terms
    CELER_FUNCTION SpanConstReal3 first() const { return {&g_, 3}; }

    //! Zeroth-order term
    CELER_FUNCTION real_type zeroth() const { return j_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&a_, 10}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate the quadric expression at a point
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
    // Second-order terms (a, b, c)
    real_type a_, b_, c_;
    // Second-order cross terms (d, e, f)
    real_type d_, e_, f_;
    // First-order terms (g, h, i)
    real_type g_, h_, i_;
    // Constant term
    real_type j_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<class R>
CELER_FUNCTION GeneralQuadric::GeneralQuadric(Span<R, StorageSpan::extent> data)
    : a_(data[0])
    , b_(data[1])
    , c_(data[2])
    , d_(data[3])
    , e_(data[4])
    , f_(data[5])
    , g_(data[6])
    , h_(data[7])
    , i_(data[8])
    , j_(data[9])
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
CELER_FUNCTION SignedSense GeneralQuadric::calc_sense(Real3 const& pos) const
{
    return real_to_sense(this->calc_value(pos));
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the quadric expression at a point.
 */
CELER_FUNCTION real_type GeneralQuadric::calc_value(Real3 const& pos) const
{
    real_type const x = pos[0];
    real_type const y = pos[1];
    real_type const z = pos[2];

    real_type result = (a_ * x + d_ * y + f_ * z + g_) * x
                       + (b_ * y + e_ * z + h_) * y + (c_ * z + i_) * z + j_;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto GeneralQuadric::calc_intersections(
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
CELER_FUNCTION auto GeneralQuadric::calc_intersection_quadratic(
    Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const
    -> detail::QuadraticIntersectionInput
{
    real_type const x = pos[0];
    real_type const y = pos[1];
    real_type const z = pos[2];
    real_type const u = dir[0];
    real_type const v = dir[1];
    real_type const w = dir[2];

    // Quadratic values
    real_type a = (a_ * u + d_ * v) * u + (b_ * v + e_ * w) * v
                  + (c_ * w + f_ * u) * w;
    real_type b = (2 * a_ * x + d_ * y + f_ * z + g_) * u
                  + (2 * b_ * y + d_ * x + e_ * z + h_) * v
                  + (2 * c_ * z + e_ * y + f_ * x + i_) * w;
    real_type c = ((a_ * x + d_ * y + g_) * x + (b_ * y + e_ * z + h_) * y
                   + (c_ * z + f_ * x + i_) * z + j_);

    return {a, b / 2, c, on_surface};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
CELER_FUNCTION Real3 GeneralQuadric::calc_normal(Real3 const& pos) const
{
    return make_unit_vector(this->calc_gradient(pos));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the gradient at a point.
 */
CELER_FUNCTION Real3 GeneralQuadric::calc_gradient(Real3 const& pos) const
{
    real_type const x = pos[to_int(Axis::x)];
    real_type const y = pos[to_int(Axis::y)];
    real_type const z = pos[to_int(Axis::z)];

    return {2 * a_ * x + d_ * y + f_ * z + g_,
            2 * b_ * y + d_ * x + e_ * z + h_,
            2 * c_ * z + e_ * y + f_ * x + i_};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
