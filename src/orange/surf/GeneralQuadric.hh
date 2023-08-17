//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/GeneralQuadric.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

#include "detail/QuadraticSolver.hh"

namespace celeritas
{
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
 */
class GeneralQuadric
{
  public:
    //@{
    //! Type aliases
    using Intersections = Array<real_type, 2>;
    using Storage = Span<const real_type, 10>;
    using SpanConstReal3 = Span<const real_type, 3>;
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

    // Construct with radius
    explicit inline CELER_FUNCTION GeneralQuadric(Real3 const& abc,
                                                  Real3 const& def,
                                                  Real3 const& ghi,
                                                  real_type j);

    // Construct from raw data
    explicit inline CELER_FUNCTION GeneralQuadric(Storage);

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
    CELER_FUNCTION Storage data() const { return {&a_, 10}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

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
 * Construct with all coefficients.
 */
CELER_FUNCTION GeneralQuadric::GeneralQuadric(Real3 const& abc,
                                              Real3 const& def,
                                              Real3 const& ghi,
                                              real_type j)
    : a_(abc[0])
    , b_(abc[1])
    , c_(abc[2])
    , d_(def[0])
    , e_(def[1])
    , f_(def[2])
    , g_(ghi[0])
    , h_(ghi[1])
    , i_(ghi[2])
    , j_(j)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
CELER_FUNCTION GeneralQuadric::GeneralQuadric(Storage data)
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
    const real_type x = pos[0];
    const real_type y = pos[1];
    const real_type z = pos[2];

    real_type result = (a_ * x + d_ * y + f_ * z + g_) * x
                       + (b_ * y + e_ * z + h_) * y + (c_ * z + i_) * z + j_;

    return real_to_sense(result);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto
GeneralQuadric::calc_intersections(Real3 const& pos,
                                   Real3 const& dir,
                                   SurfaceState on_surface) const
    -> Intersections
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

    return detail::QuadraticSolver::solve_general(a, b / 2, c, on_surface);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
CELER_FUNCTION Real3 GeneralQuadric::calc_normal(Real3 const& pos) const
{
    real_type const x = pos[0];
    real_type const y = pos[1];
    real_type const z = pos[2];

    Real3 norm;
    norm[0] = 2 * a_ * x + d_ * y + f_ * z + g_;
    norm[1] = 2 * b_ * y + d_ * x + e_ * z + h_;
    norm[2] = 2 * c_ * z + e_ * y + f_ * x + i_;

    normalize_direction(&norm);
    return norm;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
