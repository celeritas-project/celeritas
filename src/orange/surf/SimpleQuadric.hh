//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.hh
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
 * General quadric expression but with no off-axis terms.
 *
 * Stored:
 * \f[
   ax^2 + by^2 + cz^2 + dx + ey + fz + g = 0
  \f]
 *
 * This can represent hyperboloids, ellipsoids, elliptical cylinders, etc.
 */
class SimpleQuadric
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 2>;
    using Storage = Span<const real_type, 7>;
    using SpanConstReal3 = Span<const real_type, 3>;
    //@}

    //// CLASS ATTRIBUTES ////

    //! Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::sq;
    }

    //! Safety is *not* the intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

  public:
    //// CONSTRUCTORS ////

    // Construct from coefficients
    inline CELER_FUNCTION
    SimpleQuadric(Real3 const& abc, Real3 const& def, real_type g);

    // Construct from another SQ and a translation
    inline CELER_FUNCTION
    SimpleQuadric(SimpleQuadric const& other, Real3 const& translation);

    // Construct from raw data
    explicit inline CELER_FUNCTION SimpleQuadric(Storage);

    //// ACCESSORS ////

    //! Second-order terms
    CELER_FUNCTION SpanConstReal3 second() const { return {&a_, 3}; }

    //! First-order terms
    CELER_FUNCTION SpanConstReal3 first() const { return {&d_, 3}; }

    //! Zeroth-order term
    CELER_FUNCTION real_type zeroth() const { return g_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION Storage data() const { return {&a_, 7}; }

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
    // First-order terms (d, e, f)
    real_type d_, e_, f_;
    // Constant term
    real_type g_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with coefficients.
 *
 * The quadric is ill-defined if all non-constants are zero.
 */
CELER_FUNCTION
SimpleQuadric::SimpleQuadric(Real3 const& abc, Real3 const& def, real_type g)
    : a_(abc[0])
    , b_(abc[1])
    , c_(abc[2])
    , d_(def[0])
    , e_(def[1])
    , f_(def[2])
    , g_(g)
{
    CELER_EXPECT(a_ != 0 || b_ != 0 || c_ != 0 || d_ != 0 || e_ != 0
                 || f_ != 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with another quadric and a translation.
 */
CELER_FUNCTION
SimpleQuadric::SimpleQuadric(SimpleQuadric const& other, Real3 const& origin)
    : SimpleQuadric{other}
{
    Real3 const orig_def{d_, e_, f_};

    constexpr auto X = to_int(Axis::x);
    constexpr auto Y = to_int(Axis::y);
    constexpr auto Z = to_int(Axis::z);

    // Expand out origin into the other terms
    d_ -= 2 * a_ * origin[X];
    e_ -= 2 * b_ * origin[Y];
    f_ -= 2 * c_ * origin[Z];

    g_ += a_ * origin[X] * origin[X];
    g_ += b_ * origin[Y] * origin[Y];
    g_ += c_ * origin[Z] * origin[Z];

    g_ -= 2 * orig_def[X] * origin[X];
    g_ -= 2 * orig_def[Y] * origin[Y];
    g_ -= 2 * orig_def[Z] * origin[Z];
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
CELER_FUNCTION SimpleQuadric::SimpleQuadric(Storage data)
    : a_{data[0]}
    , b_{data[1]}
    , c_{data[2]}
    , d_{data[3]}
    , e_{data[4]}
    , f_{data[5]}
    , g_{data[6]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
CELER_FUNCTION SignedSense SimpleQuadric::calc_sense(Real3 const& pos) const
{
    real_type const x = pos[to_int(Axis::x)];
    real_type const y = pos[to_int(Axis::y)];
    real_type const z = pos[to_int(Axis::z)];

    return real_to_sense((a_ * ipow<2>(x) + b_ * ipow<2>(y) + c_ * ipow<2>(z))
                         + (d_ * x + e_ * y + f_ * z) + (g_));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto
SimpleQuadric::calc_intersections(Real3 const& pos,
                                  Real3 const& dir,
                                  SurfaceState on_surface) const
    -> Intersections
{
    real_type const x = pos[to_int(Axis::x)];
    real_type const y = pos[to_int(Axis::y)];
    real_type const z = pos[to_int(Axis::z)];
    real_type const u = dir[to_int(Axis::x)];
    real_type const v = dir[to_int(Axis::y)];
    real_type const w = dir[to_int(Axis::z)];

    // Quadratic values
    real_type a = (a_ * u) * u + (b_ * v) * v + (c_ * w) * w;
    real_type b = (2 * a_ * x + d_) * u + (2 * b_ * y + e_) * v
                  + (2 * c_ * z + f_) * w;
    real_type c = (a_ * x + d_) * x + (b_ * y + e_) * y + (c_ * z + f_) * z
                  + g_;

    return detail::QuadraticSolver::solve_general(a, b / 2, c, on_surface);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
CELER_FUNCTION Real3 SimpleQuadric::calc_normal(Real3 const& pos) const
{
    real_type const x = pos[to_int(Axis::x)];
    real_type const y = pos[to_int(Axis::y)];
    real_type const z = pos[to_int(Axis::z)];

    Real3 norm{2 * a_ * x + d_, 2 * b_ * y + e_, 2 * c_ * z + f_};
    normalize_direction(&norm);
    return norm;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
