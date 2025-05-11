//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"

#include "detail/QuadraticSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<Axis T>
class ConeAligned;
template<Axis T>
class CylAligned;
class Plane;
class Sphere;

//---------------------------------------------------------------------------//
/*!
 * General quadric expression but with no off-axis terms.
 *
 * Stored:
 * \f[
   ax^2 + by^2 + cz^2 + dx + ey + fz + g = 0
  \f]
 *
 * This can represent axis-aligned hyperboloids, ellipsoids, elliptical
 * cylinders, etc.
 */
class SimpleQuadric
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 2>;
    using StorageSpan = Span<real_type const, 7>;
    using SpanConstReal3 = Span<real_type const, 3>;
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
    SimpleQuadric(Real3 const& abc, Real3 const& def, real_type g);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION SimpleQuadric(Span<R, StorageSpan::extent>);

    // Promote from a plane
    explicit SimpleQuadric(Plane const& other) noexcept(!CELERITAS_DEBUG);

    // Promote from an axis-aligned cylinder
    template<Axis T>
    explicit SimpleQuadric(CylAligned<T> const& other) noexcept;

    // Promote from a sphere
    explicit SimpleQuadric(Sphere const& other) noexcept(!CELERITAS_DEBUG);

    // Promote from a cone
    template<Axis T>
    explicit SimpleQuadric(ConeAligned<T> const& other) noexcept;

    //// ACCESSORS ////

    //! Second-order terms
    CELER_FUNCTION SpanConstReal3 second() const { return {&a_, 3}; }

    //! First-order terms
    CELER_FUNCTION SpanConstReal3 first() const { return {&d_, 3}; }

    //! Zeroth-order term
    CELER_FUNCTION real_type zeroth() const { return g_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&a_, 7}; }

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
 * Construct from raw data.
 */
template<class R>
CELER_FUNCTION SimpleQuadric::SimpleQuadric(Span<R, StorageSpan::extent> data)
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
    return make_unit_vector(norm);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
