//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/CylCentered.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"

#include "detail/QuadraticSolver.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Axis-aligned cylinder centered about the origin.
 *
 * The cylinder is centered along an Axis template parameter.
 *
 * For a cylinder along the x axis:
 * \f[
    y^2 + z^2 - R^2 = 0
   \f]
 *
 * This is an optimization of the Cyl. The motivations are:
 * - Many geometries have units with concentric cylinders centered about the
 *   origin, so having this as a special case reduces the memory usage of those
 *   units (improved memory localization).
 * - Cylindrical mesh geometries have lots of these cylinders, so efficient
 *   tracking through its cells should make this optimization worthwhile.
 */
template<Axis T>
class CylCentered
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 2>;
    using StorageSpan = Span<real_type const, 1>;
    //@}

  private:
    static constexpr Axis U{T == Axis::x ? Axis::y : Axis::x};
    static constexpr Axis V{T == Axis::z ? Axis::y : Axis::z};

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type();

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return true; }

    //!@{
    //! Axes
    static CELER_CONSTEXPR_FUNCTION Axis t_axis() { return T; }
    static CELER_CONSTEXPR_FUNCTION Axis u_axis() { return U; }
    static CELER_CONSTEXPR_FUNCTION Axis v_axis() { return V; }
    //!@}

  public:
    //// CONSTRUCTORS ////

    // Construct with square of radius for simplification
    static inline CylCentered from_radius_sq(real_type rsq);

    // Construct with radius
    explicit inline CELER_FUNCTION CylCentered(real_type radius);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION CylCentered(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! Get the square of the radius
    CELER_FUNCTION real_type radius_sq() const { return radius_sq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&radius_sq_, 1}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    //! Square of cylinder radius
    real_type radius_sq_;

    //! Private default constructor for manual construction
    CylCentered() = default;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using CCylX = CylCentered<Axis::x>;
using CCylY = CylCentered<Axis::y>;
using CCylZ = CylCentered<Axis::z>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Surface type identifier.
 */
template<Axis T>
CELER_CONSTEXPR_FUNCTION SurfaceType CylCentered<T>::surface_type()
{
    return T == Axis::x   ? SurfaceType::cxc
           : T == Axis::y ? SurfaceType::cyc
           : T == Axis::z ? SurfaceType::czc
                          : SurfaceType::size_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from the square of the radius.
 *
 * This is used for surface simplification.
 */
template<Axis T>
CylCentered<T> CylCentered<T>::from_radius_sq(real_type rsq)
{
    CELER_EXPECT(rsq > 0);

    CylCentered<T> result;
    result.radius_sq_ = rsq;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with radius.
 */
template<Axis T>
CELER_FUNCTION CylCentered<T>::CylCentered(real_type radius)
    : radius_sq_(ipow<2>(radius))
{
    CELER_EXPECT(radius > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<Axis T>
template<class R>
CELER_FUNCTION CylCentered<T>::CylCentered(Span<R, StorageSpan::extent> data)
    : radius_sq_(data[0])
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
template<Axis T>
CELER_FUNCTION SignedSense CylCentered<T>::calc_sense(Real3 const& pos) const
{
    real_type const u = pos[to_int(U)];
    real_type const v = pos[to_int(V)];

    return real_to_sense(ipow<2>(u) + ipow<2>(v) - radius_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
template<Axis T>
CELER_FUNCTION auto
CylCentered<T>::calc_intersections(Real3 const& pos,
                                   Real3 const& dir,
                                   SurfaceState on_surface) const
    -> Intersections
{
    // 1 - (\omega \dot t)^2 where t is axis of cylinder
    real_type const a = 1 - ipow<2>(dir[to_int(T)]);

    if (a < ipow<2>(Tolerance<>::sqrt_quadratic()))
    {
        // No intersection if we're traveling along the cylinder axis
        return {no_intersection(), no_intersection()};
    }

    real_type const u = pos[to_int(U)];
    real_type const v = pos[to_int(V)];

    // b/2 = \omega \dot (x - x_0)
    detail::QuadraticSolver solve_quadric(
        a, dir[to_int(U)] * u + dir[to_int(V)] * v);
    if (on_surface == SurfaceState::on)
    {
        // Solve degenerate case (c=0)
        return solve_quadric();
    }

    // c = (x - x_0) \dot (x - x_0) - R * R
    return solve_quadric(ipow<2>(u) + ipow<2>(v) - radius_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
template<Axis T>
CELER_FUNCTION Real3 CylCentered<T>::calc_normal(Real3 const& pos) const
{
    Real3 norm{0, 0, 0};

    norm[to_int(U)] = pos[to_int(U)];
    norm[to_int(V)] = pos[to_int(V)];

    return make_unit_vector(norm);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
