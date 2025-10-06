//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/ConeAligned.hh
//---------------------------------------------------------------------------//
#pragma once

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
/*!
 * Axis-aligned cone (infinite and double-sheeted).
 *
 * For a cone parallel to the z axis:
 * \f[
    (x - x_0)^2 + (y - y_0)^2 - t^2 (z - z_0)^2 = 0
   \f]

 * where \em t is the tangent of the opening angle (\f$r/h\f$ for a finite cone
 * with radius \em r and height \em h). In the cone below, the tangent of the
 * inner angle is \f$ t = \tan(\theta) = r/h \f$. Negative values of \em t are
 * equivalent and valid, representing the four points on a slice of the
 * double-sheet cone.
 * \verbatim
  \        |   r    /
   o- - - -+- - - -o
    '--_   :   _--'
 h      --.:.--
 ---       O
       _--':'--_
    .--    :    --.
   o-------+-------o
  /        |        \
   \endverbatim
 *
 */
template<Axis T>
class ConeAligned
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 2>;
    using StorageSpan = Span<real_type const, 4>;
    //@}

  private:
    static constexpr Axis U{T == Axis::x ? Axis::y : Axis::x};
    static constexpr Axis V{T == Axis::z ? Axis::y : Axis::z};

  public:
    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type();

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return false; }

    //!@{
    //! Axes
    static CELER_CONSTEXPR_FUNCTION Axis t_axis() { return T; }
    static CELER_CONSTEXPR_FUNCTION Axis u_axis() { return U; }
    static CELER_CONSTEXPR_FUNCTION Axis v_axis() { return V; }
    //!@}

  public:
    //// CONSTRUCTORS ////

    // Construct with square of tangent for simplification
    static ConeAligned from_tangent_sq(Real3 const& origin, real_type tsq);

    // Construct from origin and tangent of the angle of its opening
    ConeAligned(Real3 const& origin, real_type tangent);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION ConeAligned(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! Get the origin position along the normal axis
    CELER_FUNCTION Real3 const& origin() const { return origin_; }

    //! Get the square of the tangent of the opening angle
    CELER_FUNCTION real_type tangent_sq() const { return tsq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&origin_[0], 4}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    // Location of the vanishing point
    Real3 origin_;

    // Quadric value
    real_type tsq_;

    //! Private default constructor for manual construction
    ConeAligned() = default;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using ConeX = ConeAligned<Axis::x>;
using ConeY = ConeAligned<Axis::y>;
using ConeZ = ConeAligned<Axis::z>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Surface type identifier.
 */
template<Axis T>
CELER_CONSTEXPR_FUNCTION SurfaceType ConeAligned<T>::surface_type()
{
    return T == Axis::x   ? SurfaceType::kx
           : T == Axis::y ? SurfaceType::ky
           : T == Axis::z ? SurfaceType::kz
                          : SurfaceType::size_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<Axis T>
template<class R>
CELER_FUNCTION ConeAligned<T>::ConeAligned(Span<R, StorageSpan::extent> data)
    : origin_{data[0], data[1], data[2]}, tsq_{data[3]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
template<Axis T>
CELER_FUNCTION SignedSense ConeAligned<T>::calc_sense(Real3 const& pos) const
{
    real_type const x = pos[to_int(T)] - origin_[to_int(T)];
    real_type const y = pos[to_int(U)] - origin_[to_int(U)];
    real_type const z = pos[to_int(V)] - origin_[to_int(V)];

    return real_to_sense((-tsq_ * ipow<2>(x)) + ipow<2>(y) + ipow<2>(z));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 *
 * \f[
    (y - yc)^2 + (z - zc)^2 - t^2 * (x - xc)^2 = 0
   \f]
 */
template<Axis T>
CELER_FUNCTION auto
ConeAligned<T>::calc_intersections(Real3 const& pos,
                                   Real3 const& dir,
                                   SurfaceState on_surface) const
    -> Intersections
{
    // Expand translated positions into 'xyz' coordinate system
    real_type const x = pos[to_int(T)] - origin_[to_int(T)];
    real_type const y = pos[to_int(U)] - origin_[to_int(U)];
    real_type const z = pos[to_int(V)] - origin_[to_int(V)];

    real_type const u = dir[to_int(T)];
    real_type const v = dir[to_int(U)];
    real_type const w = dir[to_int(V)];

    // Scaled direction
    real_type a = (-tsq_ * ipow<2>(u)) + ipow<2>(v) + ipow<2>(w);
    real_type half_b = (-tsq_ * x * u) + (y * v) + (z * w);
    real_type c = (-tsq_ * ipow<2>(x)) + ipow<2>(y) + ipow<2>(z);

    return detail::QuadraticSolver::solve_general(a, half_b, c, on_surface);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position.
 */
template<Axis T>
CELER_FUNCTION Real3 ConeAligned<T>::calc_normal(Real3 const& pos) const
{
    Real3 norm;
    for (auto i = to_int(Axis::x); i < to_int(Axis::size_); ++i)
    {
        norm[i] = pos[i] - origin_[i];
    }
    norm[to_int(T)] *= -tsq_;

    return make_unit_vector(norm);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
