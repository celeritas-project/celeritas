//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/CylAligned.hh
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
template<Axis T>
class CylCentered;

//---------------------------------------------------------------------------//
/*!
 * Axis-aligned cylinder.
 *
 * The cylinder is centered about the template parameter Axis.
 *
 * For a cylinder along the x axis:
 * \f[
    (y - y_0)^2 + (z - z_0)^2 - R^2 = 0
   \f]
 *
 * The axis-aligned cylinder is used to calculate a tolerance for the
 * second-order coefficient on the quadratic equation, where a small value is
 * approximately zero.
 *
 * Basing the intersection tolerance on a particle traveling nearly parallel to
 * the cylinder, the angle \f$\theta \f$
 * between the particle's direction and the center axis of the cylinder needed
 * to give an intersection distance of \f$ 1/\delta \f$ when a distance of 1
 * from the cylinder (or equivalently, an intersection of 1 when a distance
 * \f$ \delta \f$ away) is
 * \f[
   \sin \theta = \frac{1}{1/\delta} \;.
   \f]
 * Letting \f$ \mu = \cos \theta = \Omega \cdot t \f$ we have \f[
   \mu^2 + \delta^2 = 1 \;.
   \f]
 * The quadratic intersection for an axis-aligned cylinder where the particle
 * is at \f$ R + \delta \f$ is
 * \f[
   0 = ax^2 + bx + c
     = (1 - \mu^2)x^2 + 2 \sqrt{1 - \mu^2} (R + 1) x + (R + 1)^2 - R^2
  \f]
 * so the quadric coefficient when a particle is effectively parallel to a
 * cylinder is:
  \f[
  a = (1 - \mu^2) = \delta^2 \;.
  \f]
 *
 * Because \em a is calculated by subtraction, this puts a hard limit on the
 * accuracy of the intersection distance: the default value of a=1e-10
 * corresponds to a tolerance of 1e-5, not the default tolerance 1e-8.
 */
template<Axis T>
class CylAligned
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 2>;
    using StorageSpan = Span<real_type const, 3>;
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

    // Construct with square of radius for simplification
    static CylAligned from_radius_sq(Real3 const& origin, real_type rsq);

    // Construct with radius
    inline CELER_FUNCTION CylAligned(Real3 const& origin, real_type radius);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION CylAligned(Span<R, StorageSpan::extent>);

    // Promote from a centered axis-aligned cylinder
    explicit CylAligned(CylCentered<T> const& other) noexcept;

    //// ACCESSORS ////

    //! Get the origin vector along the 'u' axis
    CELER_FUNCTION real_type origin_u() const { return origin_u_; }

    //! Get the origin vector along the 'v' axis
    CELER_FUNCTION real_type origin_v() const { return origin_v_; }

    //! Get the square of the radius
    CELER_FUNCTION real_type radius_sq() const { return radius_sq_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&origin_u_, 3}; }

    // Helper function to get the origin as a 3-vector
    Real3 calc_origin() const;

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    // Off-axis location
    real_type origin_u_;
    real_type origin_v_;

    // Square of the radius
    real_type radius_sq_;

    //! Private default constructor for manual construction
    CylAligned() = default;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using CylX = CylAligned<Axis::x>;
using CylY = CylAligned<Axis::y>;
using CylZ = CylAligned<Axis::z>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Surface type identifier.
 */
template<Axis T>
CELER_CONSTEXPR_FUNCTION SurfaceType CylAligned<T>::surface_type()
{
    return T == Axis::x   ? SurfaceType::cx
           : T == Axis::y ? SurfaceType::cy
           : T == Axis::z ? SurfaceType::cz
                          : SurfaceType::size_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from origin and radius.
 */
template<Axis T>
CELER_FUNCTION CylAligned<T>::CylAligned(Real3 const& origin, real_type radius)
    : origin_u_{origin[to_int(U)]}
    , origin_v_{origin[to_int(V)]}
    , radius_sq_{ipow<2>(radius)}
{
    CELER_EXPECT(radius > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<Axis T>
template<class R>
CELER_FUNCTION CylAligned<T>::CylAligned(Span<R, StorageSpan::extent> data)
    : origin_u_{data[0]}, origin_v_{data[1]}, radius_sq_{data[2]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
template<Axis T>
CELER_FUNCTION SignedSense CylAligned<T>::calc_sense(Real3 const& pos) const
{
    real_type const u = pos[to_int(U)] - origin_u_;
    real_type const v = pos[to_int(V)] - origin_v_;

    return real_to_sense(ipow<2>(u) + ipow<2>(v) - radius_sq_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
template<Axis T>
CELER_FUNCTION auto
CylAligned<T>::calc_intersections(Real3 const& pos,
                                  Real3 const& dir,
                                  SurfaceState on_surface) const
    -> Intersections
{
    // 1 - \omega \dot e
    real_type const a = 1 - ipow<2>(dir[to_int(T)]);

    if (a < ipow<2>(Tolerance<>::sqrt_quadratic()))
    {
        // No intersection if we're traveling along the cylinder axis
        return {no_intersection(), no_intersection()};
    }

    real_type const u = pos[to_int(U)] - origin_u_;
    real_type const v = pos[to_int(V)] - origin_v_;

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
CELER_FUNCTION Real3 CylAligned<T>::calc_normal(Real3 const& pos) const
{
    Real3 norm{0, 0, 0};

    norm[to_int(U)] = pos[to_int(U)] - origin_u_;
    norm[to_int(V)] = pos[to_int(V)] - origin_v_;

    return make_unit_vector(norm);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
