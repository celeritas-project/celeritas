//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Plane.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<Axis T>
class PlaneAligned;

//---------------------------------------------------------------------------//
/*!
 * Arbitrarily oriented plane.
 *
 * A plane is a first-order quadric that satisfies \f[
    ax + bx + cz - d = 0
    \f]
 */
class Plane
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 1>;
    using StorageSpan = Span<real_type const, 4>;
    //@}

    //// CLASS ATTRIBUTES ////

    //! Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type()
    {
        return SurfaceType::p;
    }

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return true; }

  public:
    //// CONSTRUCTORS ////

    // Construct with normal and point
    explicit inline CELER_FUNCTION Plane(Real3 const& n, Real3 const& p);

    // Construct with normal and displacement
    explicit inline CELER_FUNCTION Plane(Real3 const& n, real_type d);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION Plane(Span<R, StorageSpan::extent>);

    // Promote from an axis-aligned plane
    template<Axis T>
    explicit Plane(PlaneAligned<T> const& other) noexcept;

    //// ACCESSORS ////

    //! Normal to the plane
    CELER_FUNCTION Real3 const& normal() const { return normal_; }

    //! Distance from the origin along the normal to the plane
    CELER_FUNCTION real_type displacement() const { return d_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&normal_[0], 4}; }

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position on the surface
    inline CELER_FUNCTION Real3 calc_normal(Real3 const&) const;

  private:
    // Normal to plane (a,b,c)
    Real3 normal_;

    // n \dot P (d)
    real_type d_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with unit normal and point.
 *
 * Displacement is the dot product of the point and the normal.
 */
CELER_FUNCTION Plane::Plane(Real3 const& n, Real3 const& p)
    : Plane{n, dot_product(n, p)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with unit normal and displacement.
 */
CELER_FUNCTION Plane::Plane(Real3 const& n, real_type d) : normal_{n}, d_{d}
{
    CELER_EXPECT(is_soft_unit_vector(normal_));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<class R>
CELER_FUNCTION Plane::Plane(Span<R, StorageSpan::extent> data)
    : normal_{data[0], data[1], data[2]}, d_{data[3]}
{
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
CELER_FUNCTION SignedSense Plane::calc_sense(Real3 const& pos) const
{
    return real_to_sense(dot_product(normal_, pos) - d_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
CELER_FUNCTION auto
Plane::calc_intersections(Real3 const& pos,
                          Real3 const& dir,
                          SurfaceState on_surface) const -> Intersections
{
    real_type const n_dir = dot_product(normal_, dir);
    if (on_surface == SurfaceState::off && n_dir != 0)
    {
        real_type const n_pos = dot_product(normal_, pos);
        real_type dist = (d_ - n_pos) / n_dir;
        if (dist > 0)
        {
            return {dist};
        }
    }
    return {no_intersection()};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal at a position on the surface.
 */
CELER_FORCEINLINE_FUNCTION Real3 Plane::calc_normal(Real3 const&) const
{
    return normal_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
