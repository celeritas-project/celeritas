//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/PlaneAligned.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Axis-aligned plane with positive-facing normal.
 */
template<Axis T>
class PlaneAligned
{
  public:
    //@{
    //! \name Type aliases
    using Intersections = Array<real_type, 1>;
    using StorageSpan = Span<real_type const, 1>;
    //@}

    //// CLASS ATTRIBUTES ////

    // Surface type identifier
    static CELER_CONSTEXPR_FUNCTION SurfaceType surface_type();

    //! Safety is intersection along surface normal
    static CELER_CONSTEXPR_FUNCTION bool simple_safety() { return true; }

  public:
    //// CONSTRUCTORS ////

    // Construct with radius
    explicit inline CELER_FUNCTION PlaneAligned(real_type position);

    // Construct from raw data
    template<class R>
    explicit inline CELER_FUNCTION PlaneAligned(Span<R, StorageSpan::extent>);

    //// ACCESSORS ////

    //! Get the square of the radius
    CELER_FUNCTION real_type position() const { return position_; }

    //! Get a view to the data for type-deleted storage
    CELER_FUNCTION StorageSpan data() const { return {&position_, 1}; }

    // Construct outward normal vector
    inline CELER_FUNCTION Real3 calc_normal() const;

    //// CALCULATION ////

    // Determine the sense of the position relative to this surface
    inline CELER_FUNCTION SignedSense calc_sense(Real3 const& pos) const;

    // Calculate all possible straight-line intersections with this surface
    inline CELER_FUNCTION Intersections calc_intersections(
        Real3 const& pos, Real3 const& dir, SurfaceState on_surface) const;

    // Calculate outward normal at a position
    inline CELER_FUNCTION Real3 calc_normal(Real3 const& pos) const;

  private:
    //! Intersection with the axis
    real_type position_;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using PlaneX = PlaneAligned<Axis::x>;
using PlaneY = PlaneAligned<Axis::y>;
using PlaneZ = PlaneAligned<Axis::z>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Surface type identifier.
 */
template<Axis T>
CELER_CONSTEXPR_FUNCTION SurfaceType PlaneAligned<T>::surface_type()
{
    return T == Axis::x   ? SurfaceType::px
           : T == Axis::y ? SurfaceType::py
           : T == Axis::z ? SurfaceType::pz
                          : SurfaceType::size_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from axis intercept.
 */
template<Axis T>
CELER_FUNCTION PlaneAligned<T>::PlaneAligned(real_type position)
    : position_(position)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from raw data.
 */
template<Axis T>
template<class R>
CELER_FUNCTION PlaneAligned<T>::PlaneAligned(Span<R, StorageSpan::extent> data)
    : position_(data[0])
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate outward normal.
 */
template<Axis T>
CELER_FUNCTION Real3 PlaneAligned<T>::calc_normal() const
{
    Real3 norm{0, 0, 0};

    norm[to_int(T)] = 1.;
    return norm;
}

//---------------------------------------------------------------------------//
/*!
 * Determine the sense of the position relative to this surface.
 */
template<Axis T>
CELER_FUNCTION SignedSense PlaneAligned<T>::calc_sense(Real3 const& pos) const
{
    return real_to_sense(pos[to_int(T)] - position_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate all possible straight-line intersections with this surface.
 */
template<Axis T>
CELER_FUNCTION auto
PlaneAligned<T>::calc_intersections(Real3 const& pos,
                                    Real3 const& dir,
                                    SurfaceState on_surface) const
    -> Intersections
{
    real_type const n_dir = dir[to_int(T)];
    if (on_surface == SurfaceState::off && n_dir != 0)
    {
        real_type const n_pos = pos[to_int(T)];
        real_type dist = (position_ - n_pos) / n_dir;
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
template<Axis T>
CELER_FUNCTION Real3 PlaneAligned<T>::calc_normal(Real3 const&) const
{
    return this->calc_normal();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
