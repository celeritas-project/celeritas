//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/NegatedSurfaceClipper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "BoundingZone.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Truncate a bounding zone from a negated plane.
 *
 * A negated plane is one when "inside" the CSG node has an outward-facing
 * normal.
 *
 * \verbatim
            |--> PlaneAligned<T> outward normal
  exterior  |
       <----+---->  axis
            |
            |  interior
 * \endverbatim
 */
class NegatedSurfaceClipper
{
  public:
    // Construct with the bounding zone to clip
    explicit inline NegatedSurfaceClipper(BoundingZone* bz);

    //! Clip axis-aligned planes.
    template<Axis T>
    CELER_FORCEINLINE void operator()(PlaneAligned<T> const& s)
    {
        return this->clip_impl(T, s.position());
    }

    //! All other operations invalidate the "interior" box
    template<class S>
    CELER_FORCEINLINE void operator()(S const&)
    {
        return this->invalidate();
    }

  private:
    BoundingZone* bzone_;

    // Clip based on the given orthogonal plane
    inline void clip_impl(Axis ax, real_type pos);

    // Invalidate the interior zone due to non-convex surface
    inline void invalidate();
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the bounding zone to clip.
 */
NegatedSurfaceClipper::NegatedSurfaceClipper(BoundingZone* bz) : bzone_{bz}
{
    CELER_EXPECT(bzone_);
}

//---------------------------------------------------------------------------//
/*!
 * Clip based on the given orthogonal plane.
 */
void NegatedSurfaceClipper::clip_impl(Axis ax, real_type pos)
{
    bzone_->interior.shrink(Bound::lo, ax, pos);
    bzone_->exterior.shrink(Bound::lo, ax, pos);
}

//---------------------------------------------------------------------------//
/*!
 * Invalidate the interior zone due to non-convex surface.
 */
void NegatedSurfaceClipper::invalidate()
{
    bzone_->interior = {};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
