//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/NegatedSurfaceClipper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/BoundingBox.hh"
#include "orange/surf/PlaneAligned.hh"

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
    // Construct with interior/exterior
    inline NegatedSurfaceClipper(BBox* interior, BBox* exterior);

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
    BBox* interior_{nullptr};
    BBox* exterior_{nullptr};

    // Clip based on the given orthogonal plane
    inline void clip_impl(Axis ax, real_type pos);

    // Invalidate the interior zone due to non-convex surface
    inline void invalidate();
};

//---------------------------------------------------------------------------//
/*!
 * Construct with the bounding zone to clip.
 */
NegatedSurfaceClipper::NegatedSurfaceClipper(BoundingZone* bz)
    : interior_{&bz->interior}, exterior_{&bz->exterior}
{
    CELER_EXPECT(bz && interior_ && exterior_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with explicit but optional bounding boxes
 */
NegatedSurfaceClipper::NegatedSurfaceClipper(BBox* interior, BBox* exterior)
    : interior_{interior}, exterior_{exterior}
{
    CELER_EXPECT(interior_ || exterior_);
}

//---------------------------------------------------------------------------//
/*!
 * Clip based on the given orthogonal plane.
 */
void NegatedSurfaceClipper::clip_impl(Axis ax, real_type pos)
{
    if (interior_)
    {
        interior_->shrink(Bound::lo, ax, pos);
    }
    if (exterior_)
    {
        exterior_->shrink(Bound::lo, ax, pos);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Invalidate the interior zone due to non-convex surface.
 */
void NegatedSurfaceClipper::invalidate()
{
    if (interior_)
    {
        *interior_ = {};
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
