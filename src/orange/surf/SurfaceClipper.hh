//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

#include "VariantSurface.hh"
#include "detail/AllSurfaces.hh"
#include "detail/SurfaceClipperImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Truncate a bounding box to its intersection with a surface interior.
 *
 * Even though most quadric surfaces are infinite, their intersection with a
 * bounding box may be a smaller bounding box. This operation accelerates
 * "distance to in" calculations.
 */
class SurfaceClipper
{
  public:
    // Construct with a reference to the bbox being clipped
    explicit inline SurfaceClipper(BBox* bbox);

    //// OPERATION ////

    // Apply to a surface with a known type
    template<class S>
    void operator()(Sense s, S const& surf);

    // Apply to a surface with unknown type
    void operator()(Sense s, VariantSurface const& surf);

  private:
    BBox* bbox_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a bounding box reference.
 */
SurfaceClipper::SurfaceClipper(BBox* bbox) : bbox_{bbox}
{
    CELER_EXPECT(bbox);
}

//---------------------------------------------------------------------------//
/*!
 * Apply to a surface with a known type.
 */
template<class S>
void SurfaceClipper::operator()(Sense sense, S const& surf)
{
    if (sense == Sense::inside)
    {
        return detail::SurfaceClipperImpl<Sense::inside>{bbox_}(surf);
    }
    else
    {
        return detail::SurfaceClipperImpl<Sense::outside>{bbox_}(surf);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Apply to a surface with an unknown type.
 */
void SurfaceClipper::operator()(Sense sense, VariantSurface const& surf)
{
    CELER_ASSUME(!surf.valueless_by_exception());
    if (sense == Sense::inside)
    {
        return std::visit(detail::SurfaceClipperImpl<Sense::inside>{bbox_},
                          surf);
    }
    else
    {
        return std::visit(detail::SurfaceClipperImpl<Sense::outside>{bbox_},
                          surf);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
