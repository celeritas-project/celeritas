//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipperImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

#include "../SurfaceFwd.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<Sense>
class SurfaceClipperImpl;

//---------------------------------------------------------------------------//
/*!
 * Truncate a bounding box to its intersection with a surface interior.
 *
 * Even though most quadric surfaces are infinite, their intersection with a
 * bounding box may be a smaller bounding box. This operation accelerates
 * "distance to in" calculations.
 */
template<>
class SurfaceClipperImpl<Sense::inside>
{
  public:
    // Construct with a reference to the bbox being clipped
    explicit inline SurfaceClipperImpl(BBox* bbox);

    //// OPERATION ////

    template<Axis T>
    void operator()(PlaneAligned<T> const&) const;

    template<Axis T>
    void operator()(CylCentered<T> const&) const;

    void operator()(SphereCentered const&) const;

    template<Axis T>
    void operator()(CylAligned<T> const&) const;

    void operator()(Sphere const&) const;

    //! All unspecified surfaces are null-ops
    template<class S>
    void operator()(S const&) const
    {
    }

  private:
    BBox* bbox_;
};

//---------------------------------------------------------------------------//
/*!
 * Truncate a bounding box to its intersection with a surface exterior.
 *
 * Even though most quadric surfaces are infinite, their intersection with a
 * bounding box may be a smaller bounding box. This operation accelerates
 * "distance to in" calculations.
 */
template<>
class SurfaceClipperImpl<Sense::outside>
{
  public:
    // Construct with a reference to the bbox being clipped
    explicit inline SurfaceClipperImpl(BBox* bbox);

    template<Axis T>
    void operator()(PlaneAligned<T> const&) const;

    //! All unspecified surfaces are null-ops
    template<class S>
    void operator()(S const&) const
    {
    }

  private:
    BBox* bbox_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a bounding box reference.
 */
SurfaceClipperImpl<Sense::inside>::SurfaceClipperImpl(BBox* bbox) : bbox_{bbox}
{
    CELER_EXPECT(bbox);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a bounding box reference.
 */
SurfaceClipperImpl<Sense::outside>::SurfaceClipperImpl(BBox* bbox)
    : bbox_{bbox}
{
    CELER_EXPECT(bbox);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
