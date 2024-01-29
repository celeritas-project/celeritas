//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceClipperImpl.cc
//---------------------------------------------------------------------------//
#include "SurfaceClipperImpl.hh"

#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"

#include "../CylAligned.hh"
#include "../CylCentered.hh"
#include "../PlaneAligned.hh"
#include "../Sphere.hh"
#include "../SphereCentered.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(SENSE, IN)                                    \
    template void SurfaceClipperImpl<SENSE>::operator()(IN<Axis::x> const&) \
        const;                                                              \
    template void SurfaceClipperImpl<SENSE>::operator()(IN<Axis::y> const&) \
        const;                                                              \
    template void SurfaceClipperImpl<SENSE>::operator()(IN<Axis::z> const&) \
        const

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding box to an axis-aligned plane.
 */
template<Axis T>
void SurfaceClipperImpl<Sense::inside>::operator()(PlaneAligned<T> const& s) const
{
    bbox_->clip(Bound::hi, T, s.position());
}

//!\cond
ORANGE_INSTANTIATE_OP(Sense::inside, PlaneAligned);
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding box to an axis-aligned cylinder.
 */
template<Axis T>
void SurfaceClipperImpl<Sense::inside>::operator()(CylCentered<T> const& s) const
{
    real_type radius = std::sqrt(s.radius_sq());
    for (auto ax : range(Axis::size_))
    {
        if (T != ax)
        {
            bbox_->clip(Bound::lo, ax, -radius);
            bbox_->clip(Bound::hi, ax, radius);
        }
    }
}

//!\cond
ORANGE_INSTANTIATE_OP(Sense::inside, CylCentered);
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding box to a centered sphere.
 */
void SurfaceClipperImpl<Sense::inside>::operator()(SphereCentered const& s) const
{
    real_type radius = std::sqrt(s.radius_sq());
    for (auto ax : range(Axis::size_))
    {
        bbox_->clip(Bound::lo, ax, -radius);
        bbox_->clip(Bound::hi, ax, radius);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding box to a cylinder.
 */
template<Axis T>
void SurfaceClipperImpl<Sense::inside>::operator()(CylAligned<T> const& s) const
{
    real_type radius = std::sqrt(s.radius_sq());
    auto origin = s.calc_origin();
    for (auto ax : range(Axis::size_))
    {
        if (T != ax)
        {
            bbox_->clip(Bound::lo, ax, origin[to_int(ax)] - radius);
            bbox_->clip(Bound::hi, ax, origin[to_int(ax)] + radius);
        }
    }
}

//!\cond
ORANGE_INSTANTIATE_OP(Sense::inside, CylAligned);
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding box to a sphere.
 */
void SurfaceClipperImpl<Sense::inside>::operator()(Sphere const& s) const
{
    real_type radius = std::sqrt(s.radius_sq());
    auto const& origin = s.origin();
    for (auto ax : range(Axis::size_))
    {
        bbox_->clip(Bound::lo, ax, origin[to_int(ax)] - radius);
        bbox_->clip(Bound::hi, ax, origin[to_int(ax)] + radius);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding box to an axis-aligned plane.
 */
template<Axis T>
void SurfaceClipperImpl<Sense::outside>::operator()(PlaneAligned<T> const& s) const
{
    bbox_->clip(Bound::lo, T, s.position());
}

//!\cond
ORANGE_INSTANTIATE_OP(Sense::outside, PlaneAligned);
//!\endcond

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
