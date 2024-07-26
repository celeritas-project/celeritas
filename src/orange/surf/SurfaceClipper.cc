//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipper.cc
//---------------------------------------------------------------------------//
#include "SurfaceClipper.hh"

#include "corecel/Constants.hh"
#include "orange/BoundingBoxUtils.hh"

#include "detail/AllSurfaces.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(IN)                                       \
    template void SurfaceClipper::operator()(IN<Axis::x> const&) const; \
    template void SurfaceClipper::operator()(IN<Axis::y> const&) const; \
    template void SurfaceClipper::operator()(IN<Axis::z> const&) const

constexpr real_type sqrt_half = constants::sqrt_two / 2;
constexpr real_type sqrt_third = constants::sqrt_three / 2;
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with interior and exterior bounding boxes.
 */
SurfaceClipper::SurfaceClipper(BBox* interior, BBox* exterior)
    : int_{interior}, ext_{exterior}
{
    CELER_EXPECT(int_ && ext_);
    CELER_EXPECT(encloses(*ext_, *int_));
}

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to an axis-aligned plane.
 */
template<Axis T>
void SurfaceClipper::operator()(PlaneAligned<T> const& s) const
{
    int_->shrink(Bound::hi, T, s.position());
    ext_->shrink(Bound::hi, T, s.position());
}

//!\cond
ORANGE_INSTANTIATE_OP(PlaneAligned);
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to an axis-aligned cylinder.
 */
template<Axis T>
void SurfaceClipper::operator()(CylCentered<T> const& s) const
{
    return (*this)(CylAligned<T>{s});
}

//!\cond
ORANGE_INSTANTIATE_OP(CylCentered);
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to a centered sphere.
 */
void SurfaceClipper::operator()(SphereCentered const& s) const
{
    return (*this)(Sphere{s});
}

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to a cylinder.
 */
template<Axis T>
void SurfaceClipper::operator()(CylAligned<T> const& s) const
{
    real_type radius = std::sqrt(s.radius_sq());
    auto origin = s.calc_origin();
    for (auto ax : range(Axis::size_))
    {
        if (T != ax)
        {
            int_->shrink(
                Bound::lo, ax, origin[to_int(ax)] - sqrt_half * radius);
            int_->shrink(
                Bound::hi, ax, origin[to_int(ax)] + sqrt_half * radius);
            ext_->shrink(Bound::lo, ax, origin[to_int(ax)] - radius);
            ext_->shrink(Bound::hi, ax, origin[to_int(ax)] + radius);
        }
    }
}

//!\cond
ORANGE_INSTANTIATE_OP(CylAligned);
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to a plane.
 */
void SurfaceClipper::operator()(Plane const&) const
{
    // We no longer can guarantee any point being inside the shape; reset it
    *int_ = BoundingBox{};
    // TODO: check for being in an axial plane and leave the orthogonal plane's
    // extents intact
}

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to a sphere.
 */
void SurfaceClipper::operator()(Sphere const& s) const
{
    real_type radius = std::sqrt(s.radius_sq());
    auto const& origin = s.origin();
    for (auto ax : range(Axis::size_))
    {
        int_->shrink(Bound::lo, ax, origin[to_int(ax)] - sqrt_third * radius);
        int_->shrink(Bound::hi, ax, origin[to_int(ax)] + sqrt_third * radius);
        ext_->shrink(Bound::lo, ax, origin[to_int(ax)] - radius);
        ext_->shrink(Bound::hi, ax, origin[to_int(ax)] + radius);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to a cone.
 */
template<Axis T>
void SurfaceClipper::operator()(ConeAligned<T> const&) const
{
    // We no longer can guarantee any point being inside the shape; reset it
    *int_ = BoundingBox{};
}

//!\cond
ORANGE_INSTANTIATE_OP(ConeAligned);
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to a simple quadric.
 */
void SurfaceClipper::operator()(SimpleQuadric const&) const
{
    // We no longer can guarantee any point being inside the shape; reset it
    *int_ = BoundingBox{};
}

//---------------------------------------------------------------------------//
/*!
 * Clip the bounding boxes to a general quadric.
 */
void SurfaceClipper::operator()(GeneralQuadric const&) const
{
    // We no longer can guarantee any point being inside the shape; reset it
    *int_ = BoundingBox{};
}

//---------------------------------------------------------------------------//
/*!
 * Clip a variant surface.
 */
void SurfaceClipper::operator()(VariantSurface const& surf) const
{
    CELER_ASSUME(!surf.valueless_by_exception());
    return std::visit(*this, surf);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
