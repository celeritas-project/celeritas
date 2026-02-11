//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SurfaceHashPoint.hh
//! \todo Inline into LocalSurfaceInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "orange/surf/detail/AllSurfaces.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a point to hash for deduplicating surfaces.
 *
 * Surfaces that \em can be soft equal \em must have a difference in hash
 * points that is less than or equal to epsilon.
 *
 * \todo We could reduce the number of collisions by turning this
 * into a two- or three-dimensional point that's then hashed in an infinite
 * grid.
 */
struct SurfaceHashPoint
{
    template<Axis T>
    real_type operator()(PlaneAligned<T> const& s) const
    {
        return s.position();
    }

    template<Axis T>
    real_type operator()(CylCentered<T> const& s) const
    {
        return std::sqrt(s.radius_sq());
    }

    real_type operator()(SphereCentered const& s) const
    {
        return std::sqrt(s.radius_sq());
    }

    template<Axis T>
    real_type operator()(CylAligned<T> const& s) const
    {
        // Usually cylinders in the same geometry have the same size, so we
        // hash on one of the orthogonal coordinates.
        return s.origin_u();
    }

    real_type operator()(Plane const& p) const { return p.displacement(); }

    real_type operator()(Sphere const& s) const
    {
        return std::sqrt(s.radius_sq());
    }

    template<Axis T>
    real_type operator()(ConeAligned<T> const& s) const
    {
        return s.origin()[to_int(s.u_axis())];
    }

    real_type operator()(Involute const& s) const
    {
        return s.displacement_angle();
    }

    real_type operator()(SimpleQuadric const& s) const
    {
        return std::sqrt(s.zeroth());
    }

    real_type operator()(GeneralQuadric const& s) const
    {
        return std::sqrt(s.zeroth());
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
