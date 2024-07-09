//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SurfaceHashPoint.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "orange/surf/VariantSurface.hh"

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
 * Surfaces that *can* be soft equal *must* result in a point that is less than
 * or equal to epsilon.
 *
 * \todo We could potentially reduce the number of collisions by turning this
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

    real_type operator()(Involute const& s) const { return s.r_b(); }

    template<Axis T>
    real_type operator()(CylAligned<T> const& s) const
    {
        return std::sqrt(s.radius_sq());
    }

    real_type operator()(Plane const& p) const { return p.displacement(); }

    real_type operator()(Sphere const& s) const
    {
        return std::sqrt(s.radius_sq());
    }

    template<Axis T>
    real_type operator()(ConeAligned<T> const& s) const
    {
        return norm(s.origin());
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
