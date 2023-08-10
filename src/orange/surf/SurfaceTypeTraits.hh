//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/OrangeTypes.hh"

#include "SurfaceFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map surface enumeration to surface type.
 */
template<SurfaceType S>
struct SurfaceTypeTraits;

#define ORANGE_SURFACE_TRAITS(ENUM_VALUE, CLS)        \
    template<>                                        \
    struct SurfaceTypeTraits<SurfaceType::ENUM_VALUE> \
    {                                                 \
        using type = CLS;                             \
    }

// clang-format off
ORANGE_SURFACE_TRAITS(px,  PlaneAligned<Axis::x>);
ORANGE_SURFACE_TRAITS(py,  PlaneAligned<Axis::y>);
ORANGE_SURFACE_TRAITS(pz,  PlaneAligned<Axis::z>);
ORANGE_SURFACE_TRAITS(cxc, CylCentered<Axis::x>);
ORANGE_SURFACE_TRAITS(cyc, CylCentered<Axis::y>);
ORANGE_SURFACE_TRAITS(czc, CylCentered<Axis::z>);
ORANGE_SURFACE_TRAITS(sc,  SphereCentered);
ORANGE_SURFACE_TRAITS(cx,  CylAligned<Axis::x>);
ORANGE_SURFACE_TRAITS(cy,  CylAligned<Axis::y>);
ORANGE_SURFACE_TRAITS(cz,  CylAligned<Axis::z>);
ORANGE_SURFACE_TRAITS(p,   Plane);
ORANGE_SURFACE_TRAITS(s,   Sphere);
ORANGE_SURFACE_TRAITS(kx,  ConeAligned<Axis::x>);
ORANGE_SURFACE_TRAITS(ky,  ConeAligned<Axis::y>);
ORANGE_SURFACE_TRAITS(kz,  ConeAligned<Axis::z>);
ORANGE_SURFACE_TRAITS(sq,  SimpleQuadric);
ORANGE_SURFACE_TRAITS(gq,  GeneralQuadric);
// clang-format on

#undef ORANGE_SURFACE_TRAITS

//---------------------------------------------------------------------------//
}  // namespace celeritas
