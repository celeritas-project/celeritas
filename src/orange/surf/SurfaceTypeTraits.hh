//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceTypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/EnumClassUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

#include "SurfaceFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map surface enumeration to surface type.
 *
 * This class can be passed as a "tag" to functors that can then retrieve its
 * value or the associated class. It can be implicitly converted into a
 * SurfaceType enum for use in template parameters.
 */
template<SurfaceType S>
struct SurfaceTypeTraits;

#define ORANGE_SURFACE_TRAITS(ENUM_VALUE, CLS)                          \
    template<>                                                          \
    struct SurfaceTypeTraits<SurfaceType::ENUM_VALUE>                   \
        : public EnumToClass<SurfaceType, SurfaceType::ENUM_VALUE, CLS> \
    {                                                                   \
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
/*!
 * Expand a macro to a switch statement over all possible surface types.
 *
 * The \c func argument should be a functor that takes a single argument which
 * is a SurfaceTypeTraits instance.
 */
template<class F>
CELER_CONSTEXPR_FUNCTION decltype(auto)
visit_surface_type(F&& func, SurfaceType st)
{
#define ORANGE_ST_VISIT_CASE(TYPE)          \
    case SurfaceType::TYPE:                 \
        return celeritas::forward<F>(func)( \
            SurfaceTypeTraits<SurfaceType::TYPE>{})

    switch (st)
    {
        ORANGE_ST_VISIT_CASE(px);
        ORANGE_ST_VISIT_CASE(py);
        ORANGE_ST_VISIT_CASE(pz);
        ORANGE_ST_VISIT_CASE(cxc);
        ORANGE_ST_VISIT_CASE(cyc);
        ORANGE_ST_VISIT_CASE(czc);
        ORANGE_ST_VISIT_CASE(sc);
        ORANGE_ST_VISIT_CASE(cx);
        ORANGE_ST_VISIT_CASE(cy);
        ORANGE_ST_VISIT_CASE(cz);
        ORANGE_ST_VISIT_CASE(p);
        ORANGE_ST_VISIT_CASE(s);
        ORANGE_ST_VISIT_CASE(kx);
        ORANGE_ST_VISIT_CASE(ky);
        ORANGE_ST_VISIT_CASE(kz);
        ORANGE_ST_VISIT_CASE(sq);
        ORANGE_ST_VISIT_CASE(gq);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#undef ORANGE_ST_VISIT_CASE
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
