//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "geocel/g4/GeantGeoTraits.hh"
#include "geocel/vg/VecgeomGeoTraits.hh"
#include "orange/OrangeGeoTraits.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
//! Geometry to be used for ray tracing
enum class Geometry
{
    orange,
    vecgeom,
    geant4,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Get the user-facing CoreGeoParams class from a Geometry enum.
 */
template<Geometry>
struct GeoTraitsFromEnum;

template<>
struct GeoTraitsFromEnum<Geometry::orange>
{
    using type = OrangeParams;
};

template<>
struct GeoTraitsFromEnum<Geometry::vecgeom>
{
    using type = VecgeomParams;
};

template<>
struct GeoTraitsFromEnum<Geometry::geant4>
{
    using type = GeantGeoParams;
};

//---------------------------------------------------------------------------//
//! Type alias for accessing GeoParams class from a Geometry enum
template<Geometry G>
using GeoParams_t = typename GeoTraitsFromEnum<G>::type;

//---------------------------------------------------------------------------//
// FUNCTIONS
//---------------------------------------------------------------------------//

// Convert a geometry enum to a string
char const* to_cstring(Geometry value);

// Default memory space for rendering
MemSpace default_memspace();

//! Default geometry type for rendering
inline constexpr Geometry default_geometry()
{
    // clang-format off
    return CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE  ? Geometry::orange
         : CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM ? Geometry::vecgeom
         : CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4  ? Geometry::geant4
                                                            : Geometry::size_;
    // clang-format on
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
