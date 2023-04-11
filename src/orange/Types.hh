//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTypes.hh
//! Shared (VecGeom + ORANGE) geometry type definitions.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Identifier for a geometry volume
using VolumeId = OpaqueId<struct Volume>;

//! Local identifier for a geometry volume in a universe
using LocalVolumeId = OpaqueId<struct LocalVolume>;

//! Identifier for a surface (for surface-based geometries)
using SurfaceId = OpaqueId<struct Surface>;

//! Local Identifier for a surface within a universe
using LocalSurfaceId = OpaqueId<struct LocalSurface>;

//! Fixed-size array for 3D space
using Real3 = Array<real_type, 3>;

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//
/*!
 * Enumeration for cartesian axes.
 */
enum class Axis
{
    x,  //!< X axis/I index coordinate
    y,  //!< Y axis/J index coordinate
    z,  //!< Z axis/K index coordinate
    size_  //!< Sentinel value for looping over axes
};

//---------------------------------------------------------------------------//
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * Data required to initialize a geometry state.
 */
struct GeoTrackInitializer
{
    Real3 pos;
    Real3 dir;
};

//---------------------------------------------------------------------------//
/*!
 * Result of a propagation step.
 *
 * The boundary flag means that the geometry is step limiting, but the surface
 * crossing must be called externally.
 */
struct Propagation
{
    real_type distance{0};  //!< Distance traveled
    bool boundary{false};  //!< True if hit a boundary before given distance
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//
//! Get the lowercase name of the axis.
inline constexpr char to_char(Axis ax)
{
    return "xyz\a"[static_cast<int>(ax)];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
