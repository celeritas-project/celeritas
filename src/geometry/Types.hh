//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//! Type definitions for geometry
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/OpaqueId.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//
//! Identifier for a geometry volume
using VolumeId  = OpaqueId<struct Volume>;
using SurfaceId = OpaqueId<struct Surface>;

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
    real_type distance{0}; //!< Distance traveled
    bool boundary{false};  //!< True if hit a boundary before given distance
};

//---------------------------------------------------------------------------//
// ENUMS
//---------------------------------------------------------------------------//
/*!
 * Enumeration for cartesian axes.
 */
enum class Axis
{
    x,    //!< X axis/I index coordinate
    y,    //!< Y axis/J index coordinate
    z,    //!< Z axis/K index coordinate
    size_ //!< Sentinel value for looping over axes
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
} // namespace celeritas
