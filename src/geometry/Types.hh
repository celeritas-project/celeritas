//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
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
//! Identifier for a geometry volume
using VolumeId = OpaqueId<struct Volume>;

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
 * Enhanced intersection for curved-line travel.
 */
struct NearbyIntersection
{
    real_type distance; //!< Distance along a straight-line direction
    real_type safety;   //!< Known-safe distance along any direction
};

//---------------------------------------------------------------------------//
/*!
 * Result of a propagation step.
 */
struct Propagation
{
    real_type distance{0}; //!< Distance traveled
    bool boundary{false};  //!< True if hit a boundary before given distance
};

//---------------------------------------------------------------------------//
} // namespace celeritas
