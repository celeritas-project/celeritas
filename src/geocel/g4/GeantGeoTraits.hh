//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "geocel/GeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeantGeoParams;
class GeantGeoTrackView;
template<Ownership W, MemSpace M>
struct GeantGeoParamsData;
template<Ownership W, MemSpace M>
struct GeantGeoStateData;

#if CELERITAS_USE_GEANT4
//---------------------------------------------------------------------------//
/*!
 * Traits specialization for Geant4 geometry.
 */
template<>
struct GeoTraits<GeantGeoParams>
{
    //! Params data used during runtime
    template<Ownership W, MemSpace M>
    using ParamsData = GeantGeoParamsData<W, M>;

    //! State data used during runtime
    template<Ownership W, MemSpace M>
    using StateData = GeantGeoStateData<W, M>;

    //! Geometry track view
    using TrackView = GeantGeoTrackView;

    //! Descriptive name for the geometry
    static constexpr char const name[] = "Geant4";

    //! TO BE REMOVED: "native" file extension for this geometry
    static constexpr char const ext[] = ".gdml";
};
#else
//! Geant4 is unavailable
template<>
struct GeoTraits<GeantGeoParams> : NotConfiguredGeoTraits
{
};
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
