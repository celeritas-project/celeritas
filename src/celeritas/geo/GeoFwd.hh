//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoFwd.hh
//! \brief Forward-declare configure-time geometry implementation
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// FORWARD DECLARATIONS
//---------------------------------------------------------------------------//
class VecgeomParams;
template<Ownership, MemSpace>
struct VecgeomParamsData;
template<Ownership, MemSpace>
struct VecgeomStateData;
class VecgeomTrackView;

class OrangeParams;
template<Ownership, MemSpace>
struct OrangeStateData;
template<Ownership, MemSpace>
struct OrangeParamsData;
class OrangeTrackView;

class GeantGeoParams;
template<Ownership, MemSpace>
struct GeantGeoStateData;
template<Ownership, MemSpace>
struct GeantGeoParamsData;
class GeantGeoTrackView;

//---------------------------------------------------------------------------//
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
using GeoParams = VecgeomParams;
using GeoTrackView = VecgeomTrackView;
template<Ownership W, MemSpace M>
using GeoParamsData = VecgeomParamsData<W, M>;
template<Ownership W, MemSpace M>
using GeoStateData = VecgeomStateData<W, M>;

#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
using GeoParams = OrangeParams;
using GeoTrackView = OrangeTrackView;
template<Ownership W, MemSpace M>
using GeoParamsData = OrangeParamsData<W, M>;
template<Ownership W, MemSpace M>
using GeoStateData = OrangeStateData<W, M>;

#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
using GeoParams = GeantGeoParams;
using GeoTrackView = GeantGeoTrackView;
template<Ownership W, MemSpace M>
using GeoParamsData = GeantGeoParamsData<W, M>;
template<Ownership W, MemSpace M>
using GeoStateData = GeantGeoStateData<W, M>;
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
