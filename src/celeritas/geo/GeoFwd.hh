//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoFwd.hh
//! \brief Forward-declare configure-time geometry implementation
//! \todo v1.0: Rename CoreGeoFwd.hh, rename type aliases
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "geocel/g4/GeantGeoTraits.hh"
#include "geocel/vg/VecgeomGeoTraits.hh"
#include "orange/OrangeGeoTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

//!@{
//! \name Core geometry type aliases
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
using CoreGeoParams = VecgeomParams;
using CoreGeoTrackView = VecgeomTrackView;
template<Ownership W, MemSpace M>
using CoreGeoParamsData = VecgeomParamsData<W, M>;
template<Ownership W, MemSpace M>
using CoreGeoStateData = VecgeomStateData<W, M>;

#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
using CoreGeoParams = OrangeParams;
using CoreGeoTrackView = OrangeTrackView;
template<Ownership W, MemSpace M>
using CoreGeoParamsData = OrangeParamsData<W, M>;
template<Ownership W, MemSpace M>
using CoreGeoStateData = OrangeStateData<W, M>;

#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
using CoreGeoParams = GeantGeoParams;
using CoreGeoTrackView = GeantGeoTrackView;
template<Ownership W, MemSpace M>
using CoreGeoParamsData = GeantGeoParamsData<W, M>;
template<Ownership W, MemSpace M>
using CoreGeoStateData = GeantGeoStateData<W, M>;
#endif
//!@}

//---------------------------------------------------------------------------//

//!@{
//! \name Old geometry type aliases
//! \deprecated Use CoreX instead
using GeoParams = CoreGeoParams;
using GeoTrackView = CoreGeoTrackView;
template<Ownership W, MemSpace M>
using GeoParamsData = CoreGeoParamsData<W, M>;
template<Ownership W, MemSpace M>
using GeoStateData = CoreGeoStateData<W, M>;
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
