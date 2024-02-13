//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoData.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "geocel/vg/VecgeomData.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeData.hh"
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
#    include "geocel/g4/GeantGeoData.hh"
#endif

// Include type alias for Geo state/params data
#include "GeoFwd.hh"  // IWYU pragma: export
