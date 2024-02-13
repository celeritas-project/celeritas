//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoParams.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "geocel/vg/VecgeomParams.hh"  // IWYU pragma: export
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeParams.hh"  // IWYU pragma: export
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
#    include "geocel/g4/GeantGeoParams.hh"  // IWYU pragma: export
#endif

// Include type alias for GeoParams
#include "GeoFwd.hh"  // IWYU pragma: export
