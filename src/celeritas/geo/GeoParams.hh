//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoParams.hh
//! \brief Select geometry implementation at configure time
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
#    include "geocel/vg/VecgeomParams.hh"  // IWYU pragma: export
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
#    include "orange/OrangeParams.hh"  // IWYU pragma: export
#elif CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
#    include "geocel/GeantGeoParams.hh"  // IWYU pragma: export
#endif

// Include traits and type aliases for GeoParams
#include "GeoFwd.hh"  // IWYU pragma: export
