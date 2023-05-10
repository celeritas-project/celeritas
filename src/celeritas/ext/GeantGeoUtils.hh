//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Load a GDML file and return the world volume (Geant4 owns!)
G4VPhysicalVolume* load_geant_geometry(std::string const& gdml_filename);

//---------------------------------------------------------------------------//
// Reset all Geant4 geometry stores if *not* using RunManager
void reset_geant_geometry();

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline G4VPhysicalVolume* load_geant_geometry(std::string const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline void reset_geant_geometry()
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
