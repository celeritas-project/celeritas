//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>
#include <unordered_set>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"

class G4LogicalVolume;
class G4VPhysicalVolume;
class G4VTouchable;

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Wrap around a touchable to get a descriptive output.
struct PrintableNavHistory
{
    G4VTouchable const* touch{nullptr};
};

//---------------------------------------------------------------------------//
//! Wrap around a G4LogicalVolume to get a descriptive output.
struct PrintableLV
{
    G4LogicalVolume const* lv{nullptr};
};

// Print detailed information about the touchable history.
std::ostream& operator<<(std::ostream& os, PrintableNavHistory const& pnh);

// Print the logical volume name, ID, and address.
std::ostream& operator<<(std::ostream& os, PrintableLV const& pnh);

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Load a GDML file and return the world volume (Geant4 owns!)
G4VPhysicalVolume* load_geant_geometry(std::string const& gdml_filename);

// Load a GDML file, stripping pointers
G4VPhysicalVolume* load_geant_geometry_native(std::string const& gdml_filename);

//---------------------------------------------------------------------------//
// Reset all Geant4 geometry stores if *not* using RunManager
void reset_geant_geometry();

//---------------------------------------------------------------------------//
// Get a view to the Geant4 LV store
Span<G4LogicalVolume*> geant_logical_volumes();

// Find Geant4 logical volumes corresponding to a list of names
std::unordered_set<G4LogicalVolume const*>
    find_geant_volumes(std::unordered_set<std::string>);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline G4VPhysicalVolume* load_geant_geometry(std::string const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline G4VPhysicalVolume* load_geant_geometry_native(std::string const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline void reset_geant_geometry()
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline Span<G4LogicalVolume*> geant_logical_volumes()
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
