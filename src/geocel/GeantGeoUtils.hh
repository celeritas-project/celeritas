//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>
#include <unordered_set>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Label.hh"

//---------------------------------------------------------------------------//
// Forward declarations
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Navigator;
class G4NavigationHistory;

#if CELERITAS_GEANT4_VERSION >= 0x0b0200
// Geant4 11.2 removed G4VTouchable
class G4TouchableHistory;
#else
class G4VTouchable;
#endif

namespace celeritas
{
struct GeantPhysicalInstance;
//---------------------------------------------------------------------------//
#if CELERITAS_GEANT4_VERSION >= 0x0b0200
//! Version-independent typedef to Geant4 touchable history
using GeantTouchableBase = G4TouchableHistory;
#else
using GeantTouchableBase = G4VTouchable;
#endif

//---------------------------------------------------------------------------//
//! Wrap around a touchable to get a descriptive output.
struct PrintableNavHistory
{
    G4NavigationHistory const* nav{nullptr};
};

//---------------------------------------------------------------------------//
//! Wrap around a G4LogicalVolume to get a descriptive output.
struct PrintableLV
{
    G4LogicalVolume const* lv{nullptr};
};

// Print detailed information about the touchable history.
std::ostream& operator<<(std::ostream&, PrintableNavHistory const&);

// Print the logical volume name, ID, and address.
std::ostream& operator<<(std::ostream&, PrintableLV const&);

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Write a GDML file to the given filename
void save_gdml(G4VPhysicalVolume const* world, std::string const& out_filename);

//---------------------------------------------------------------------------//
// Reset all Geant4 geometry stores if *not* using RunManager
void reset_geant_geometry();

//---------------------------------------------------------------------------//
// Get a view to the Geant4 LV store
Span<G4LogicalVolume*> geant_logical_volumes();

//---------------------------------------------------------------------------//
// Get the world volume if the geometry has been set up
G4VPhysicalVolume const* geant_world_volume();

//---------------------------------------------------------------------------//
// Whether the volume is a replica/parameterization
bool is_replica(G4VPhysicalVolume const&);

//---------------------------------------------------------------------------//
// Find Geant4 logical volumes corresponding to a list of names
std::unordered_set<G4LogicalVolume const*>
    find_geant_volumes(std::unordered_set<std::string>);

// Get a reproducible vector of LV instance ID -> label from the given world
std::vector<Label> make_logical_vol_labels(G4VPhysicalVolume const& world);

// Get a reproducible vector of PV instance ID -> label from the given world
std::vector<Label> make_physical_vol_labels(G4VPhysicalVolume const& world);

//---------------------------------------------------------------------------//
// Update a nav history to match the given volume instance stack
void set_history(Span<GeantPhysicalInstance const> stack,
                 G4NavigationHistory* nav);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline void reset_geant_geometry()
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline Span<G4LogicalVolume*> geant_logical_volumes()
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline std::unordered_set<G4LogicalVolume const*>
find_geant_volumes(std::unordered_set<std::string>)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline std::ostream& operator<<(std::ostream&, PrintableNavHistory const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline std::ostream& operator<<(std::ostream&, PrintableLV const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline bool is_replica(G4VPhysicalVolume const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
