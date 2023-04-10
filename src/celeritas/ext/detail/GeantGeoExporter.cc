//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantGeoExporter.cc
//---------------------------------------------------------------------------//
#include "GeantGeoExporter.hh"

#include <algorithm>
#include <G4GDMLParser.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Threading.hh>
#include <G4VPhysicalVolume.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct a name for a temporary GDML output filename.
 */
std::string GeantGeoExporter::make_tmpfile_name()
{
    // TODO: use std::filesystem, RNG seeded by clock, PID, something.
    return "geant-celeritas-export.gdml";
}

//---------------------------------------------------------------------------//
/*!
 * Create with a reference to the world volume.
 */
GeantGeoExporter::GeantGeoExporter(G4VPhysicalVolume const* world)
    : world_(world)
{
    CELER_EXPECT(world_);
}

//---------------------------------------------------------------------------//
/*!
 * Write out the given geometry to a file.
 *
 * This should be called only by the master thread. (It's a null-op in the GDML
 * parser for other threads, but we should be explicit.)
 */
void GeantGeoExporter::operator()(std::string const& filename) const
{
    CELER_EXPECT(G4Threading::IsMasterThread());

    CELER_LOG(status) << "Exporting Geant4 geometry to GDML";
    ScopedTimeAndRedirect time_and_output_("G4GDMLParser");

    G4GDMLParser parser;
    parser.SetEnergyCutsExport(false);
    parser.SetSDExport(false);
    parser.SetOverlapCheck(false);
#if G4VERSION_NUMBER >= 1070
    parser.SetOutputFileOverwrite(true);
#endif

    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    CELER_ASSERT(lv_store);
    bool all_uniquified = std::all_of(
        lv_store->begin(), lv_store->end(), [](G4LogicalVolume* lv) {
            std::string const& name = lv->GetName();
            return name.find("0x") != std::string::npos;
        });

    parser.Write(filename, world_, /* append_pointers = */ !all_uniquified);
    CELER_LOG(debug) << "Wrote temporary GDML to " << filename
                     << (all_uniquified ? "without " : "with")
                     << "additional pointer suffix";
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
