//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantGeoExporter.cc
//---------------------------------------------------------------------------//
#include "GeantGeoExporter.hh"

#include <G4GDMLParser.hh>
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
    parser.SetEnergyCutsExport(true);
    parser.SetSDExport(true);
    parser.SetOverlapCheck(true);
#if G4VERSION_NUMBER > 1070
    parser.SetOutputFileOverwrite(true);
#endif
    constexpr bool append_pointers = true;
    parser.Write(filename, world_, append_pointers);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
