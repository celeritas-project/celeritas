//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGdmlLoader.cc
//---------------------------------------------------------------------------//
#include "GeantGdmlLoader.hh"

#include <G4GDMLAuxStructType.hh>
#include <G4GDMLParser.hh>

#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"

#include "ScopedGeantExceptionHandler.hh"
#include "ScopedGeantLogger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load a gdml input file, creating a pointer owned by Geant4.
 *
 * Geant4's constructors for physical/logical volumes register \c this pointers
 * in the "volume stores" which can be cleared with
 * \c celeritas::reset_geant_geometry .
 *
 * Note that material and element names (at least as
 * of Geant4@11.0) are \em always stripped: only volumes and solids keep
 * their extension.
 */
auto GeantGdmlLoader::operator()(std::string const& filename) const -> Result
{
    CELER_EXPECT(!filename.empty());
    CELER_LOG(info) << "Loading Geant4 geometry from GDML at " << filename;

    if (!G4Threading::IsMasterThread())
    {
        // Always-on debug assertion (not a "runtime" error but a
        // subtle programming logic error that always causes a crash)
        CELER_DEBUG_FAIL(
            "Geant4 geometry cannot be loaded from a worker thread", internal);
    }

    ScopedMem record_mem("load_geant_geometry");
    ScopedTimeLog scoped_time;

    ScopedGeantLogger scoped_logger;
    ScopedGeantExceptionHandler scoped_exceptions;

    G4GDMLParser gdml_parser;
    gdml_parser.SetStripFlag(opts_.clean);

    gdml_parser.Read(filename, /* validate_gdml_schema = */ false);

    Result result;
    result.world = gdml_parser.GetWorldVolume();

    if (opts_.detectors)
    {
        // Find sensitive detectors
        for (auto const& lv_vecaux : *gdml_parser.GetAuxMap())
        {
            for (G4GDMLAuxStructType const& aux : lv_vecaux.second)
            {
                if (aux.type == "SensDet")
                {
                    result.detectors.insert({aux.value, lv_vecaux.first});
                }
            }
        }
    }

    CELER_ENSURE(result.world);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
