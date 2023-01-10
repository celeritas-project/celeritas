//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/LoadGdml.cc
//---------------------------------------------------------------------------//
#include "LoadGdml.hh"

#include <G4GDMLParser.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Delete a geant4 volume pointer.
 */
void detail::PVDeleter::operator()(G4VPhysicalVolume* vol) const
{
    delete vol;
}

//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Load a gdml input file, creating a pointer with ownership semantics.
 */
UPG4PhysicalVolume load_gdml(std::string const& filename)
{
    CELER_LOG(info) << "Loading Geant4 geometry from GDML at " << filename;

    // Create parser; do *not* strip `0x` extensions since those are needed to
    // deduplicate complex geometries (e.g. CMS) and are handled by the Label
    // and LabelIdMultiMap. Note that material and element names (at least as
    // of Geant4@11.0) are *always* stripped: only volumes and solids keep
    // their extension.
    G4GDMLParser gdml_parser;
    gdml_parser.SetStripFlag(false);

    constexpr bool validate_gdml_schema = false;
    gdml_parser.Read(filename, validate_gdml_schema);

    UPG4PhysicalVolume result(gdml_parser.GetWorldVolume());
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
