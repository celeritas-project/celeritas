//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "celeritas/io/ImportData.hh"

#include "GeantSetup.hh"

// Geant4 forward declaration
class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load problem data directly from Geant4.
 *
 * This can be used to circumvent ROOT as a serialization tool, whether to
 * simplify the toolchain or to integrate better with Acceleritas.
 *
 * \code
    GeantImporter import(GeantSetup("blah.gdml"));
    ImportData data = import();
   \endcode
 * or to import from an existing, initialized Geant4 state:
 * \code
 *  GeantImport import(world_volume);
    ImportData data = import();
 *  \endcode
 */
class GeantImporter
{
  public:
    //!@{
    //! Type aliases
    //!@}

    //! Only import a subset of available Geant4 data (TODO)
    struct DataSelection
    {
        // Particle types for processes: std::function<bool(PDGNumber)>
        // Process selection (full or minimal)
    };

  public:
    // Construct from an existing Geant4 geometry, assuming physics is loaded
    explicit GeantImporter(const G4VPhysicalVolume* world);

    // Construct by capturing a GeantSetup object
    explicit GeantImporter(GeantSetup&& setup);

    // Fill data from Geant4
    ImportData operator()(const DataSelection& selection);

    //! Fill all available data from Geant4
    ImportData operator()() { return (*this)(DataSelection{}); }

  private:
    // World detector
    const G4VPhysicalVolume* world_{nullptr};
    // Optional setup if celeritas handles initialization
    GeantSetup setup_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline GeantImporter::GeantImporter(const G4VPhysicalVolume*)
{
    (void)sizeof(world_);
    CELER_NOT_CONFIGURED("Geant4");
}

inline GeantImporter::GeantImporter(GeantSetup&&)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline ImportData GeantImporter::operator()(const DataSelection&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
