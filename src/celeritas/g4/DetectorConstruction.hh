//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <G4VUserDetectorConstruction.hh>

#include "geocel/GeantGdmlLoader.hh"

class G4VSensitiveDetector;
class G4LogicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load a GDML file and construct sensitive detectors.
 *
 * - In \c Construct on the main thread, load the GDML file (including
 *   detectors)
 * - In \c ConstructSDandField on each worker thread, call the SDBuilder for
 *   each distinct SD name, for all LVs that share the SD name
 *
 * \internal
 * \todo This is extracted from app/celer-g4; it should be extended and will
 * replace the app/celer-g4 construction.
 */
class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    //!@{
    //! \name Type aliases
    using UPSD = std::unique_ptr<G4VSensitiveDetector>;
    using SDBuilder = std::function<UPSD(std::string const&)>;
    //!@}

  public:
    // Build with a GDML filename and a function to construct sensitive dets
    DetectorConstruction(std::string const& filename,
                         SDBuilder build_worker_sd);

    // Load the geometry on the main thread
    G4VPhysicalVolume* Construct() override;

    // Build SDs on worker threads
    void ConstructSDandField() override;

    //! Get the filename used by the GDML loader
    std::string const& filename() const { return filename_; }

    //! Access the constructed world
    G4VPhysicalVolume* world() const { return world_; }

  private:
    //// TYPES ////

    using MapDetectors = GeantGdmlLoader::MapDetectors;
    using MapDetCIter = MapDetectors::const_iterator;

    //// DATA ////

    // Construction arguments
    std::string filename_;
    SDBuilder build_worker_sd_;

    // Built during Construct
    G4VPhysicalVolume* world_{nullptr};
    MapDetectors detectors_;

    //// METHODS ////

    void build_worker_sd() const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
