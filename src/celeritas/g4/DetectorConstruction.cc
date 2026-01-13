//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include <memory>
#include <G4LogicalVolume.hh>
#include <G4SDManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSensitiveDetector.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGdmlLoader.hh"
#include "geocel/g4/Convert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas SD options during construction.
 *
 * This should be done only during the main/serial thread.
 */
DetectorConstruction::DetectorConstruction(std::string const& filename,
                                           SDBuilder build_worker_sd)

    : filename_(filename), build_worker_sd_{build_worker_sd}
{
    CELER_EXPECT(!filename_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Load geometry and sensitive detector volumes.
 *
 * This should only be called once from the master thread, toward the very
 * beginning of the program.
 */
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    CELER_LOG_LOCAL(debug) << R"(Loading detector geometry)";

    auto loaded = [this] {
        GeantGdmlLoader::Options opts;
        opts.detectors = static_cast<bool>(build_worker_sd_);
        GeantGdmlLoader load(opts);
        return load(filename_);
    }();

    if (build_worker_sd_)
    {
        if (loaded.detectors.empty())
        {
            CELER_LOG(warning)
                << R"(Detector setup is provided, but no SDs were found in the GDML file)";
            build_worker_sd_ = {};
        }
        else
        {
            CELER_LOG(debug)
                << "Found " << loaded.detectors.size() << " detectors";
        }
    }

    CELER_ASSERT(loaded.world);
    world_ = loaded.world;
    detectors_ = std::move(loaded.detectors);

    // TODO: construct shared Celeritas field

    return world_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct thread-local sensitive detectors and field.
 */
void DetectorConstruction::ConstructSDandField()
{
    if (build_worker_sd_)
    {
        this->build_worker_sd();
    }

    // TODO: construct Geant4 wrappers for Celeritas field
}

//---------------------------------------------------------------------------//
/*!
 * Construct thread-local sensitive detectors, adding to G4SDManager.
 */
void DetectorConstruction::build_worker_sd() const
{
    CELER_LOG_LOCAL(debug) << R"(Constructing sensitive detectors)";
    auto* sd_manager = G4SDManager::GetSDMpointer();

    foreach_detector(detectors_, [&](MapDetCIter start, MapDetCIter stop) {
        // Construct an SD based on the name
        UPSD sd = build_worker_sd_(start->first);
        if (!sd)
        {
            CELER_LOG(warning)
                << "No SD specified for volume '" << start->first << "'";
            return;
        }

        // Attach sensitive detectors
        for (MapDetCIter iter = start; iter != stop; ++iter)
        {
            CELER_LOG_LOCAL(debug)
                << "Attaching '" << iter->first << "'@" << sd.get()
                << " to volume '" << iter->second->GetName() << "'@"
                << static_cast<void const*>(iter->second);
            iter->second->SetSensitiveDetector(sd.get());
        }

        // Hand SD to the manager
        sd_manager->AddNewDetector(sd.release());
    });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
