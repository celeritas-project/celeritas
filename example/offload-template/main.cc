//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/main.cc
//! \brief Minimal Geant4 application with Celeritas offloading.
//---------------------------------------------------------------------------//
#include <FTFP_BERT.hh>
#include <G4RunManager.hh>

#include "src/ActionInitialization.hh"
#include "src/DetectorConstruction.hh"

//---------------------------------------------------------------------------//
/*!
 * Geant4-Celeritas offloading template.
 *
 * See README for details.
 */
int main(int argc, char* argv[])
{
    if (argc != 1)
    {
        // Print help message
        std::cout << "Usage: " << argv[0] << std::endl;
        return EXIT_FAILURE;
    }

    // Construct run manager
    G4RunManager run_manager;
    run_manager.SetVerboseLevel(1);  // Print minimal information about the run

    // Initialize physics, geometry, and actions
    run_manager.SetUserInitialization(new FTFP_BERT(/* verbosity = */ 0));
    run_manager.SetUserInitialization(new DetectorConstruction());
    run_manager.SetUserInitialization(new ActionInitialization());

    // Run one event
    run_manager.Initialize();
    run_manager.BeamOn(1);

    return EXIT_SUCCESS;
}
