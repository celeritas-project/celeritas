//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/main.cc
//! \brief Minimal Geant4 application with Celeritas offloading
//---------------------------------------------------------------------------//
#include <memory>
#include <FTFP_BERT.hh>
#include <G4RunManagerFactory.hh>
#include <accel/TrackingManagerConstructor.hh>
#include <accel/TrackingManagerIntegration.hh>

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "MakeCelerOptions.hh"

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

    std::unique_ptr<G4RunManager> run_manager;
    run_manager.reset(
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default));

    // Initialize Celeritas
    auto& tmi = celeritas::TrackingManagerIntegration::Instance();

    // Initialize physics with celeritas offload
    auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
    physics_list->RegisterPhysics(
        new celeritas::TrackingManagerConstructor(&tmi));
    run_manager->SetUserInitialization(physics_list);
    tmi.SetOptions(MakeCelerOptions());

    // Initialize geometry and actions
    run_manager->SetUserInitialization(new DetectorConstruction());
    run_manager->SetUserInitialization(new ActionInitialization());

    // Run one event
    run_manager->Initialize();
    run_manager->BeamOn(1);
    return EXIT_SUCCESS;
}
