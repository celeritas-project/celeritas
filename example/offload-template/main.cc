//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/main.cc
//! \brief Minimal Geant4 application with Celeritas offloading
//---------------------------------------------------------------------------//
#include <memory>
#include <FTFP_BERT.hh>
#include <G4RunManagerFactory.hh>
#include <accel/AlongStepFactory.hh>
#include <accel/SetupOptions.hh>
#include <accel/TrackingManagerConstructor.hh>
#include <accel/TrackingManagerIntegration.hh>

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"

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
    celeritas::SetupOptions& so = tmi.Options();

    so.max_num_tracks = 1024 * 16;
    so.initializer_capacity = 1024 * 128 * 4;
    so.secondary_stack_factor = 2.0;
    so.ignore_processes = {"CoulombScat", "Rayl"};  // Ignored processes

    // Set along-step factory with zero field
    so.make_along_step = celeritas::UniformAlongStepFactory();

    // Save diagnostic information
    so.output_file = "celeritas-offload-diagnostic.json";

    // Initialize physics with celeritas offload
    auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
    physics_list->RegisterPhysics(
        new celeritas::TrackingManagerConstructor(&tmi));
    run_manager->SetUserInitialization(physics_list);

    // Initialize geometry and actions
    run_manager->SetUserInitialization(new DetectorConstruction());
    run_manager->SetUserInitialization(new ActionInitialization());

    // Run one event
    run_manager->Initialize();
    run_manager->BeamOn(1);
    return EXIT_SUCCESS;
}
