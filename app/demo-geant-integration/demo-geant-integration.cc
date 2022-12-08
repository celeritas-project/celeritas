//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/demo-geant-integration.cc
//---------------------------------------------------------------------------//

#include <cstddef>
#include <iostream>
// #include <FTFP_BERT.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4TransportationManager.hh>
#include <Randomize.hh>

#include "celeritas/ext/GeantVersion.hh"
#if !CELERITAS_G4_V10
#    include <G4RunManagerFactory.hh>
#endif

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/LoadGdml.hh"
#include "celeritas/ext/detail/GeantPhysicsList.hh"
#include "accel/ActionInitialization.hh"
#include "accel/Logger.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 2 || args[1] == "--help" || args[1] == "-h")
    {
        std::cerr << "usage: " << args[0] << " {input}.gdml" << std::endl;
        return EXIT_FAILURE;
    }

    CLHEP::HepRandom::setTheSeed(0xcf39c1fa9a6e29bcul);

    std::unique_ptr<G4RunManager> run_manager;
#if CELERITAS_G4_V10
    run_manager = std::make_unique<G4RunManager>();
#else
    run_manager.reset(
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::MT));
#endif
    CELER_ASSERT(run_manager);

    celeritas::self_logger() = celeritas::make_mt_logger(*run_manager);

    celeritas::GeantPhysicsOptions geant_phys_opts{};
    auto opts = std::make_shared<celeritas::SetupOptions>();

    // Construct physics, geometry, celeritas setup, and user setup
    run_manager->SetUserInitialization(new demo_geant::DetectorConstruction{args[1]});
    // run_manager->SetUserInitialization(new FTFP_BERT);
    run_manager->SetUserInitialization(
        new celeritas::detail::GeantPhysicsList{geant_phys_opts});
    run_manager->SetUserInitialization(new celeritas::ActionInitialization(
        opts, std::make_unique<demo_geant::ActionInitialization>()));

    CELER_LOG_LOCAL(debug) << "G4RunManager::Initialize";
    // run_manager->SetVerboseLevel(1);
    run_manager->Initialize();

    celeritas::GeantImporter load_geant_data(
        G4TransportationManager::GetTransportationManager()
            ->GetNavigatorForTracking()
            ->GetWorldVolume());

    auto imported = load_geant_data();
    CELER_LOG_LOCAL(info) << "loaded data";

    CELER_LOG_LOCAL(status) << "beam on";
    run_manager->BeamOn(1);

    return EXIT_SUCCESS;
}
