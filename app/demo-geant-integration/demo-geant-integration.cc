//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/demo-geant-integration.cc
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <CLHEP/Random/Random.h>
#include <G4RunManager.hh>
#include <G4UImanager.hh>

#include "celeritas/ext/GeantConfig.hh"
#if !CELERITAS_G4_V10
#    include <G4RunManagerFactory.hh>
#else
#    include <G4MTRunManager.hh>
#endif

#include <FTFP_BERT.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/Logger.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "GlobalSetup.hh"
#include "PrimaryGeneratorAction.hh"

namespace
{
//---------------------------------------------------------------------------//
void run(std::string const& macro_filename)
{
    // Set the random seed *before* the run manager is instantiated
    // (G4MTRunManager constructor uses the RNG)
    CLHEP::HepRandom::setTheSeed(0xcf39c1fa9a6e29bcul);

    std::unique_ptr<G4RunManager> run_manager;
    if constexpr (!CELERITAS_G4_V10)
    {
        run_manager.reset(G4RunManagerFactory::CreateRunManager(
            CELERITAS_G4_MT ? G4RunManagerType::MT : G4RunManagerType::Serial));
    }
    else if constexpr (CELERITAS_G4_MT)
    {
        run_manager = std::make_unique<G4MTRunManager>();
    }
    else
    {
        run_manager = std::make_unique<G4RunManager>();
    }
    CELER_ASSERT(run_manager);
    celeritas::self_logger() = celeritas::make_mt_logger(*run_manager);
    CELER_LOG(info) << "Run manager type: "
                    << celeritas::TypeDemangler<G4RunManager>{}(*run_manager);

    // Construct geometry, SD factory, physics, actions
    run_manager->SetUserInitialization(new demo_geant::DetectorConstruction{});
    run_manager->SetUserInitialization(new FTFP_BERT{/* verbosity = */ 0});
    run_manager->SetUserInitialization(new demo_geant::ActionInitialization());

    demo_geant::GlobalSetup::Instance()->SetIgnoreProcesses(
        {"CoulombScat", "muIoni", "muBrems", "muPairProd"});

    G4UImanager* ui = G4UImanager::GetUIpointer();
    CELER_ASSERT(ui);
    CELER_LOG(status) << "Executing macro commands from '" << macro_filename
                      << "'";
    ui->ApplyCommand("/control/execute " + macro_filename);

    // Initialize run and process events
    run_manager->Initialize();

    // Load the input file
    int num_events{0};
    CELER_TRY_HANDLE(
        num_events = demo_geant::PrimaryGeneratorAction::NumEvents(),
        celeritas::ExceptionConverter{"demo-geant000"});

    run_manager->BeamOn(num_events);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 2 || args[1] == "--help" || args[1] == "-h")
    {
        std::cerr << "usage: " << args[0] << " {commands}.mac\n"
                  << "Environment variables:\n"
                  << "  G4FORCE_RUN_MANAGER_TYPE: MT or Serial\n"
                  << "  G4FORCENUMBEROFTHREADS: set CPU worker thread count\n"
                  << "  CELER_DISABLE_DEVICE: nonempty disables CUDA\n"
                  << "  CELER_LOG: global logging level\n"
                  << "  CELER_LOG_LOCAL: thread-local logging level\n"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Run with threads and macro filename
    run(args[1]);

    CELER_LOG(status) << "Run completed successfully; exiting";
    return EXIT_SUCCESS;
}
