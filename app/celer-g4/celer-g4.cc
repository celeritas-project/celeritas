//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/celer-g4.cc
//---------------------------------------------------------------------------//

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <CLHEP/Random/Random.h>
#include <G4RunManager.hh>
#include <G4UIExecutive.hh>
#include <G4UImanager.hh>
#include <G4Version.hh>

#include "accel/HepMC3PrimaryGenerator.hh"

#include "GlobalSetup.hh"

#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#else
#    include <G4MTRunManager.hh>
#endif
#if G4VERSION_NUMBER >= 1060
#    include <G4GlobalConfig.hh>
#endif

#include <FTFP_BERT.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/ext/detail/GeantPhysicsList.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/Logger.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "GlobalSetup.hh"
#include "HepMC3PrimaryGeneratorAction.hh"
#include "PGPrimaryGeneratorAction.hh"

using namespace std::literals::string_view_literals;

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
void run(int argc, char** argv)
{
    ScopedRootErrorHandler scoped_root_error;

    // Set the random seed *before* the run manager is instantiated
    // (G4MTRunManager constructor uses the RNG)
    CLHEP::HepRandom::setTheSeed(0xcf39c1fa9a6e29bcul);

    std::unique_ptr<G4RunManager> run_manager;
#if G4VERSION_NUMBER >= 1100
#    ifdef G4MULTITHREADED
    auto default_rmt = G4RunManagerType::MT;
#    else
    auto default_rmt = G4RunManagerType::Serial;
#    endif
    run_manager.reset(G4RunManagerFactory::CreateRunManager(default_rmt));
#elif defined(G4MULTITHREADED)
    run_manager = std::make_unique<G4MTRunManager>();
#else
    run_manager = std::make_unique<G4RunManager>();
#endif
    CELER_ASSERT(run_manager);
    self_logger() = MakeMTLogger(*run_manager);
    CELER_LOG(info) << "Run manager type: "
                    << TypeDemangler<G4RunManager>{}(*run_manager);

    // Construct singleton, also making it available to UI
    GlobalSetup::Instance();

    // Read user input
    std::string_view filename{argv[1]};
    if (filename == "--interactive")
    {
        G4UIExecutive exec(argc, argv);
        exec.SessionStart();
        return;
    }
    if (ends_with(filename, ".mac"sv))
    {
        CELER_LOG(status) << "Executing macro commands from '" << filename
                          << "'";
        G4UImanager* ui = G4UImanager::GetUIpointer();
        CELER_ASSERT(ui);
        ui->ApplyCommand(std::string("/control/execute ")
                         + std::string(filename));
    }
    else
    {
        CELER_LOG(status) << "Reading JSON input from '" << filename << "'";
        GlobalSetup::Instance()->ReadInput(std::string(filename));
    }

    std::vector<std::string> ignore_processes = {"CoulombScat"};
    if (G4VERSION_NUMBER >= 1110)
    {
        CELER_LOG(warning) << "Default Rayleigh scattering 'MinKinEnergyPrim' "
                              "is not compatible between Celeritas and "
                              "Geant4@11.1: disabling Rayleigh scattering";
        ignore_processes.push_back("Rayl");
    }
    GlobalSetup::Instance()->SetIgnoreProcesses(ignore_processes);

    // Construct geometry, SD factory, physics, actions
    run_manager->SetUserInitialization(new DetectorConstruction{});
    switch (GlobalSetup::Instance()->GetPhysicsList())
    {
        case PhysicsListSelection::ftfp_bert: {
            run_manager->SetUserInitialization(
                new FTFP_BERT{/* verbosity = */ 0});
            break;
        }
        case PhysicsListSelection::geant_physics_list: {
            auto opts = GlobalSetup::Instance()->GetPhysicsOptions();
            if (std::find(
                    ignore_processes.begin(), ignore_processes.end(), "Rayl")
                != ignore_processes.end())
            {
                opts.rayleigh_scattering = false;
            }
            run_manager->SetUserInitialization(
                new detail::GeantPhysicsList{opts});
            break;
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }
    run_manager->SetUserInitialization(new ActionInitialization());

    // Initialize run and process events
    CELER_LOG(status) << "Initializing run manager";
    run_manager->Initialize();

    int num_events{0};
    if (!GlobalSetup::Instance()->GetEventFile().empty())
    {
        // Load the input file
        CELER_TRY_HANDLE(num_events = HepMC3PrimaryGeneratorAction::NumEvents(),
                         ExceptionConverter{"celer-g4000"});
    }
    else
    {
        num_events
            = GlobalSetup::Instance()->GetPrimaryGeneratorOptions().num_events;
    }

    if (!celeritas::getenv("CELER_DISABLE").empty())
    {
        CELER_LOG(info)
            << "Disabling Celeritas offloading since the 'CELER_DISABLE' "
               "environment variable is present and non-empty";
    }

    CELER_LOG(status) << "Transporting " << num_events << " events";
    run_manager->BeamOn(num_events);
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace app
}  // namespace celeritas

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    if (argc != 2 || argv[1] == "--help"sv || argv[1] == "-h"sv)
    {
        std::cerr << "usage: " << argv[0] << " {input}.json\n"
                  << "       " << argv[0] << " {commands}.mac\n"
                  << "       " << argv[0] << " --interactive\n"
                  << "Environment variables:\n"
                  << "  G4FORCE_RUN_MANAGER_TYPE: MT or Serial\n"
                  << "  G4FORCENUMBEROFTHREADS: set CPU worker thread count\n"
                  << "  CELER_DISABLE: nonempty disables offloading\n"
                  << "  CELER_DISABLE_DEVICE: nonempty disables CUDA\n"
                  << "  CELER_LOG: global logging level\n"
                  << "  CELER_LOG_LOCAL: thread-local logging level\n"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Run with threads and macro filename
    celeritas::app::run(argc, argv);

    CELER_LOG(status) << "Run completed successfully; exiting";
    return EXIT_SUCCESS;
}
