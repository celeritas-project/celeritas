//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/celer-g4.cc
//---------------------------------------------------------------------------//

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <CLHEP/Random/Random.h>
#include <FTFP_BERT.hh>
#include <G4ParticleTable.hh>
#include <G4RunManager.hh>
#include <G4Version.hh>

#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#else
#    include <G4MTRunManager.hh>
#endif
#if G4VERSION_NUMBER >= 1060
#    include <G4GlobalConfig.hh>
#endif

#include <nlohmann/json.hpp>

#include "corecel/Config.hh"
#include "corecel/Version.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TracingSession.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"
#include "celeritas/ext/EmPhysicsList.hh"
#include "celeritas/ext/FtfpBertPhysicsList.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "accel/SharedParams.hh"

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "GlobalSetup.hh"
#include "LogHandlers.hh"
#include "RunInputIO.json.hh"

using namespace std::literals::string_view_literals;

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
void print_usage(std::string_view exec_name)
{
    // clang-format off
    std::cerr << "usage: " << exec_name << " {input}.json\n"
                 "       " << exec_name << " -\n"
                 "       " << exec_name << " [--help|-h]\n"
                 "       " << exec_name << " --version\n"
                 "       " << exec_name << " --dump-default\n"
                 "Environment variables:\n"
                 "  G4FORCE_RUN_MANAGER_TYPE: MT or Serial\n"
                 "  G4FORCENUMBEROFTHREADS: set CPU worker thread count\n"
                 "  CELER_DISABLE: nonempty disables offloading\n"
                 "  CELER_DISABLE_DEVICE: nonempty disables CUDA\n"
                 "  CELER_DISABLE_ROOT: nonempty disables ROOT I/O\n"
                 "  CELER_KILL_OFFLOAD: nonempty kills offload tracks\n"
                 "  CELER_LOG: global logging level\n"
                 "  CELER_LOG_LOCAL: thread-local logging level\n"
              << std::endl;
    // clang-format on
}

//---------------------------------------------------------------------------//
void run(std::string_view filename, std::shared_ptr<SharedParams> params)
{
    CELER_VALIDATE(filename != "--interactive",
                   << "Interactive celer-g4 was removed in v0.6");

    // Disable external error handlers
    ScopedRootErrorHandler scoped_root_errors;
    disable_geant_signal_handler();

    // Set the random seed *before* the run manager is instantiated
    // (G4MTRunManager constructor uses the RNG)
    CLHEP::HepRandom::setTheSeed(0xcf39c1fa9a6e29bcul);

    // Construct global setup singleton and make options available to UI
    auto& setup = *GlobalSetup::Instance();
    // Read user input
    setup.ReadInput(std::string(filename));

    // Start tracing session
    celeritas::TracingSession tracing{setup.input().tracing_file};
    tracing.start();

    auto run_manager = [] {
        // Run manager writes output that cannot be redirected with
        // GeantLoggerAdapter: capture all output from this section
        ScopedTimeAndRedirect scoped_time{"G4RunManager"};
        ScopedGeantExceptionHandler scoped_exceptions;

        // Access the particle table before creating the run manager, so that
        // missing environment variables like G4ENSDFSTATEDATA get caught
        // cleanly rather than segfaulting
        G4ParticleTable::GetParticleTable();

#if G4VERSION_NUMBER >= 1100
#    ifdef G4MULTITHREADED
        auto default_rmt = G4RunManagerType::MT;
#    else
        auto default_rmt = G4RunManagerType::Serial;
#    endif
        return std::unique_ptr<G4RunManager>(
            G4RunManagerFactory::CreateRunManager(default_rmt));
#elif defined(G4MULTITHREADED)
        return std::make_unique<G4MTRunManager>();
#else
        return std::make_unique<G4RunManager>();
#endif
    }();
    CELER_ASSERT(run_manager);

    // Set up loggers
    world_logger() = Logger::from_handle_env(make_world_handler(), "CELER_LOG");
    self_logger() = Logger::from_handle_env(
        make_self_handler(get_geant_num_threads(*run_manager)),
        "CELER_LOG_LOCAL");

    // Redirect Geant4 output and exceptions through Celeritas objects
    ScopedGeantLogger scoped_logger;
    ScopedGeantExceptionHandler scoped_exceptions;

    CELER_LOG(info) << "Run manager type: "
                    << TypeDemangler<G4RunManager>{}(*run_manager);

    std::vector<std::string> ignore_processes = {"CoulombScat"};
    setup.SetIgnoreProcesses(ignore_processes);

    // Construct geometry and SD factory
    run_manager->SetUserInitialization(new DetectorConstruction{params});

    // Construct physics
    if (setup.input().physics_list == PhysicsListSelection::ftfp_bert)
    {
        auto pl = std::make_unique<FTFP_BERT>(/* verbosity = */ 0);
        run_manager->SetUserInitialization(pl.release());
    }
    else
    {
        auto opts = setup.GetPhysicsOptions();
        if (setup.input().physics_list == PhysicsListSelection::celer_ftfp_bert)
        {
            // FTFP BERT with Celeritas EM standard physics
            auto pl = std::make_unique<celeritas::FtfpBertPhysicsList>(opts);
            run_manager->SetUserInitialization(pl.release());
        }
        else
        {
            // Celeritas EM standard physics only
            auto pl = std::make_unique<celeritas::EmPhysicsList>(opts);
            run_manager->SetUserInitialization(pl.release());
        }
    }

    // Create action initializer
    auto act_init = std::make_unique<ActionInitialization>(params);
    int num_events = act_init->num_events();
    run_manager->SetUserInitialization(act_init.release());

    // Initialize run and process events
    {
        ScopedMem record_mem("run.initialize");
        ScopedTimeLog scoped_time;
        ScopedProfiling profile_this{"celer-g4-setup"};
        CELER_LOG(status) << "Initializing run manager";
        run_manager->Initialize();
    }
    {
        ScopedMem record_mem("run.beamon");
        ScopedTimeLog scoped_time;
        ScopedProfiling profile_this{"celer-g4-run"};
        CELER_LOG(status) << "Transporting " << num_events << " events";
        run_manager->BeamOn(num_events);
    }

    CELER_LOG(debug) << "Destroying run manager";
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
    using celeritas::ScopedMpiInit;

    ScopedMpiInit scoped_mpi(&argc, &argv);

    if (scoped_mpi.is_world_multiprocess())
    {
        CELER_LOG(critical) << "This app cannot run with MPI parallelism.";
        return EXIT_FAILURE;
    }

    // Process input arguments
    if (argc != 2)
    {
        celeritas::app::print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    std::string_view filename{argv[1]};
    if (filename == "--help"sv || filename == "-h"sv)
    {
        celeritas::app::print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (filename == "--version"sv || filename == "-v"sv)
    {
        std::cout << celeritas::version_string << std::endl;
        return EXIT_SUCCESS;
    }
    if (filename == "--dump-default"sv)
    {
        std::cout << nlohmann::json(celeritas::app::RunInput()).dump(1)
                  << std::endl;
        return EXIT_SUCCESS;
    }
    if (celeritas::starts_with(filename, "--"))
    {
        CELER_LOG(critical) << "Unknown option \"" << filename << "\"";
        celeritas::app::print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Create params, which need to be shared with detectors as well as
    // initialization, and can be written for output (default to stdout)
    auto params = std::make_shared<celeritas::SharedParams>();

    try
    {
        celeritas::app::run(filename, params);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical) << "While running " << argv[1] << ": " << e.what();
        auto e_output = std::make_shared<celeritas::ExceptionOutput>(
            std::current_exception());
        if (*params)
        {
            try
            {
                params->output_reg()->insert(e_output);
                params->Finalize();
            }
            catch (std::exception const& e)
            {
                CELER_LOG(critical)
                    << "Another exception occurred while finalizing output: "
                    << e.what();
                // Write a null JSON oboject since we didn't output anything;
                std::cout << "null\n";
            }
        }
        else
        {
            celeritas::OutputRegistry reg;
            reg.insert(e_output);
            reg.output(&std::cout);
        }
        return EXIT_FAILURE;
    }

    CELER_LOG(status) << "Run completed successfully; exiting";
    return EXIT_SUCCESS;
}
