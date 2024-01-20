//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-export-geant.cc
//! Import Celeritas input data from Geant4 and serialize as ROOT.
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"

#if CELERITAS_USE_JSON
#    include <fstream>
#    include <nlohmann/json.hpp>

#    include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#endif

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
void print_usage(char const* exec_name)
{
    // clang-format off
    std::cerr
        << "usage: " << exec_name << " {input}.gdml "
                                     "[{options}.json, -, ''] {output}.root\n"
           "       " << exec_name << " --dump-default\n";
    // clang-format on
}

//---------------------------------------------------------------------------//

GeantPhysicsOptions load_options(std::string const& option_filename)
{
    GeantPhysicsOptions options;
    if (option_filename.empty())
    {
        CELER_LOG(info) << "Using default Celeritas Geant4 options";
        // ... but add verbosity
        options.verbose = true;
    }
    else if (!CELERITAS_USE_JSON)
    {
        CELER_LOG(critical) << "JSON is unavailable so only default Geant4 "
                               "options are supported: use '' as the second "
                               "argument";
        CELER_NOT_CONFIGURED("JSON");
    }
#if CELERITAS_USE_JSON
    else if (option_filename == "-")
    {
        auto inp = nlohmann::json::parse(std::cin);
        inp.get_to(options);
        CELER_LOG(info) << "Loaded Geant4 setup options: "
                        << nlohmann::json{options}.dump();
    }
    else
    {
        std::ifstream infile(option_filename);
        CELER_VALIDATE(infile, << "failed to open '" << option_filename << "'");
        auto inp = nlohmann::json::parse(infile);
        inp.get_to(options);
        CELER_LOG(info) << "Loaded Geant4 setup options from "
                        << option_filename << ": "
                        << nlohmann::json{options}.dump();
    }
#endif
    return options;
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
    using namespace celeritas;
    using namespace celeritas::app;

    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && MpiCommunicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() == 1 && (args.front() == "--help" || args.front() == "-h"))
    {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (args.size() == 1 && args.front() == "--dump-default")
    {
#if CELERITAS_USE_JSON
        GeantPhysicsOptions options;
        constexpr int indent = 1;
        std::cout << nlohmann::json{options}.dump(indent) << std::endl;
        return EXIT_SUCCESS;
#else
        CELER_LOG(error) << "JSON is unavailable: can't output geant options";
        return EXIT_FAILURE;
#endif
    }
    if (args.size() != 3)
    {
        // Incorrect number of arguments: print help and exit
        print_usage(argv[0]);
        return 2;
    }

    try
    {
        // Construct options, set up Geant4, read data
        auto imported = [&args] {
            GeantImporter import(GeantSetup(args[0], load_options(args[1])));
            GeantImporter::DataSelection selection;
            selection.particles = GeantImporter::DataSelection::em;
            selection.processes = GeantImporter::DataSelection::em;
            selection.reader_data = true;
            return import(selection);
        }();

        // Open ROOT file, write
        ScopedRootErrorHandler scoped_root_error;
        RootExporter export_root(args[2].c_str());
        export_root(imported);
        scoped_root_error.throw_if_errors();
    }
    catch (RuntimeError const& e)
    {
        CELER_LOG(critical) << "Runtime error: " << e.what();
        return EXIT_FAILURE;
    }
    catch (DebugError const& e)
    {
        CELER_LOG(critical) << "Assertion failure: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
