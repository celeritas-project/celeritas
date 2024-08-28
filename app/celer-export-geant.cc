//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-export-geant.cc
//! Import Celeritas input data from Geant4 and serialize as ROOT.
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/ext/RootJsonDumper.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"

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
                                     "[{options}.json, -, ''] {output}.[root, json]\n"
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
    return options;
}

//---------------------------------------------------------------------------//
void run(std::string const& gdml_filename,
         std::string const& opts_filename,
         std::string const& out_filename)
{
    // Construct options, set up Geant4, read data
    auto imported = [&] {
        GeantImporter import(
            GeantSetup(gdml_filename, load_options(opts_filename)));
        GeantImporter::DataSelection selection;
        selection.particles = GeantImporter::DataSelection::em
                              | GeantImporter::DataSelection::optical;
        selection.processes = selection.particles;
        selection.reader_data = true;
        return import(selection);
    }();

    ScopedRootErrorHandler scoped_root_error;

    if (ends_with(out_filename, ".root"))
    {
        // Write ROOT file
        RootExporter export_root(out_filename.c_str());
        export_root(imported);
    }
    else if (ends_with(out_filename, ".json"))
    {
        // Write JSON to file
        CELER_LOG(info) << "Opening JSON output at " << out_filename;
        std::ofstream os(out_filename);
        RootJsonDumper dump_root(&os);
        dump_root(imported);
    }
    else if (out_filename == "-")
    {
        // Write JSON to stdout
        CELER_LOG(info) << "Writing JSON to stdout";
        RootJsonDumper dump_root(&std::cout);
        dump_root(imported);
    }
    else
    {
        CELER_VALIDATE(false,
                       << "invalid output filename '" << out_filename << "'");
    }

    scoped_root_error.throw_if_errors();
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
        GeantPhysicsOptions options;
        constexpr int indent = 1;
        std::cout << nlohmann::json{options}.dump(indent) << std::endl;
        return EXIT_SUCCESS;
    }
    if (args.size() != 3)
    {
        // Incorrect number of arguments: print help and exit
        print_usage(argv[0]);
        return 2;
    }

    try
    {
        run(args[0], args[1], args[2]);
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
