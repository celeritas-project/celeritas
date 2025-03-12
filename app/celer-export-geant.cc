//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-export-geant.cc
//! Import Celeritas input data from Geant4 and serialize as ROOT.
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/io/FileOrConsole.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/ext/RootJsonDumper.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportDataTrimmer.hh"

#include "CliUtils.hh"
#include "CLI/CLI.hpp"

namespace celeritas
{
namespace app
{
namespace
{

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
    else
    {
        FileOrStdin infile(option_filename);
        nlohmann::json::parse(infile).get_to(options);
        CELER_LOG(info) << "Loaded Geant4 setup options from "
                        << infile.filename() << ": "
                        << nlohmann::json{options}.dump();
    }
    return options;
}

//---------------------------------------------------------------------------//
void run(std::string const& gdml_filename,
         std::string const& opts_filename,
         std::string const& out_filename,
         bool gen_test)
{
    // TODO: expose data selection to JSON users?
    GeantImporter::DataSelection selection;
    selection.particles = GeantImporter::DataSelection::em
                          | GeantImporter::DataSelection::optical;
    selection.processes = selection.particles;
    selection.reader_data = !gen_test;

    // Construct options, set up Geant4, read data
    auto imported = [&] {
        GeantImporter import(
            GeantSetup(gdml_filename, load_options(opts_filename)));
        return import(selection);
    }();

    // TODO: expose trim data rather than bool 'gen_test'
    if (gen_test)
    {
        ImportDataTrimmer::Input options;
        options.mupp = true;
        options.max_size = 16;
        ImportDataTrimmer trim(options);
        trim(imported);
    }

    ScopedRootErrorHandler scoped_root_error;

    if (gen_test)
    {
        CELER_LOG(info) << "Trimming data for testing";
    }

    if (ends_with(out_filename, ".root"))
    {
        // Write ROOT file
        RootExporter export_root(out_filename.c_str());
        export_root(imported);
    }
    else
    {
        celeritas::FileOrStdout outstream{out_filename};

        // Write JSON to file
        CELER_LOG(info) << "Opening JSON output at " << outstream.filename();
        RootJsonDumper dump_json(outstream);
        dump_json(imported);
    }

    scoped_root_error.throw_if_errors();
}

void run_dump_default()
{
    GeantPhysicsOptions options;
    constexpr int indent = 1;
    std::cout << nlohmann::json{options}.dump(indent) << std::endl;
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
    using namespace celeritas::app;

    celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    auto& cli = cli_app();
    cli.description("Export Geant4 data to ROOT or JSON");

    bool dump_default = false;
    bool gen_test = false;
    std::string gdml_filename;
    std::string opts_filename;
    std::string out_filename;

    cli.add_flag("--dump-default", dump_default, "Dump default options");
    cli.add_flag("--gen-test", gen_test, "Generate test data");
    cli.add_option("gdml", gdml_filename, "Input GDML file")
        ->check(CLI::ExistingFile);
    cli.add_option("physopt", opts_filename, "Geant physics options JSON")
        ->check(CLI::ExistingFile | dash_validator()
                | empty_string_validator());
    cli.add_option("output",
                   out_filename,
                   R"(Output file (ROOT or JSON or '-' for stdout JSON)");

    CELER_CLI11_PARSE(argc, argv);

    if ((!gdml_filename.empty() + dump_default) != 1)
    {
        return process_parse_error(ConflictingArguments{
            R"(provide a GDML file, or the gen/dump options)"});
    }

    if (dump_default)
    {
        return run_safely(run_dump_default);
    }

    if (out_filename.empty())
    {
        return process_parse_error(CLI::RequiredError("output"));
    }

    return run_safely(
        run, gdml_filename, opts_filename, out_filename, gen_test);
}
