//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange-update.cc
//! \brief Read in and write back an ORANGE JSON file
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <string>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "corecel/io/FileOrConsole.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "orange/OrangeInputIO.json.hh"  // IWYU pragma: keep

#include "CliUtils.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
void run(std::string const& input_file, std::string const& output_file)
{
    OrangeInput inp;
    {
        celeritas::FileOrStdin instream{input_file};
        nlohmann::json::parse(instream).get_to(inp);
    }

    celeritas::FileOrStdout outstream{output_file};
    outstream << nlohmann::json(inp).dump(/* indent = */ 0);
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
    cli.description("Read in and write back an ORANGE JSON file");

    std::string input_file;
    std::string output_file;
    cli.add_option("input", input_file, "Input ORANGE JSON file")
        ->required()
        ->check(CLI::ExistingFile | dash_validator());
    cli.add_option("output", output_file, "Output ORANGE JSON file")
        ->required()
        ->check(CLI::ExistingFile | dash_validator());

    CELER_CLI11_PARSE(argc, argv);
    return run_safely(run, input_file, output_file);
}
