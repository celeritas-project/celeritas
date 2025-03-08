//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange-update.cc
//! \brief Read in and write back an ORANGE JSON file
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "orange/OrangeInputIO.json.hh"

#include "detail/CliCommon.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
void run(std::istream* is, std::string output_file)
{
    OrangeInput inp;
    nlohmann::json::parse(*is).get_to(inp);

    auto result = nlohmann::json(inp).dump(/* indent = */ 0);

    if (output_file == "-")
    {
        std::cout << result;
    }
    else
    {
        // Open the specified file
        std::ofstream outfile{output_file};
        CELER_VALIDATE(
            outfile, << "failed to open '" << output_file << "' for writing");
        outfile << result;
    }
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

    CLI::App cli{"Read in and write back an ORANGE JSON file"};
    detail::setup_app(cli);

    std::string input_file;
    std::string output_file;
    cli.add_option("input", input_file, "Input ORANGE JSON file")
        ->required()
        ->check(CLI::ExistingFile | detail::dash_validator());
    cli.add_option("output", output_file, "Output ORANGE JSON file")
        ->required()
        ->check(CLI::ExistingFile | detail::dash_validator());

    CLI11_PARSE(cli, argc, argv);

    // Set up input/output files
    std::ifstream infile;
    std::istream* instream = nullptr;
    if (input_file == "-")
    {
        instream = &std::cin;
    }
    else
    {
        // Open the specified file
        infile.open(input_file);
        if (!infile)
        {
            CELER_LOG(critical) << "Failed to open '" << input_file << "'";
            return EXIT_FAILURE;
        }
        instream = &infile;
    }

    return detail::run_safely(cli, run, instream, output_file);
}
