//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange-update.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"

#if CELERITAS_USE_JSON
#    include <fstream>
#    include <nlohmann/json.hpp>

#    include "orange/construct/OrangeInputIO.json.hh"
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
    std::cerr << "usage: " << exec_name
              << " {input}.org.json {output}.org.json\n";
}

//---------------------------------------------------------------------------//
std::string run(std::istream* is)
{
#if CELERITAS_USE_JSON
    OrangeInput inp;
    nlohmann::json::parse(*is).get_to(inp);

    return nlohmann::json(inp).dump(/* indent = */ 0);
#else
    CELER_DISCARD(is);
    CELER_NOT_CONFIGURED("JSON");
#endif
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
    if (args.size() != 2)
    {
        // Incorrect number of arguments: print help and exit
        print_usage(argv[0]);
        return 2;
    }

    // Set up input/output files
    std::ifstream infile;
    std::istream* instream = nullptr;
    if (args[0] == "-")
    {
        instream = &std::cin;
    }
    else
    {
        // Open the specified file
        infile.open(args[0]);
        if (!infile)
        {
            CELER_LOG(critical) << "Failed to open '" << args[0] << "'";
            return EXIT_FAILURE;
        }
        instream = &infile;
    }

    std::string result;
    try
    {
        result = celeritas::app::run(instream);
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

    if (args[1] == "-")
    {
        std::cout << result;
    }
    else
    {
        // Open the specified file
        std::ofstream outfile{args[1]};
        if (!outfile)
        {
            CELER_LOG(critical)
                << "Failed to open '" << args[1] << "' for writing";
            return EXIT_FAILURE;
        }
        outfile << result;
    }

    return EXIT_SUCCESS;
}
