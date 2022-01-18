//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop.cc
//---------------------------------------------------------------------------//
#include <cstddef>
#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "base/Stopwatch.hh"
#include "celeritas_version.h"
#include "comm/Communicator.hh"
#include "comm/Device.hh"
#include "comm/DeviceIO.json.hh"
#include "comm/Environment.hh"
#include "comm/EnvironmentIO.json.hh"
#include "comm/KernelDiagnostics.hh"
#include "comm/KernelDiagnosticsIO.json.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"

#include "LDemoIO.hh"
#include "Transporter.hh"
#include "Transporter.json.hh"

using std::cerr;
using std::cout;
using std::endl;
using namespace demo_loop;
using celeritas::TransporterBase;

namespace
{
//---------------------------------------------------------------------------//
nlohmann::json get_runtime_json()
{
    return {
        {"version", std::string(celeritas_version)},
        {"device", celeritas::device()},
        {"kernels", celeritas::kernel_diagnostics()},
        {"environ", celeritas::environment()},
    };
}

//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream& is)
{
    using celeritas::Stopwatch;
    using celeritas::TrackInitParams;
    using celeritas::TransporterResult;

    // Read input options
    auto inp = nlohmann::json::parse(is);

    if (inp.count("cuda_stack_size"))
    {
        celeritas::set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }
    if (inp.count("environ"))
    {
        // Specify env variables
        inp["environ"].get_to(celeritas::environment());
    }

    // For now, only do a single run
    auto run_args = inp.at("run").get<LDemoArgs>();
    CELER_EXPECT(run_args);

    // Start timer for overall execution
    Stopwatch get_total_time;

    // Load all the problem data and create transporter
    auto transport_ptr = build_transporter(run_args);

    // Run all the primaries
    auto primaries = load_primaries(transport_ptr->input().particles, run_args);
    auto result    = (*transport_ptr)(*primaries);
    result.time.total = get_total_time();

    // Save output
    nlohmann::json outp = {
        {"run", run_args},
        {"result", result},
        {"runtime", get_runtime_json()},
    };
    cout << outp.dump() << endl;
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    using celeritas::Communicator;
    using celeritas::ScopedMpiInit;

    ScopedMpiInit scoped_mpi(&argc, &argv);

    Communicator comm
        = (ScopedMpiInit::status() == ScopedMpiInit::Status::disabled
               ? Communicator{}
               : Communicator::comm_world());

    if (comm.size() > 1)
    {
        CELER_LOG(critical) << "TODO: this app cannot run in parallel";
        return EXIT_FAILURE;
    }

    // Process input arguments
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 2 || args[1] == "--help" || args[1] == "-h")
    {
        cerr << "usage: " << args[0] << " {input}.json" << endl;
        return EXIT_FAILURE;
    }

    // Initialize GPU
    celeritas::activate_device(celeritas::Device::from_round_robin(comm));

    std::string   filename = args[1];
    std::ifstream infile;
    std::istream* instream = nullptr;
    if (filename == "-")
    {
        instream = &std::cin;
        filename = "<stdin>"; // For nicer output on failure
    }
    else
    {
        // Open the specified file
        infile.open(filename);
        if (!infile)
        {
            CELER_LOG(critical) << "Failed to open '" << filename << "'";
            return EXIT_FAILURE;
        }
        instream = &infile;
    }

    try
    {
        run(*instream);
    }
    catch (const std::exception& e)
    {
        CELER_LOG(critical)
            << "While running input at  " << filename << ": " << e.what();

        // Write runtime even though results aren't available
        cout << nlohmann::json{{"runtime", get_runtime_json()}}.dump() << endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
