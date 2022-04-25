//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop.cc
//---------------------------------------------------------------------------//
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "celeritas_version.h"
#include "base/Stopwatch.hh"
#include "comm/BuildOutput.hh"
#include "comm/Communicator.hh"
#include "comm/Device.hh"
#include "comm/DeviceIO.json.hh"
#include "comm/Environment.hh"
#include "comm/EnvironmentIO.json.hh"
#include "comm/KernelDiagnostics.hh"
#include "comm/KernelDiagnosticsIO.json.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"
#include "physics/base/PhysicsParamsOutput.hh"
#include "sim/OutputInterfaceAdapter.hh"
#include "sim/OutputManager.hh"

#include "LDemoIO.hh"
#include "Transporter.hh"
#include "Transporter.json.hh"

using std::cerr;
using std::cout;
using std::endl;
using namespace demo_loop;
using namespace celeritas;

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream* is, OutputManager* output)
{
    // Read input options
    auto inp = nlohmann::json::parse(*is);

    if (inp.contains("cuda_stack_size"))
    {
        celeritas::set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }
    if (inp.contains("environ"))
    {
        // Specify env variables
        inp["environ"].get_to(celeritas::environment());
    }

    // For now, only do a single run
    auto run_args = inp.get<LDemoArgs>();
    CELER_EXPECT(run_args);
    output->insert(std::make_shared<OutputInterfaceAdapter<LDemoArgs>>(
        OutputInterface::Category::input,
        "*",
        std::make_shared<LDemoArgs>(run_args)));

    // Start timer for overall execution
    Stopwatch get_setup_time;

    // Load all the problem data and create transporter
    auto         transport_ptr = build_transporter(run_args);
    const double setup_time    = get_setup_time();

    {
        // Save diagnostic information
        const auto& tinp = transport_ptr->input();
        output->insert(std::make_shared<PhysicsParamsOutput>(tinp.physics));
    }

    // Run all the primaries
    auto primaries = load_primaries(transport_ptr->input().particles, run_args);
    auto result    = (*transport_ptr)(*primaries);

    result.time.setup = setup_time;
    // TODO: convert individual results into OutputInterface so we don't have
    // to use this ugly "global" hack
    output->insert(OutputInterfaceAdapter<TransporterResult>::from_rvalue_ref(
        OutputInterface::Category::result, "*", std::move(result)));
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
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

    // Set up output
    OutputManager output;
    output.insert(OutputInterfaceAdapter<Device>::from_const_ref(
        OutputInterface::Category::system, "device", celeritas::device()));
    output.insert(OutputInterfaceAdapter<KernelDiagnostics>::from_const_ref(
        OutputInterface::Category::system,
        "kernels",
        celeritas::kernel_diagnostics()));
    output.insert(OutputInterfaceAdapter<Environment>::from_const_ref(
        OutputInterface::Category::system, "environ", celeritas::environment()));
    output.insert(std::make_shared<BuildOutput>());

    int return_code = EXIT_SUCCESS;
    try
    {
        run(instream, &output);
    }
    catch (const std::exception& e)
    {
        CELER_LOG(critical)
            << "While running input at  " << filename << ": " << e.what();
        return_code = EXIT_FAILURE;
    }

    // Write system properties even though results aren't available
    CELER_LOG(status) << "Saving output";
    output.output(&cout);
    cout << endl;

    return return_code;
}
