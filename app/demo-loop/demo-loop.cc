//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop.cc
//---------------------------------------------------------------------------//
#include <cstddef>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "celeritas_version.h"
#include "comm/Communicator.hh"
#include "comm/Device.hh"
#include "comm/DeviceIO.json.hh"
#include "comm/KernelDiagnostics.hh"
#include "comm/KernelDiagnosticsIO.json.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"

#include "LDemoIO.hh"
#include "LDemoRun.hh"

using std::cerr;
using std::cout;
using std::endl;
using namespace demo_loop;

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream& is)
{
    // Read input options
    auto inp = nlohmann::json::parse(is);

    // For now, only do a single run
    auto run_args = inp.at("run").get<LDemoArgs>();
    CELER_EXPECT(run_args);

    auto result = run_gpu(run_args);

    nlohmann::json outp = {
        {"run", run_args},
        {"result", result},
        {
            "runtime",
            {
                {"version", std::string(celeritas_version)},
                {"device", celeritas::device()},
                {"kernels", celeritas::kernel_diagnostics()},
            },
        },
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

    if (!celeritas::device())
    {
        CELER_LOG(critical) << "CUDA capability is disabled";
        return EXIT_FAILURE;
    }

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
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
