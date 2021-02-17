//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor.cc
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
#include "physics/base/ParticleParams.hh"

#include "KNDemoIO.hh"
#include "KNDemoRunner.hh"
#include "LoadXs.hh"

using namespace celeritas;
using namespace demo_interactor;
using std::cerr;
using std::cout;
using std::endl;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Construct particle parameters and send to GPU.
 */
std::shared_ptr<ParticleParams> load_params()
{
    using namespace celeritas::units;
    constexpr auto zero   = zero_quantity();
    constexpr auto stable = ParticleDef::stable_decay_constant();

    return std::make_shared<ParticleParams>(
        ParticleParams::Input{{"electron",
                               pdg::electron(),
                               MevMass{0.5109989461},
                               ElementaryCharge{-1},
                               stable},
                              {"gamma", pdg::gamma(), zero, zero, stable}});
}

//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream& is)
{
    // Read input options
    auto inp = nlohmann::json::parse(is);

    // Construct runner
    auto         grid_params = inp.at("grid_params").get<CudaGridParams>();
    KNDemoRunner run(load_params(), load_xs(), grid_params);

    // For now, only do a single run
    auto run_args = inp.at("run").get<KNDemoRunArgs>();
    CELER_EXPECT(run_args.energy > 0);
    CELER_EXPECT(run_args.num_tracks > 0);
    CELER_EXPECT(run_args.max_steps > 0);
    auto result = run(run_args);

    nlohmann::json outp = {
        {"grid_params", grid_params},
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
} // namespace demo_interactor

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && Communicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
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
    celeritas::activate_device(Device(0));

    if (!celeritas::device())
    {
        CELER_LOG(critical) << "CUDA capability is disabled";
        return EXIT_FAILURE;
    }

    if (args[1] != "-")
    {
        std::ifstream infile(args[1]);
        if (!infile)
        {
            CELER_LOG(critical) << "Failed to open '" << args[1] << "'";
            return EXIT_FAILURE;
        }
        run(infile);
    }
    else
    {
        // Read input from STDIN
        run(std::cin);
    }

    return EXIT_SUCCESS;
}
