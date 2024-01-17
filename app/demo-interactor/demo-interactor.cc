//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/demo-interactor.cc
//---------------------------------------------------------------------------//

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "celeritas_version.h"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "KNDemoIO.hh"
#include "KNDemoRunner.hh"
#include "LoadXs.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Construct particle parameters and send to GPU.
 */
std::shared_ptr<ParticleParams> load_params()
{
    using namespace celeritas::units;
    using namespace constants;
    constexpr auto zero = zero_quantity();

    return std::make_shared<ParticleParams>(ParticleParams::Input{
        {"electron",
         pdg::electron(),
         MevMass{0.5109989461},
         ElementaryCharge{-1},
         stable_decay_constant},
        {"gamma", pdg::gamma(), zero, zero, stable_decay_constant}});
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
    auto grid_params = inp.at("grid_params").get<DeviceGridParams>();
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
                {"device", device()},
                {"kernels", kernel_registry()},
            },
        },
    };
    std::cout << outp.dump() << std::endl;
}
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

    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && MpiCommunicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    // Process input arguments
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 2 || args[1] == "--help" || args[1] == "-h")
    {
        std::cerr << "usage: " << args[0] << " {input}.json" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize GPU
    activate_device();

    if (!device())
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
        celeritas::app::run(infile);
    }
    else
    {
        // Read input from STDIN
        celeritas::app::run(std::cin);
    }

    return EXIT_SUCCESS;
}
