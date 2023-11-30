//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/host-demo-interactor.cc
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "celeritas_version.h"
#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "DetectorData.hh"
#include "HostKNDemoRunner.hh"
#include "KNDemoIO.hh"
#include "LoadXs.hh"

using std::cerr;
using std::cout;
using std::endl;

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct particle parameters.
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
    HostKNDemoRunner run(load_params(), celeritas::app::load_xs());

    // For now, only do a single run
    auto run_args = inp.at("run").get<celeritas::app::KNDemoRunArgs>();
    CELER_EXPECT(run_args.energy > 0);
    CELER_EXPECT(run_args.num_tracks > 0);
    CELER_EXPECT(run_args.max_steps > 0);
    auto result = run(run_args);

    nlohmann::json outp = {
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
    cout << outp.dump() << endl;
}
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
        cerr << "usage: " << args[0] << " {input}.json" << endl;
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
