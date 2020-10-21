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
#include "comm/Communicator.hh"
#include "comm/ScopedMpiInit.hh"
#include "comm/Utils.hh"
#include "physics/base/ParticleParams.hh"
#include "LoadXs.hh"
#include "KNDemoIO.hh"
#include "HostKNDemoRunner.hh"

using namespace celeritas;
using namespace demo_interactor_cpu;
using std::cerr;
using std::cout;
using std::endl;

namespace demo_interactor_cpu
{
//---------------------------------------------------------------------------//
/*!
 * Construct particle parameters and send to GPU.
 */
std::shared_ptr<ParticleParams> load_params()
{
    using namespace celeritas::units;
    celeritas::ZeroQuantity zero;
    auto                    stable = ParticleDef::stable_decay_constant();

    ParticleParams::VecAnnotatedDefs defs
        = {{{"electron", pdg::electron()},
            {MevMass{0.5109989461}, ElementaryCharge{-1}, stable}},
           {{"gamma", pdg::gamma()}, {zero, zero, stable}}};
    return std::make_shared<ParticleParams>(std::move(defs));
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
    HostKNDemoRunner run(load_params(), demo_interactor::load_xs());

    // For now, only do a single run
    auto run_args = inp.at("run").get<demo_interactor::KNDemoRunArgs>();
    REQUIRE(run_args.energy > 0);
    REQUIRE(run_args.num_tracks > 0);
    REQUIRE(run_args.max_steps > 0);
    auto result = run(run_args);

    nlohmann::json outp = {
        {"run", run_args},
        {"result", result},
    };
    cout << outp.dump() << endl;
}
} // namespace demo_interactor_cpu

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    Communicator  comm = Communicator::comm_world();
    if (comm.size() != 1)
    {
        if (comm.rank() == 0)
        {
            cerr << "This app is currently serial-only. Run with 1 proc."
                 << endl;
        }
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
            cerr << "fatal: failed to open '" << args[1] << "'" << endl;
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
