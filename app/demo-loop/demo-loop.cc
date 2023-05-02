//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/demo-loop.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedDeviceProfiling.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/Types.hh"

#include "Runner.hh"
#include "RunnerInput.hh"
#include "RunnerInputIO.json.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
//!@{
//! Save data to JSON
inline void to_json(nlohmann::json& j, RunTimingResult const& v)
{
    j = nlohmann::json{{"steps", v.steps},
                       {"total", v.total},
                       {"setup", v.setup},
                       {"actions", v.actions}};
}

inline void to_json(nlohmann::json& j, RunnerResult const& v)
{
    j = nlohmann::json{{"initializers", v.initializers},
                       {"active", v.active},
                       {"alive", v.alive},
                       {"steps", v.steps},
                       {"time", v.time}};
}
//!@}

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream* is, std::shared_ptr<celeritas::OutputRegistry> output)
{
    using celeritas::OutputInterface;
    using celeritas::OutputInterfaceAdapter;
    using celeritas::ScopedMem;

    ScopedMem record_mem("demo_loop.run");

    // Read input options and save a copy for output
    auto run_input = std::make_shared<RunnerInput>();
    nlohmann::json::parse(*is).get_to(*run_input);
    output->insert(std::make_shared<OutputInterfaceAdapter<RunnerInput>>(
        OutputInterface::Category::input, "*", run_input));

    // Create runner and save setup time
    celeritas::Stopwatch get_setup_time;
    Runner run_stream(*run_input, output);
    double const setup_time = get_setup_time();

    // Run on a single thread
    RunnerResult result = [&run_stream] {
        celeritas::ScopedDeviceProfiling profile_this;
        return run_stream(celeritas::StreamId{0});
    }();
    result.time.setup = setup_time;

    // TODO: convert individual results into OutputInterface so we don't have
    // to use this ugly "global" hack
    output->insert(OutputInterfaceAdapter<RunnerResult>::from_rvalue_ref(
        OutputInterface::Category::result, "*", std::move(result)));
}
//---------------------------------------------------------------------------//
}  // namespace
}  // namespace demo_loop

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 */
int main(int argc, char* argv[])
{
    using celeritas::MpiCommunicator;
    using celeritas::ScopedMpiInit;
    using std::cerr;
    using std::cout;
    using std::endl;

    ScopedMpiInit scoped_mpi(&argc, &argv);

    MpiCommunicator comm = [] {
        if (ScopedMpiInit::status() == ScopedMpiInit::Status::disabled)
            return MpiCommunicator{};

        return MpiCommunicator::comm_world();
    }();

    if (comm.size() > 1)
    {
        CELER_LOG(critical) << "TODO: this app cannot run in parallel";
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
    celeritas::activate_device(celeritas::make_device(comm));

    std::string filename = args[1];
    std::ifstream infile;
    std::istream* instream = nullptr;
    if (filename == "-")
    {
        instream = &std::cin;
        filename = "<stdin>";  // For nicer output on failure
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
    auto output = std::make_shared<celeritas::OutputRegistry>();

    int return_code = EXIT_SUCCESS;
    try
    {
        demo_loop::run(instream, output);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical)
            << "While running input at " << filename << ": " << e.what();
        return_code = EXIT_FAILURE;
        output->insert(std::make_shared<celeritas::ExceptionOutput>(
            std::current_exception()));
    }

    // Write system properties and (if available) results
    CELER_LOG(status) << "Saving output";
    output->output(&cout);
    cout << endl;

    return return_code;
}
