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

#include "celeritas_config.h"
#if CELERITAS_USE_OPENMP
#    include <omp.h>
#endif

#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ScopedDeviceProfiling.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/Types.hh"

#include "Runner.hh"
#include "RunnerInput.hh"
#include "RunnerInputIO.json.hh"
#include "RunnerOutput.hh"

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get the OpenMP thread number.
 */
int get_openmp_thread()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream* is, std::shared_ptr<celeritas::OutputRegistry> output)
{
    using celeritas::EventId;
    using celeritas::OutputInterface;
    using celeritas::OutputInterfaceAdapter;
    using celeritas::ScopedMem;
    using celeritas::size_type;
    using celeritas::StreamId;

    ScopedMem record_mem("demo_loop.run");

    // Read input options and save a copy for output
    auto run_input = std::make_shared<RunnerInput>();
    nlohmann::json::parse(*is).get_to(*run_input);
    output->insert(std::make_shared<OutputInterfaceAdapter<RunnerInput>>(
        OutputInterface::Category::input, "*", run_input));

    // Create runner and save setup time
    celeritas::Stopwatch get_setup_time;
    Runner run_stream(*run_input, output);
    SimulationResult result;
    result.setup_time = get_setup_time();
    result.events.resize(run_stream.num_events());

    celeritas::Stopwatch get_transport_time;
    if (run_input->merge_events)
    {
        // Run all events simultaneously on a single stream
        result.events.front() = run_stream();
    }
    else
    {
        celeritas::MultiExceptionHandler capture_exception;
        celeritas::ScopedDeviceProfiling profile_this;

#ifdef _OPENMP
        // Set the maximum number of nested parallel regions
        // TODO: Enable nested OpenMP parallel regions for multithreaded CPU
        // once the performance issues have been resolved. For now, limit the
        // level of nesting to a single parallel region (over events) and
        // deactivate any deeper nested parallel regions.
        omp_set_max_active_levels(1);
#pragma omp parallel for num_threads(run_stream.num_streams())
#endif
        for (size_type event = 0; event < run_stream.num_events(); ++event)
        {
            // Run a single event on a single thread
            CELER_TRY_HANDLE(result.events[event] = run_stream(
                                 StreamId(get_openmp_thread()), EventId(event)),
                             capture_exception);
        }
        celeritas::log_and_rethrow(std::move(capture_exception));
    }
    result.total_time = get_transport_time();
    output->insert(std::make_shared<RunnerOutput>(std::move(result)));
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
