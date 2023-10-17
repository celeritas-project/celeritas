//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/celer-sim.cc
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

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "celeritas_config.h"
#include "celeritas_version.h"
#include "corecel/device_runtime_api.h"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/Types.hh"

#include "Runner.hh"
#include "RunnerInput.hh"
#include "RunnerOutput.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "RunnerInputIO.json.hh"
#endif

using namespace std::literals::string_view_literals;

namespace celeritas
{
namespace app
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
void run(std::istream* is, std::shared_ptr<OutputRegistry> output)
{
    CELER_EXPECT(is);

    ScopedProfiling profile_this{"celer-sim"};
    ScopedMem record_mem("celer-sim.run");

    // Read input options and save a copy for output
    auto run_input = std::make_shared<RunnerInput>();
#if CELERITAS_USE_JSON
    nlohmann::json::parse(*is).get_to(*run_input);
#else
    CELER_NOT_CONFIGURED("nlohmann_json");
#endif
    output->insert(std::make_shared<OutputInterfaceAdapter<RunnerInput>>(
        OutputInterface::Category::input, "*", run_input));

    // Create runner and save setup time
    Stopwatch get_setup_time;
    Runner run_stream(*run_input, output);
    SimulationResult result;
    result.setup_time = get_setup_time();
    result.events.resize(run_stream.num_events());

    // Allocate device streams, or use the default stream if there is only one.
    size_type num_streams = run_stream.num_streams();
    if (run_input->use_device && !run_input->default_stream && num_streams > 1)
    {
        CELER_ASSERT(device());
        device().create_streams(num_streams);
    }
    result.num_streams = num_streams;

    if (run_input->warm_up)
    {
        get_setup_time = {};
        run_stream.warm_up();
        result.warmup_time = get_setup_time();
    }

    // Start profiling *after* initialization and warmup are complete
    Stopwatch get_transport_time;
    if (run_input->merge_events)
    {
        // Run all events simultaneously on a single stream
        result.events.front() = run_stream();
    }
    else
    {
        MultiExceptionHandler capture_exception;
#ifdef _OPENMP
        // Set the maximum number of nested parallel regions
        // TODO: Enable nested OpenMP parallel regions for multithreaded CPU
        // once the performance issues have been resolved. For now, limit the
        // level of nesting to a single parallel region (over events) and
        // deactivate any deeper nested parallel regions.
        omp_set_max_active_levels(1);
#    pragma omp parallel for num_threads(num_streams)
#endif
        for (size_type event = 0; event < run_stream.num_events(); ++event)
        {
            activate_device_local();

            // Run a single event on a single thread
            CELER_TRY_HANDLE(result.events[event] = run_stream(
                                 StreamId(get_openmp_thread()), EventId(event)),
                             capture_exception);
        }
        log_and_rethrow(std::move(capture_exception));
    }
    result.action_times = run_stream.get_action_times();
    result.total_time = get_transport_time();
    record_mem = {};
    output->insert(std::make_shared<RunnerOutput>(std::move(result)));
}

//---------------------------------------------------------------------------//
void print_usage(char const* exec_name)
{
    // clang-format off
    std::cerr << "usage: " << exec_name << " {input}.json\n"
                 "       " << exec_name << " [--help|-h]\n"
                 "       " << exec_name << " --version\n"
                 "       " << exec_name << " --config\n"
                 "       " << exec_name << " --dump-default\n";
    // clang-format on
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
    using celeritas::MpiCommunicator;
    using celeritas::ScopedMpiInit;
    using celeritas::to_string;
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
    if (argc != 2)
    {
        celeritas::app::print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    std::string_view filename{argv[1]};
    if (filename == "--help"sv || filename == "-h"sv)
    {
        celeritas::app::print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (filename == "--version"sv || filename == "-v"sv)
    {
        std::cout << celeritas_version << std::endl;
        return EXIT_SUCCESS;
    }
    if (!CELERITAS_USE_JSON)
    {
        // Check for JSON *before* checking the options below
        std::cerr << argv[0]
                  << ": JSON is not enabled in this build of Celeritas"
                  << std::endl;
        return EXIT_FAILURE;
    }
    if (filename == "--config"sv)
    {
        std::cout << to_string(celeritas::BuildOutput{}) << std::endl;
        return EXIT_SUCCESS;
    }
    if (filename == "--dump-default"sv)
    {
#if CELERITAS_USE_JSON
        std::cout << nlohmann::json{celeritas::app::RunnerInput{}}.dump(1)
                  << std::endl;
#endif
        return EXIT_SUCCESS;
    }

    // Initialize GPU
    activate_device(comm);

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
        infile.open(std::string{filename});
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
        celeritas::app::run(instream, output);
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
