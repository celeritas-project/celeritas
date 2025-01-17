//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include <nlohmann/json.hpp>

#include "corecel/Config.hh"
#include "corecel/DeviceRuntimeApi.hh"
#include "corecel/Version.hh"

#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/TracingSession.hh"
#include "celeritas/Types.hh"

#include "Runner.hh"
#include "RunnerInput.hh"
#include "RunnerInputIO.json.hh"
#include "RunnerOutput.hh"

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
    ScopedMem record_mem("celer-sim.run");

    // Read input options and save a copy for output
    auto run_input = std::make_shared<RunnerInput>();
    nlohmann::json::parse(*is).get_to(*run_input);
    output->insert(std::make_shared<OutputInterfaceAdapter<RunnerInput>>(
        OutputInterface::Category::input, "*", run_input));

    // Start profiling
    TracingSession tracing_session{run_input->tracing_file};
    tracing_session.start();
    ScopedProfiling profile_this{"celer-sim"};

    // Create runner and save setup time
    Stopwatch get_setup_time;
    Runner run_stream(*run_input, output);
    SimulationResult result;
    result.setup_time = get_setup_time();
    if (run_input->transporter_result)
    {
        result.events.resize(run_stream.num_events());
    }

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
        auto event_result = run_stream();
        if (run_input->transporter_result)
        {
            result.events.front() = std::move(event_result);
        }
    }
    else
    {
        CELER_LOG(status) << "Transporting " << run_stream.num_events()
                          << " on " << num_streams << " threads";
        MultiExceptionHandler capture_exception;
        size_type const num_events = run_stream.num_events();
#if CELERITAS_OPENMP == CELERITAS_OPENMP_EVENT
#    pragma omp parallel for
#endif
        for (size_type event = 0; event < num_events; ++event)
        {
            activate_device_local();

            // Run a single event on a single thread
            TransporterResult event_result;
            CELER_TRY_HANDLE(event_result = run_stream(
                                 id_cast<StreamId>(get_openmp_thread()),
                                 id_cast<EventId>(event)),
                             capture_exception);
            if (run_input->transporter_result)
            {
                result.events[event] = std::move(event_result);
            }
        }
        log_and_rethrow(std::move(capture_exception));
    }

    result.action_times = run_stream.get_action_times();
    result.total_time = get_transport_time();
    record_mem = {};
    output->insert(std::make_shared<RunnerOutput>(std::move(result)));
}

//---------------------------------------------------------------------------//
void print_usage(std::string_view exec_name)
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
    using celeritas::ScopedMpiInit;
    using celeritas::to_string;
    using std::cout;
    using std::endl;

    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
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
    if (filename == "--config"sv)
    {
        std::cout << to_string(celeritas::BuildOutput{}) << std::endl;
        return EXIT_SUCCESS;
    }
    if (filename == "--dump-default"sv)
    {
        std::cout << nlohmann::json(celeritas::app::RunnerInput{}).dump(1)
                  << std::endl;
        return EXIT_SUCCESS;
    }

    // Initialize GPU
    celeritas::activate_device();

    if (filename == "--device"sv)
    {
        if (celeritas::Device::num_devices() == 0)
        {
            CELER_LOG(critical) << "No GPUs were detected";
            return EXIT_FAILURE;
        }
        std::cout << nlohmann::json(celeritas::device()).dump(1) << std::endl;
        return EXIT_SUCCESS;
    }

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
