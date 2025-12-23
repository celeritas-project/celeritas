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
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "corecel/Config.hh"
#include "corecel/DeviceRuntimeApi.hh"
#include "corecel/Version.hh"

#include "corecel/Assert.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/FileOrConsole.hh"
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
#include "celeritas/global/CoreParams.hh"

#include "CliUtils.hh"
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
 * Run, launch, and get output.
 */
void run(std::shared_ptr<OutputRegistry>& output, std::string const& filename)
{
    ScopedMem record_mem("celer-sim.run");

    // Read input options
    auto run_input = [&filename] {
        celeritas::FileOrStdin instream{filename};
        auto result = std::make_shared<RunnerInput>();
        nlohmann::json::parse(instream).get_to(*result);
        return result;
    }();

    // Start profiling
    TracingSession tracing_session{run_input->tracing_file};
    ScopedProfiling profile_this{"celer-sim"};

    // Create runner and save setup time
    Stopwatch get_setup_time;
    Runner run_stream(*run_input);
    SimulationResult result;
    result.setup_time = get_setup_time();
    result.events.resize(
        run_input->transporter_result ? run_stream.num_events() : 1);

    // Add processed input to resulting output
    output = run_stream.core_params().output_reg();
    CELER_ASSERT(output);
    output->insert(std::make_shared<OutputInterfaceAdapter<RunnerInput>>(
        OutputInterface::Category::input, "*", run_input));

    // Allocate device streams
    size_type num_streams = run_stream.num_streams();
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
                          << " events on " << num_streams << " threads";
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
            tracing_session.flush();
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

std::string get_device_string()
{
    celeritas::activate_device();

    CELER_VALIDATE(celeritas::Device::num_devices() != 0,
                   << "no GPUs were detected");
    return nlohmann::json(celeritas::device()).dump(1);
}

std::string get_default_string()
{
    return nlohmann::json(celeritas::app::RunnerInput{}).dump(1);
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace app
}  // namespace celeritas

int main(int argc, char* argv[])
{
    using namespace celeritas::app;
    using std::cout;
    using std::endl;

    // Set up MPI
    celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
    {
        CELER_LOG(critical) << "TODO: this app cannot run in parallel";
        return EXIT_FAILURE;
    }

    // Set up app
    auto& cli = cli_app();
    cli.description("Run standalone Celeritas");

    // TODO for 1.0: instead of separate flags, make these subcommands
    std::string filename;
    cli.add_option("filename", filename, "Input JSON")
        ->check(CLI::ExistingFile | dash_validator());

    std::function<std::string()> diagnostic;
    auto set_diagnostic = [&diagnostic](auto func) {
        return [&diagnostic, func = std::move(func)](auto count) {
            CELER_DISCARD(count);
            if (diagnostic)
            {
                throw ConflictingArguments{
                    R"(only a single diagnostic is allowed)"};
            }
            diagnostic = std::move(func);
        };
    };
    cli.add_flag(
        "--config",
        set_diagnostic([] { return to_string(celeritas::BuildOutput{}); }),
        "Show configuration");
    cli.add_flag("--dump-default",
                 set_diagnostic(get_default_string),
                 "Dump default input");
    cli.add_flag("--device",
                 set_diagnostic(get_device_string),
                 "Show device information");

    // Require exactly one option
    cli.require_option(1);

    // Parse and run
    CELER_CLI11_PARSE(argc, argv);

    if (diagnostic)
    {
        // Print diagnostic and immediately exit
        auto print_diagnostic
            = [&diagnostic] { std::cout << diagnostic() << std::endl; };
        return run_safely(print_diagnostic);
    }

    // Run and save output
    std::shared_ptr<celeritas::OutputRegistry> output;
    int return_code = EXIT_SUCCESS;
    try
    {
        run(output, filename);
    }
    catch (std::exception const& e)
    {
        return_code = process_runtime_error(e);

        if (!output)
        {
            output = std::make_shared<celeritas::OutputRegistry>();
        }
        output->insert(std::make_shared<celeritas::ExceptionOutput>(
            std::current_exception()));
    }

    if (!output)
    {
        CELER_LOG(warning) << "No output available";
        std::cout << "null\n";
        return_code = EXIT_FAILURE;
    }
    else
    {
        CELER_LOG(status) << "Saving output";
        output->output(&std::cout);
        std::cout << std::endl;
    }

    // Delete streams before end of program (TODO: this is because of a static
    // initialization order issue; CUDA can be deactivated before the global
    // celeritas::device is reset)
    if (auto& d = celeritas::device())
    {
        d.destroy_streams();
    }

    return return_code;
}
