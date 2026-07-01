//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-sim/celer-sim.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "corecel/Assert.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/FileOrConsole.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"  // IWYU pragma: keep
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/TracingSession.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/inp/StandaloneInputIO.json.hh"

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
 * Run, launch, and get output.
 */
void run(std::shared_ptr<OutputRegistry>& output,
         std::string& output_filename,
         std::string const& input_filename)
{
    // Read input options
    inp::StandaloneInput si = [&input_filename] {
        celeritas::FileOrStdin instream{input_filename};
        auto j = nlohmann::json::parse(instream);
        if (j.contains("problem"))
        {
            inp::StandaloneInput result;
            j.get_to(result);
            return result;
        }
        CELER_LOG(warning) << "Deprecated celer-sim input format. Update the "
                              "input using the 'update' subcommand.";
        RunnerInput old_inp;
        j.get_to(old_inp);
        return to_input(old_inp);
    }();

    // Get the output filename
    output_filename = si.problem.diagnostics.output_file;

    // Start profiling
    TracingSession tracing_session{si.problem.diagnostics.perfetto_file};
    std::optional<ScopedProfiling> profile_this{std::in_place, "setup"};

    // Create runner and save setup time
    Stopwatch get_setup_time;
    Runner run_stream(si);
    SimulationResult result;
    result.setup_time = get_setup_time();
    result.events.resize(
        si.problem.diagnostics.counters.event ? run_stream.num_events() : 1);

    // Add processed input to resulting output
    output = run_stream.core_params().output_reg();
    CELER_ASSERT(output);
    output->insert(
        std::make_shared<OutputInterfaceAdapter<inp::StandaloneInput>>(
            OutputInterface::Category::input,
            "*",
            std::make_shared<inp::StandaloneInput>(si)));

    // Allocate device streams
    size_type num_streams = run_stream.num_streams();
    result.num_streams = num_streams;

    if (si.problem.control.warm_up)
    {
        get_setup_time = {};
        run_stream.warm_up();
        result.warmup_time = get_setup_time();
    }

    // Start profiling *after* initialization and warmup are complete
    profile_this.emplace("run");
    Stopwatch get_transport_time;
    if (si.events.merge)
    {
        // Run all events simultaneously on a single stream
        auto event_result = run_stream();
        if (si.problem.diagnostics.counters.event)
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
#if CELERITAS_OPENMP == CELERITAS_OPENMP_EVENT
            auto stream = id_cast<StreamId>(omp_get_thread_num());
#else
            constexpr StreamId stream{0};
#endif

            // Run a single event on a single thread
            TransporterResult event_result;
            CELER_TRY_HANDLE(
                event_result = run_stream(stream, id_cast<EventId>(event)),
                capture_exception);
            tracing_session.flush();
            if (si.problem.diagnostics.counters.event)
            {
                result.events[event] = std::move(event_result);
            }
        }
        log_and_rethrow(std::move(capture_exception));
    }
    profile_this.reset();

    result.action_times = run_stream.get_action_times();
    result.step_times = run_stream.get_step_times();
    result.total_time = get_transport_time();
    output->insert(std::make_shared<RunnerOutput>(std::move(result)));
}

void print_config()
{
    std::cout << to_string(celeritas::BuildOutput{}) << std::endl;
}

void print_default()
{
    std::cout << nlohmann::json(celeritas::inp::StandaloneInput{}).dump(1)
              << std::endl;
}

void print_device()
{
    celeritas::activate_device();

    CELER_VALIDATE(celeritas::Device::num_devices() != 0,
                   << "no GPUs were detected");
    std::cout << nlohmann::json(celeritas::device()).dump(1) << std::endl;
}

void update_input(std::string const& filename)
{
    celeritas::FileOrStdin instream{filename};
    auto j = nlohmann::json::parse(instream);
    if (j.contains("problem"))
    {
        std::cout << j.dump(1) << std::endl;
        return;
    }
    RunnerInput old_inp;
    j.get_to(old_inp);
    std::cout << nlohmann::json(to_input(old_inp)).dump(1) << std::endl;
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
        CELER_LOG(critical) << "Parallel MPI execution is not yet supported";
        return EXIT_FAILURE;
    }

    // Set up app
    auto& cli = cli_app();
    cli.description("Run standalone Celeritas");

    // Set up app
    std::string input_filename;
    cli.add_option("filename", input_filename, "Input JSON")
        ->check(CLI::ExistingFile | dash_validator());

    // Add "config" subcommand to print configuration
    auto* config_cmd = cli.add_subcommand("config", "Show configuration");
    config_cmd->callback([] { std::exit(run_safely(print_config)); });

    // Add "dump-default" subcommand to print default input
    auto* default_cmd = cli.add_subcommand("default", "Show default input");
    default_cmd->callback([] { std::exit(run_safely(print_default)); });

    // Add "device" subcommand to print device information
    auto* device_cmd = cli.add_subcommand("device", "Show device information");
    device_cmd->callback([] { std::exit(run_safely(print_device)); });

    // Add "update" subcommand to convert a deprecated input JSON file
    std::string old_filename;
    auto* update_cmd
        = cli.add_subcommand("update", "Convert a deprecated input JSON file");
    update_cmd->add_option("filename", old_filename, "Deprecated input JSON")
        ->required()
        ->check(CLI::ExistingFile | dash_validator());
    update_cmd->callback([&old_filename] {
        std::exit(run_safely([&old_filename] { update_input(old_filename); }));
    });

    // TODO DEPRECATED: remove flags in favor of subcommands in v1.0
    std::function<void()> diagnostic;
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
    cli.add_flag("--config",
                 set_diagnostic(print_config),
                 "DEPRECATED: use 'config' subcommand instead");
    cli.add_flag("--dump-default",
                 set_diagnostic(print_default),
                 "DEPRECATED: use 'default' subcommand instead");
    cli.add_flag("--device",
                 set_diagnostic(print_device),
                 "DEPRECATED: use 'device' subcommand instead");

    // Parse and run
    CELER_CLI11_PARSE(argc, argv);

    if (!diagnostic && cli.get_subcommands().empty() && input_filename.empty())
    {
        CELER_LOG(critical)
            << "Either an input filename or a subcommand must be provided.\n\n"
            << cli.help();
        return EXIT_FAILURE;
    }

    if (diagnostic)
    {
        // Print diagnostic and immediately exit
        return run_safely(diagnostic);
    }

    // Run and save output
    std::shared_ptr<celeritas::OutputRegistry> output;
    std::string output_filename = "-";
    int return_code = EXIT_SUCCESS;
    try
    {
        run(output, output_filename, input_filename);
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

    // Save output
    celeritas::FileOrStdout ostream{output_filename};
    CELER_LOG(status) << "Saving output to " << ostream.filename();
    if (!output)
    {
        CELER_LOG(warning) << "No output available";
        ostream << "null\n";
        return_code = EXIT_FAILURE;
    }
    else
    {
        output->output(&static_cast<std::ostream&>(ostream));
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
