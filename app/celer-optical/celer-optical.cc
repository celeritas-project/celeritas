//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-optical/celer-optical.cc
//---------------------------------------------------------------------------//
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

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
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/TracingSession.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/StandaloneInputIO.json.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/Runner.hh"

#include "CliUtils.hh"
#include "SimulationResult.json.hh"

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
    ScopedMem record_mem("celer-optical.run");

    // Read input options
    auto input = [&input_filename] {
        celeritas::FileOrStdin instream{input_filename};
        celeritas::inp::OpticalStandaloneInput result;
        nlohmann::json::parse(instream).get_to(result);
        return result;
    }();

    // Standalone optical is run on a single GPU stream
    input.problem.num_streams = 1;

    /*! \todo Add readers/writers for distributions or initializers similar to
     * the EM "primary offload" to support other generator types.
     */
    CELER_VALIDATE(
        std::holds_alternative<celeritas::inp::OpticalPrimaryGenerator>(
            input.problem.generator),
        << "primary generator is the only optical photon generation mechanism "
           "currently supported");

    // Get the output filename
    output_filename = input.problem.output_file;

    // Start profiling
    TracingSession tracing_session{input.problem.perfetto_file};
    ScopedProfiling profile_this{"celer-optical"};

    SimulationResult result;

    // Set up optical problem
    Stopwatch get_setup_time;
    auto run = celeritas::optical::Runner(std::move(input));
    result.time.setup = get_setup_time();

    //! \todo Optical loop warmup

    // Get output registry
    output = run.params()->output_reg();

    // Transport all tracks to completion
    Stopwatch get_transport_time;
    auto run_result = run();
    result.time.total = get_transport_time();
    result.time.actions = std::move(run_result.action_times);
    result.counters = std::move(run_result.counters);

    record_mem = {};

    // Add simulation result to output
    output->insert(
        std::make_shared<celeritas::OutputInterfaceAdapter<SimulationResult>>(
            OutputInterface::Category::result,
            "*",
            std::make_shared<SimulationResult>(result)));
}

void print_config()
{
    std::cout << to_string(celeritas::BuildOutput{}) << std::endl;
}

void print_default()
{
    std::cout
        << nlohmann::json(celeritas::inp::OpticalStandaloneInput{}).dump(1)
        << std::endl;
}

void print_device()
{
    celeritas::activate_device();

    CELER_VALIDATE(celeritas::Device::num_devices() != 0,
                   << "No GPUs were detected");
    std::cout << nlohmann::json(celeritas::device()).dump(1) << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace app
}  // namespace celeritas

int main(int argc, char* argv[])
{
    using namespace celeritas::app;

    // Set up MPI
    celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
    {
        //! \todo Support parallel MPI execution
        CELER_LOG(critical) << "Parallel MPI execution is not yet supported";
        return EXIT_FAILURE;
    }

    // Set up app
    auto& cli = cli_app();
    cli.description("Run a standalone Celeritas optical simulation");

    // Add input file for standard run mode
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

    // Parse
    CELER_CLI11_PARSE(argc, argv);

    if (cli.get_subcommands().empty() && input_filename.empty())
    {
        CELER_LOG(critical)
            << "Either an input filename or a subcommand must be provided.\n\n"
            << cli.help();
        return EXIT_FAILURE;
    }

    // Set up the problem and run
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
