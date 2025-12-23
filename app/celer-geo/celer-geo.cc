//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/celer-geo.cc
//---------------------------------------------------------------------------//
#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"
#include "corecel/Version.hh"

#include "corecel/Assert.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/FileOrConsole.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterface.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "geocel/rasterize/Image.hh"
#include "geocel/rasterize/ImageIO.json.hh"
#include "orange/OrangeParamsOutput.hh"

#include "CliUtils.hh"
#include "GeoInput.hh"
#include "Runner.hh"

using namespace std::literals::string_view_literals;
using nlohmann::json;

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get a line of JSON input.
 */
nlohmann::json get_json_line(std::istream& is)
{
    static std::string jsonline;

    //! \todo Add separate thread for cin to check periodically for interrupts
    if (!std::getline(is, jsonline))
    {
        CELER_LOG(debug) << "Reached end of file";
        return nullptr;
    }
    if (trim(jsonline).empty())
    {
        CELER_LOG(debug) << "Got empty line";
        // No input provided
        return nullptr;
    }

    try
    {
        return json::parse(jsonline);
    }
    catch (json::parse_error const& e)
    {
        CELER_LOG(error) << "Failed to parse JSON input: " << e.what();
        CELER_LOG(info) << "Failed line: " << repr(jsonline);
        return nullptr;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write a line of JSON output.
 */
void put_json_line(nlohmann::json const& j)
{
    std::cout << j.dump() << std::endl;
}

//---------------------------------------------------------------------------//
/*!
 * Write a line of JSON output.
 */
void put_json_line(OutputInterface const& oi)
{
    return put_json_line(output_to_json(oi));
}

//---------------------------------------------------------------------------//
/*!
 * Create a Runner from user input.
 */
std::unique_ptr<Runner> make_runner(json const& input)
{
    ModelSetup model_setup;
    try
    {
        input.get_to(model_setup);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(error)
            << R"(Invalid model setup; expected structure written to stdout)";
        put_json_line(ModelSetup{});
        throw;
    }

    auto result = std::make_unique<Runner>(model_setup);

    // Echo setup with additions by copying base class attributes first
    ModelSetupOutput out;
    static_cast<ModelSetup&>(out) = model_setup;
    out.version_string = std::string{celeritas::version_string};
    out.version_hex = CELERITAS_VERSION;

    put_json_line(out);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Execute a single raytrace.
 */
void run_trace(Runner& runner,
               TraceSetup const& trace_setup,
               ImageInput const& image_setup)
{
    CELER_LOG(status) << "Tracing " << to_cstring(trace_setup.geometry)
                      << " image on " << to_cstring(trace_setup.memspace);

    // Run the raytrace
    auto image = [&] {
        if (image_setup)
        {
            // User specified a new image setup
            return runner.trace(trace_setup, image_setup);
        }
        else
        {
            // Reuse last image setup
            return runner.trace(trace_setup);
        }
    }();
    CELER_ASSERT(image);

    auto const& img_params = *image->params();

    // Write the output to disk
    CELER_LOG(info) << "Writing image to '" << trace_setup.bin_file << '\'';
    {
        std::ofstream image_file(trace_setup.bin_file, std::ios::binary);
        std::vector<int> image_data(img_params.num_pixels());
        image->copy_to_host(make_span(image_data));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        image_file.write(reinterpret_cast<char const*>(image_data.data()),
                         image_data.size() * sizeof(int));
    }

    json out{
        {"trace", trace_setup},
        {"image", img_params},
        {"sizeof_int", sizeof(int)},
    };
    if (trace_setup.volumes)
    {
        // Get geometry names
        out["volumes"] = runner.get_volumes(trace_setup.geometry);
    }

    put_json_line(out);
}

//---------------------------------------------------------------------------//
using CmdFuncPtr = void (*)(Runner*, nlohmann::json const&);

void cmd_config(Runner*, nlohmann::json const&)
{
    put_json_line(BuildOutput{});
}

void cmd_trace(Runner* runner, nlohmann::json const& input)
{
    CELER_EXPECT(runner);
    // Load required trace setup (geometry/memspace/output)
    TraceSetup trace_setup;
    ImageInput image_setup;
    try
    {
        input.get_to(trace_setup);
        if (auto iter = input.find("image"); iter != input.end())
        {
            iter->get_to(image_setup);
        }
    }
    catch (std::exception const& e)
    {
        CELER_LOG(error)
            << R"(Invalid trace setup; expected structure written to stdout ()"
            << e.what() << ")";
        json temp = TraceSetup{};
        temp["image"] = ImageInput{};
        put_json_line(temp);
        return;
    }

    run_trace(*runner, trace_setup, image_setup);
}

void cmd_orange_stats(Runner* runner, nlohmann::json const&)
{
    CELER_EXPECT(runner);
    put_json_line(
        OrangeParamsOutput{runner->load_geometry<Geometry::orange>()});
}

CmdFuncPtr
get_cmd_funcptr(nlohmann::json const& input, std::string_view cmd_str)
{
    auto iter = input.find("_cmd");
    if (iter == input.end())
    {
        CELER_LOG(warning) << "Missing '_cmd' key: assuming '" << cmd_str
                           << "' (DEPRECATED: will be removed in v1.0)";
    }
    else
    {
        CELER_VALIDATE(iter->is_string(), << "invalid type for _cmd");
        cmd_str = iter->get_ref<std::string const&>();
    }

    using MapCmdConverter = std::unordered_map<std::string_view, CmdFuncPtr>;
    static MapCmdConverter const cmd_to_func = {
        {"config", &cmd_config},
        {"trace", &cmd_trace},
        {"orange_stats", &cmd_orange_stats},
    };

    auto func_iter = cmd_to_func.find(cmd_str);
    CELER_VALIDATE(func_iter != cmd_to_func.end(),
                   << "invalid _cmd='" << cmd_str << "'");
    return func_iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 *
 * The input stream is expected to be in "JSON lines" format. The first input
 * \em must be a model setup; the following lines are individual commands to
 * trace an image. Newlines must be sent exactly \em once per input, and the
 * output \em must be flushed after doing so. (Recall that \em endl sends a
 * newline and flushes the output buffer.)
 */
void run(std::string const& filename)
{
    celeritas::FileOrStdin infile{filename};
    CELER_LOG(info) << "Reading JSON line input from " << infile.filename();

    ScopedSignalHandler interrupted{SIGINT};

    // Load the model
    CELER_LOG(diagnostic) << "Waiting for model setup";
    auto json_input = get_json_line(infile);
    if (json_input.is_null())
    {
        CELER_LOG(info)
            << R"(No input provided: printing build configuration and exiting)";
        cmd_config(nullptr, {});
        return;
    }

    CELER_VALIDATE(json_input.is_object(),
                   << "missing or invalid JSON-formatted run input");

    std::unique_ptr<Runner> runner;
    try
    {
        runner = make_runner(json_input);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical) << "Failed to load model";
        put_json_line(ExceptionOutput{std::current_exception()});
        throw;
    }

    while (true)
    {
        json_input = get_json_line(infile);
        if (interrupted())
        {
            CELER_LOG(diagnostic) << "Exiting raytrace loop: caught interrupt";
            interrupted = {};
            break;
        }
        if (json_input.is_null())
        {
            CELER_LOG(diagnostic) << "Exiting raytrace loop";
            break;
        }

        try
        {
            CELER_VALIDATE(json_input.is_object(),
                           << "invalid JSON input: must be object");
            auto cmd_funcptr = get_cmd_funcptr(json_input, "trace");
            (*cmd_funcptr)(runner.get(), json_input);
        }
        catch (std::exception const& e)
        {
            CELER_LOG(error) << "Command failed: " << e.what();
            put_json_line(ExceptionOutput{std::current_exception()});
        }
    }

    // Construct json output
    put_json_line(json::object({
        {"timers", runner->timers()},
        {
            "runtime",
            {
                {"device", device()},
                {"kernels", kernel_registry()},
                {"environment", environment()},
                {"build", output_to_json(BuildOutput{})},
            },
        },
    }));
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
    using namespace celeritas::app;

    celeritas::ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    auto& cli = cli_app();
    cli.description("Geometry visualization server");

    std::string filename;
    cli.add_option("filename", filename, "Input JSON lines")
        ->required()
        ->check(CLI::ExistingFile | dash_validator());

    // Parse and run
    CELER_CLI11_PARSE(argc, argv);
    return run_safely(run, filename);
}
