//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/celer-geo.cc
//---------------------------------------------------------------------------//
#include <csignal>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Repr.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/ScopedSignalHandler.hh"
#include "geocel/rasterize/Image.hh"
#include "geocel/rasterize/ImageIO.json.hh"

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
 * Create a Runner from user input.
 */
Runner make_runner(json const& input)
{
    ModelSetup model_setup;
    try
    {
        input.get_to(model_setup);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(error) << "Invalid model setup; expected structure written "
                            "to stdout";
        std::cout << json(ModelSetup{}).dump() << std::endl;
        throw;
    }

    Runner result(model_setup);
    std::cout << json(model_setup) << std::endl;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Execute a single raytrace.
 */
void run_trace(Runner& run_trace,
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
            return run_trace(trace_setup, image_setup);
        }
        else
        {
            // Reuse last image setup
            return run_trace(trace_setup);
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
        out["volumes"] = run_trace.get_volumes(trace_setup.geometry);
    }

    std::cout << out.dump() << std::endl;
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
void run(std::istream& is)
{
    ScopedSignalHandler interrupted{SIGINT};

    // Load the model
    CELER_LOG(diagnostic) << "Waiting for model setup";
    auto json_input = get_json_line(is);
    CELER_VALIDATE(json_input.is_object(),
                   << "missing or invalid JSON-formatted run input");

    auto runner = make_runner(json_input);

    while (true)
    {
        json_input = get_json_line(is);
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

        // Load required trace setup (geometry/memspace/output)
        TraceSetup trace_setup;
        ImageInput image_setup;
        try
        {
            CELER_VALIDATE(!json_input.empty(),
                           << "no raytrace input was specified");
            json_input.get_to(trace_setup);
            if (auto iter = json_input.find("image"); iter != json_input.end())
            {
                iter->get_to(image_setup);
            }
        }
        catch (std::exception const& e)
        {
            CELER_LOG(error)
                << "Invalid trace setup; expected structure written "
                   "to stdout ("
                << e.what() << ")";
            json temp = TraceSetup{};
            temp["image"] = ImageInput{};
            std::cout << json(temp).dump() << std::endl;
            continue;
        }
        try
        {
            run_trace(runner, trace_setup, image_setup);
        }
        catch (std::exception const& e)
        {
            CELER_LOG(error) << "Failed raytrace: " << e.what();
            std::cout << ExceptionOutput{std::current_exception()} << std::endl;
        }
    }

    // Construct json output (TODO: add build metadata)
    std::cout << json{
        {"timers", runner.timers()},
        {
            "runtime",
            {
                {"device", device()},
                {"kernels", kernel_registry()},
                {"build", json_pimpl_output(BuildOutput{})},
            },
        },
    } << std::endl;
}

//---------------------------------------------------------------------------//
void print_usage(std::string_view exec_name)
{
    // clang-format off
    std::cerr << "usage: " << exec_name << " {input}.json\n"
                 "       " << exec_name << " -\n"
                 "       " << exec_name << " [--help|-h]\n";
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
    using namespace celeritas;

    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (scoped_mpi.is_world_multiprocess())
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
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

    /*!
     * \todo Make a helper class that implicitly casts to std::istream&, has
     * operator bool, and retains a file or creates with '-'. We reuse this
     * paradigm in multiple classes.
     */
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
    CELER_LOG(info) << "Reading JSON line input from " << filename;

    try
    {
        celeritas::app::run(*instream);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical)
            << "While running input at " << filename << ": " << e.what();
        std::cout << ExceptionOutput{std::current_exception()} << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
