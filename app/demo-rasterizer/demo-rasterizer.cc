//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/demo-rasterizer.cc
//---------------------------------------------------------------------------//
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "celeritas_version.h"
#include "corecel/cont/Range.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/Stopwatch.hh"

#include "RDemoRunner.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream& is)
{
    // Read input options
    auto inp = nlohmann::json::parse(is);
    auto timers = nlohmann::json::object();

    if (inp.contains("cuda_heap_size"))
    {
        int heapSize = inp.at("cuda_heap_size").get<int>();
        set_cuda_heap_size(heapSize);
    }
    if (inp.contains("cuda_stack_size"))
    {
        set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }

    // Load geometry
    Stopwatch get_time;
    auto geo_params = std::make_shared<GeoParams>(
        inp.at("input").get<std::string>().c_str());
    timers["load"] = get_time();

    // Construct image
    ImageStore image(inp.at("image").get<ImageRunArgs>());

    // Construct runner
    RDemoRunner run(geo_params);
    get_time = {};
    run(&image);
    timers["trace"] = get_time();

    // Get geometry names
    std::vector<std::string> vol_names;
    for (auto vol_id : range(geo_params->num_volumes()))
    {
        vol_names.push_back(geo_params->id_to_label(VolumeId(vol_id)).name);
    }

    // Write image
    CELER_LOG(status) << "Transferring image from GPU to disk";
    get_time = {};
    std::string out_filename = inp.at("output");
    auto image_data = image.data_to_host();
    std::ofstream(out_filename, std::ios::binary)
        .write(reinterpret_cast<char const*>(image_data.data()),
               image_data.size() * sizeof(decltype(image_data)::value_type));
    timers["write"] = get_time();

    // Construct json output
    CELER_LOG(status) << "Exporting JSON metadata";
    nlohmann::json outp = {
        {"metadata", image},
        {"data", out_filename},
        {"volumes", vol_names},
        {"timers", timers},
        {
            "runtime",
            {
                {"version", std::string(celeritas_version)},
                {"device", device()},
                {"kernels", kernel_registry()},
            },
        },
    };
    std::cout << outp.dump() << std::endl;
    CELER_LOG(info) << "Exported image to " << out_filename;
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace app
}  // namespace celeritas

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 *
 * \todo This is copied from the other demo apps; move these to a shared driver
 * at some point.
 */
int main(int argc, char* argv[])
{
    using namespace celeritas;

    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && MpiCommunicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    // Process input arguments
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 2 || args[1] == "--help" || args[1] == "-h")
    {
        std::cerr << "usage: " << args[0] << " {input}.json" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream infile;
    std::istream* instream_ptr = nullptr;
    if (args[1] != "-")
    {
        infile.open(args[1]);
        if (!infile)
        {
            CELER_LOG(critical) << "Failed to open '" << args[1] << "'";
            return EXIT_FAILURE;
        }
        instream_ptr = &infile;
    }
    else
    {
        // Read input from STDIN
        instream_ptr = &std::cin;
    }

    // Initialize GPU
    activate_device();

    if (!device())
    {
        CELER_LOG(critical) << "CUDA capability is disabled";
        return EXIT_FAILURE;
    }

    try
    {
        CELER_ASSERT(instream_ptr);
        celeritas::app::run(*instream_ptr);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical) << "caught exception: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
