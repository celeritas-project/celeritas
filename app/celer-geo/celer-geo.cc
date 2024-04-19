//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-geo/celer-geo.cc
//---------------------------------------------------------------------------//
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "celeritas_config.h"
#include "celeritas_version.h"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/Stopwatch.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/rasterize/Image.hh"
#include "geocel/rasterize/ImageIO.json.hh"
#include "geocel/rasterize/RaytraceImager.hh"

// TODO: replace with factory
#include "celeritas/geo/GeoParams.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
using SPConstGeometry = std::shared_ptr<GeoParamsInterface const>;
using SPImager = std::shared_ptr<ImagerInterface>;

std::pair<SPConstGeometry, SPImager>
load_geometry_imager(std::string const& filename)
{
    auto geo = std::make_shared<GeoParams>(filename);
    auto imager = std::make_shared<RaytraceImager<GeoParams>>(geo);
    return {std::move(geo), std::move(imager)};
}

template<MemSpace M>
std::shared_ptr<Image<M>>
make_traced_image(std::shared_ptr<ImageParams const> img_params,
                  ImagerInterface& generate_image)
{
    auto image = std::make_shared<Image<M>>(std::move(img_params));
    CELER_LOG(status) << "Tracing image on " << to_cstring(M);
    generate_image(image.get());
    return image;
}

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
        set_cuda_heap_size(inp.at("cuda_heap_size").get<int>());
    }
    if (inp.contains("cuda_stack_size"))
    {
        set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }

    // Load geometry
    Stopwatch get_time;
    auto&& [geo_params, generate_image]
        = load_geometry_imager(inp.at("input").get<std::string>().c_str());
    timers["load"] = get_time();

    // Construct image
    auto img_params
        = std::make_shared<ImageParams>(inp.at("image").get<ImageInput>());

    get_time = {};
    std::shared_ptr<ImageInterface> image;
    if (device())
    {
        image
            = make_traced_image<MemSpace::device>(img_params, *generate_image);
    }
    else
    {
        image = make_traced_image<MemSpace::host>(img_params, *generate_image);
    }
    CELER_ASSERT(image);
    timers["trace"] = get_time();

    // Get geometry names
    std::vector<std::string> vol_names;
    for (auto vol_id : range(geo_params->num_volumes()))
    {
        vol_names.push_back(geo_params->id_to_label(VolumeId(vol_id)).name);
    }

    // Write image
    CELER_LOG(status) << "Transferring image and writing to disk";
    get_time = {};

    std::string out_filename = inp.at("output");
    std::vector<int> image_data(img_params->num_pixels());
    image->copy_to_host(make_span(image_data));
    std::ofstream(out_filename, std::ios::binary)
        .write(reinterpret_cast<char const*>(image_data.data()),
               image_data.size() * sizeof(int));
    timers["write"] = get_time();

    // Construct json output
    CELER_LOG(status) << "Exporting JSON metadata";
    nlohmann::json outp = {
        {"metadata", *img_params},
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
    outp["metadata"]["int_size"] = sizeof(int);
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
