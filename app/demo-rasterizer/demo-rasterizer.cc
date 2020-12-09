//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer.cc
//---------------------------------------------------------------------------//
#include <cstddef>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "base/ColorUtils.hh"
#include "base/Range.hh"
#include "comm/Communicator.hh"
#include "comm/Device.hh"
#include "comm/Logger.hh"
#include "comm/ScopedMpiInit.hh"
#include "RDemoRunner.hh"

using namespace celeritas;
using std::cerr;
using std::cout;
using std::endl;

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream& is)
{
    // Read input options
    auto inp = nlohmann::json::parse(is);

    // Initialize GPU
    celeritas::initialize_device(Communicator{});

    // Load geometry
    auto geo_params = std::make_shared<GeoParams>(
        inp.at("input").get<std::string>().c_str());

    if (inp.count("cuda_stack_size"))
    {
        GeoParams::set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }

    // Construct image
    ImageStore image(inp.at("image").get<ImageRunArgs>());

    // Construct runner
    RDemoRunner run(geo_params);
    run(&image);

    // Get geometry names
    std::vector<std::string> vol_names;
    for (auto vol_id : celeritas::range(geo_params->num_volumes()))
    {
        vol_names.push_back(
            geo_params->id_to_label(celeritas::VolumeId(vol_id)));
    }

    // Write image
    CELER_LOG(status) << "Transferring image from GPU to disk";
    std::string out_filename = inp.at("output");
    auto        image_data   = image.data_to_host();
    std::ofstream(out_filename, std::ios::binary)
        .write(reinterpret_cast<const char*>(image_data.data()),
               image_data.size() * sizeof(decltype(image_data)::value_type));

    // Construct json output
    CELER_LOG(status) << "Exporting JSON metadata";
    nlohmann::json outp = {
        {"metadata", image}, {"data", out_filename}, {"volumes", vol_names}};
    cout << outp.dump() << endl;
    CELER_LOG(info) << "Exported image to " << out_filename;
}

} // namespace demo_rasterizer

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 *
 * \todo This is copied from the other demo apps; move these to a shared driver
 * at some point.
 */
int main(int argc, char* argv[])
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::initialized
        && Communicator::comm_world().size() > 1)
    {
        CELER_LOG(critical) << "This app cannot run in parallel";
        return EXIT_FAILURE;
    }

    // Process input arguments
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() != 2 || args[1] == "--help" || args[1] == "-h")
    {
        cerr << "usage: " << args[0] << " {input}.json" << endl;
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

    if (!celeritas::is_device_enabled())
    {
        CELER_LOG(critical) << "CUDA capability is disabled";
        return EXIT_FAILURE;
    }

    try
    {
        CHECK(instream_ptr);
        demo_rasterizer::run(*instream_ptr);
    }
    catch (const std::exception& e)
    {
        CELER_LOG(critical) << "caught exception: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
