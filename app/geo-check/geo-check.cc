//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geo-check.cc
//---------------------------------------------------------------------------//

#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "nlohmann/json.hpp"

#include "GCheckRunner.hh"

using namespace celeritas;

namespace geo_check
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream& is, bool use_cuda)
{
    // Read input options
    auto inp = nlohmann::json::parse(is);

    // Load geometry
    auto geo_params = std::make_shared<GeoParams>(
        inp.at("input").get<std::string>().c_str());

    if (use_cuda && inp.count("cuda_stack_size"))
    {
        set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }

    // define track(s)
    // int ntracks = 1;

    // constexpr real_type cm = celeritas::units::centimeter;
    // constexpr real_type mm = 0.1 * cm;

    Real3 pos, dir;
    for (int i = 0; i < 3; ++i)
    {
        pos[i] = inp.at("track_origin")[i].get<real_type>();
        dir[i] = inp.at("track_direction")[i].get<real_type>();
    }
    GeoTrackInitializer trkinit{pos, dir};

    CELER_ASSERT(geo_params);
    int max_steps = inp.at("max_steps").get<int>();

    // Get geometry names
    std::vector<std::string> vol_names;
    for (auto vol_id : celeritas::range(geo_params->num_volumes()))
    {
        vol_names.push_back(to_string(
            geo_params->id_to_label(celeritas::VolumeId(vol_id))));
    }

    // Construct runner, which takes over geo_params
    GCheckRunner run(geo_params, max_steps, use_cuda);
    run(&trkinit);
}

} // namespace geo_check

//---------------------------------------------------------------------------//
/*!
 * Execute and run.
 *
 * \todo This is copied from the other demo apps; move these to a shared driver
 * at some point.
 */
int main(int argc, char* argv[])
{
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
    bool use_cuda = true;
    try
    {
        celeritas::activate_device(Device(0));
        if (!celeritas::device())
        {
            CELER_LOG(status) << "CUDA capability is disabled.";
            //return EXIT_FAILURE;
        }
    }
    catch (const std::exception& e)
    {
        CELER_LOG(status) << "No GPU device available - disable CUDA.";
        use_cuda = false;
    }

    try
    {
        CELER_ASSERT(instream_ptr);
        geo_check::run(*instream_ptr, use_cuda);
    }
    catch (const std::exception& e)
    {
        CELER_LOG(critical) << "caught exception: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
