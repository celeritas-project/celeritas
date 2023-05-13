//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geo-check.cc
//---------------------------------------------------------------------------//

#include <cstdlib>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Device.hh"
#include "orange/Types.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "GCheckRunner.hh"
#include "nlohmann/json.hpp"

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

    GeoTrackInitializer trkinit;
    inp.at("track_origin").get_to(trkinit.pos);
    inp.at("track_direction").get_to(trkinit.dir);
    normalize_direction(&trkinit.dir);

    CELER_ASSERT(geo_params);
    int max_steps = inp.at("max_steps").get<int>();

    // Get geometry names
    std::vector<std::string> vol_names;
    for (auto vol_id : celeritas::range(geo_params->num_volumes()))
    {
        vol_names.push_back(
            to_string(geo_params->id_to_label(celeritas::VolumeId(vol_id))));
    }

    // Construct runner, which takes over geo_params
    GCheckRunner run(geo_params, max_steps, use_cuda);
    run(&trkinit);
}

}  // namespace geo_check

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
        }
    }
    catch (std::exception const& e)
    {
        CELER_LOG(status) << "No GPU device available - disable CUDA.";
        use_cuda = false;
    }

    try
    {
        CELER_ASSERT(instream_ptr);
        geo_check::run(*instream_ptr, use_cuda);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical) << "Caught exception: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
