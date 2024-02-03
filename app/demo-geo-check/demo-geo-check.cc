//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geo-check/demo-geo-check.cc
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

#include "celeritas_cmake_strings.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.json.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Label.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Device.hh"
#include "geocel/Types.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "GCheckRunner.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Run, launch, and output.
 */
void run(std::istream& is)
{
    // Read input options
    auto inp = nlohmann::json::parse(is);

    // Load geometry
    auto geo_params = std::make_shared<GeoParams>(
        inp.at("input").get<std::string>().c_str());

    if (device() && inp.count("cuda_stack_size"))
    {
        set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }

    GeoTrackInitializer trkinit;
    inp.at("pos").get_to(trkinit.pos);
    inp.at("dir").get_to(trkinit.dir);
    trkinit.dir = make_unit_vector(trkinit.dir);

    CELER_ASSERT(geo_params);
    int max_steps = inp.at("max_steps").get<int>();

    // Construct runner, which takes over geo_params
    GCheckRunner run(geo_params, max_steps);
    run(&trkinit);
}

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
    CELER_LOG(info) << "Running on " << (device() ? "GPU" : "CPU") << " with "
                    << celeritas_core_geo << " geometry";

    try
    {
        CELER_ASSERT(instream_ptr);
        celeritas::app::run(*instream_ptr);
    }
    catch (std::exception const& e)
    {
        CELER_LOG(critical) << "Caught exception: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
