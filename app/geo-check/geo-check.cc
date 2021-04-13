//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geo-check.cc
//---------------------------------------------------------------------------//

#include "base/Units.hh"
#include "base/Range.hh"
#include "comm/Device.hh"
#include "comm/DeviceIO.json.hh"
// #include "comm/KernelDiagnostics.hh"
// #include "comm/KernelDiagnosticsIO.json.hh"
#include "comm/Logger.hh"

#include "GCheckRunner.hh"

using namespace celeritas;

namespace geo_check
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

    if (inp.count("cuda_stack_size"))
    {
        GeoParams::set_cuda_stack_size(inp.at("cuda_stack_size").get<int>());
    }

    int max_steps = inp.at("max_steps").get<int>();

    // define track(s)
    int ntracks = 1;

    // constexpr real_type cm = celeritas::units::centimeter;
    // constexpr real_type mm = 0.1 * cm;
    // GeoStateInitializer init{{3441.4*cm, 0, -10860*cm}, {0, 0, 1}};

    Real3 pos, dir;
    for (int i = 0; i < 3; ++i)
    {
        pos[i] = inp.at("track_origin")[i].get<real_type>();
        dir[i] = inp.at("track_direction")[i].get<real_type>();
#if CELERITAS_USE_ROOT and defined(VECGEOM_ROOT)
        pos[i] = 10.0 * pos[i];
#endif
    }
    GeoStateInitializer init{pos, dir};

    // Construct runner
    GCheckRunner run(geo_params, max_steps);
    CELER_ASSERT(geo_params);
    run(&init, ntracks);

    // Get geometry names
    std::vector<std::string> vol_names;
    for (auto vol_id : celeritas::range(geo_params->num_volumes()))
    {
        vol_names.push_back(
            geo_params->id_to_label(celeritas::VolumeId(vol_id)));
    }
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
    celeritas::activate_device(Device(0));
    if (!celeritas::device())
    {
        CELER_LOG(critical) << "CUDA capability is disabled";
        return EXIT_FAILURE;
    }

    try
    {
        CELER_ASSERT(instream_ptr);
        geo_check::run(*instream_ptr);
    }
    catch (const std::exception& e)
    {
        CELER_LOG(critical) << "caught exception: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
