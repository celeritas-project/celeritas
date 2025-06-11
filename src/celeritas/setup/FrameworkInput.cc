//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/FrameworkInput.cc
//---------------------------------------------------------------------------//
#include "FrameworkInput.hh"

#include "corecel/Version.hh"

#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "geocel/GeantGeoParams.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/inp/FrameworkInput.hh"
#include "celeritas/inp/Problem.hh"
#include "celeritas/phys/ProcessBuilder.hh"

#include "Problem.hh"
#include "System.hh"

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
/*!
 * Completely set up a Celeritas problem from a framework input.
 */
FrameworkLoaded framework_input(inp::FrameworkInput& fi)
{
    CELER_LOG(info) << "Activating Celeritas version " << version_string
                    << " on " << (Device::num_devices() > 0 ? "GPU" : "CPU");

    // Set up system
    setup::system(fi.system);

    // Load Geant4 geometry wrapper and save as global
    CELER_ASSERT(!celeritas::geant_geo());
    auto geo = GeantGeoParams::from_tracking_manager();
    CELER_ASSERT(geo);
    celeritas::geant_geo(*geo);

    // Import physics data
    ImportData imported = GeantImporter{}(fi.geant.data_selection);

    // Set up problem
    inp::Problem problem;

    // Load geometry, surfaces, regions from Geant4 world pointer
    problem.model = geo->make_model_input();

    // Load physics
    for (std::string const& process_name : fi.geant.ignore_processes)
    {
        ImportProcessClass ipc;
        try
        {
            ipc = geant_name_to_import_process_class(process_name);
        }
        catch (RuntimeError const&)
        {
            CELER_LOG(error) << "User-ignored process '" << process_name
                             << "' is unknown to Celeritas";
            continue;
        }
        CELER_ASSUME(problem.physics.em);
        problem.physics.em->user_processes.emplace(ipc,
                                                   WarnAndIgnoreProcess{ipc});
    }

    if (fi.update)
    {
        CELER_NOT_IMPLEMENTED("updating input problem from external file");
    }

    // Adjust problem
    if (fi.adjust)
    {
        fi.adjust(problem);
    }

    FrameworkLoaded result;
    result.geo = std::move(geo);

    // Set up core params
    result.problem = setup::problem(problem, imported);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
