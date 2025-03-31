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

    // Import physics data
    auto* world = GeantImporter::get_world_volume();
    CELER_ASSERT(world);
    ImportData imported = GeantImporter{world}(fi.geant.data_selection);

    // Set up problem
    inp::Problem problem;

    problem.model.geometry = world;
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
    result.world = world;

    // Set up core params
    result.problem = setup::problem(problem, imported);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
