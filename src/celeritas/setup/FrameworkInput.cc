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

#include "Import.hh"
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

    CELER_VALIDATE(!(fi.adjust && fi.adjust_optical),
                   << "cannot setup both a problem and an optical-only "
                      "problem");

    // TODO: How to determine which problem to setup without requiring adjust?
    CELER_EXPECT(fi.adjust || fi.adjust_optical);

    FrameworkLoaded result;

    // Set up system
    setup::system(fi.system);

    // Load Geant4 geometry wrapper, which saves it as global
    CELER_ASSERT(celeritas::global_geant_geo().expired());
    result.geo = GeantGeoParams::from_tracking_manager();
    CELER_ASSERT(result.geo);

    // Load Geant4 data from user setup
    ImportData imported;
    setup::physics_from(fi.physics_import, imported);

    // Load physics from external Geant4 data files
    setup::physics_from(inp::PhysicsFromGeantFiles{}, imported);

    if (fi.adjust_optical)
    {
        // Set up optical-only problem
        inp::OpticalProblem problem;

        problem.physics = imported.optical_physics;
        problem.model = result.geo->make_model_input();

        // Adjust problem
        fi.adjust_optical(problem);

        // Save filename for diagnostic output
        result.output_file = problem.output_file;

        // Set up core params
        result.problem = setup::problem(problem, imported);
    }
    else
    {
        // Set up problem
        inp::Problem problem;

        // Copy optical physics from import data
        // (TODO: will be replaced)
        problem.physics.optical = imported.optical_physics;

        // Load geometry, surfaces, regions from Geant4 world pointer
        problem.model = result.geo->make_model_input();

        // Load physics
        for (std::string const& process_name :
             fi.physics_import.ignore_processes)
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
            problem.physics.em.user_processes.emplace(
                ipc, WarnAndIgnoreProcess{ipc});
        }

        // Adjust problem
        fi.adjust(problem);

        // Save filename for diagnostic output
        result.output_file = problem.diagnostics.output_file;

        // Set up core params
        result.problem = setup::problem(problem, imported);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
