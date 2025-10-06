//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/StandaloneInput.cc
//---------------------------------------------------------------------------//
#include "StandaloneInput.hh"

#include <optional>
#include <string>
#include <variant>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/inp/Import.hh"
#include "celeritas/inp/Problem.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/io/ImportData.hh"

#include "Events.hh"
#include "Import.hh"
#include "Problem.hh"
#include "System.hh"

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
/*!
 * Completely set up a Celeritas problem from a standalone input.
 */
StandaloneLoaded standalone_input(inp::StandaloneInput& si)
{
    // Set up system
    setup::system(si.system);

    // Load problem
    auto& problem = si.problem;

    // Set up Geant4
    std::optional<GeantSetup> geant_setup;
    std::shared_ptr<GeantGeoParams> ggp;
    if (si.geant_setup)
    {
        // Take file name from problem and physics options from the arguments,
        // and set up Geant4
        CELER_ASSUME(
            std::holds_alternative<std::string>(problem.model.geometry));
        geant_setup.emplace(std::get<std::string>(problem.model.geometry),
                            *si.geant_setup);

        // Keep the geant4 geometry and set it as global
        ggp = geant_setup->geo_params();
        CELER_ASSERT(ggp);

        // Load geometry, surfaces, regions from Geant4 world pointer
        problem.model = ggp->make_model_input();
    }

    // Import physics data from Geant4 or ROOT: see Import.hh
    ImportData imported;
    std::visit(
        [&imported](auto const& physics_source_opts) {
            setup::physics_from(physics_source_opts, imported);
        },
        si.physics_import);

    // Load from external Geant4 data files
    setup::physics_from(inp::PhysicsFromGeantFiles{}, imported);

    // Copy optical physics from import data
    // (TODO: will be replaced)
    problem.physics.optical = imported.optical_physics;

    StandaloneLoaded result;

    // Set up core params
    result.problem = setup::problem(problem, imported);

    // Save geometry if loaded
    result.geant_geo = ggp;

    // Load events
    result.events = events(si.events, result.problem.core_params->particle());

    auto const& ctl = problem.control;
    if (ctl.capacity.events && ctl.num_streams > result.events.size())
    {
        CELER_LOG(warning)
            << "Configured number of streams (" << ctl.num_streams
            << ") exceeds number of loaded events (" << result.events.size()
            << ")";
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
