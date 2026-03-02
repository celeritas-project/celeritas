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
        CELER_ASSUME(
            std::holds_alternative<std::string>(problem.model.geometry));
        // Take file name from problem and physics options from the arguments,
        // and set up Geant4
        geant_setup.emplace(std::get<std::string>(problem.model.geometry),
                            *si.geant_setup);

        // Keep the geant4 geometry and set it as global
        ggp = geant_setup->geo_params();
        CELER_ASSERT(ggp);
    }
    else if (auto* s = std::get_if<std::string>(&problem.model.geometry))
    {
        // Load model directly (when loading ROOT physics)
        ggp = GeantGeoParams::from_gdml(*s);
    }
    else if (auto* pv
             = std::get_if<G4VPhysicalVolume const*>(&problem.model.geometry))
    {
        // Load model directly (unused?)
        ggp = std::make_shared<GeantGeoParams>(*pv, Ownership::reference);
    }
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }

    // Replace model input: load geometry, surfaces, regions from Geant4 world
    // pointer
    problem.model = ggp->make_model_input();

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
/*!
 * Completely set up a Celeritas optical-only problem from a standalone input.
 */
OpticalStandaloneLoaded standalone_input(inp::OpticalStandaloneInput& si)
{
    // Set up system
    setup::system(si.system);

    // Get optical physics options and deactivate everything else
    GeantPhysicsOptions gpo = GeantPhysicsOptions::deactivated();
    gpo.optical = si.geant_setup;
    if (gpo.optical->cherenkov || gpo.optical->scintillation)
    {
        // We currently load (almost) all physics data from Geant4, which means
        // setting up its physics consistently for the GeantImporter.
        // Scintillation and Cherenkov processes are for EM tracks to generate
        // optical photons, so we must have at least some EM physics present.
        // In the future, if we want to set up Celeritas from pure physics
        // input data (via inp), we don't need this restriction.
        gpo.ionization = true;
    }

    // Take geometry file name from problem and set up Geant4
    auto const& geometry = si.problem.model.geometry;
    CELER_ASSUME(std::holds_alternative<std::string>(geometry));
    GeantSetup geant_setup(
        std::get<std::string>(geometry), gpo, std::move(si.detectors));

    // Load geometry, surfaces, regions from Geant4 world pointer
    CELER_ASSERT(geant_setup.geo_params());
    si.problem.model = geant_setup.geo_params()->make_model_input();

    // Import optical physics data from Geant4
    ImportData imported;
    inp::PhysicsFromGeant pfg;
    pfg.data_selection.particles = GeantImportDataSelection::optical;
    pfg.data_selection.processes = GeantImportDataSelection::optical;
    setup::physics_from(pfg, imported);

    // Copy optical physics from import data
    si.problem.physics = imported.optical_physics;

    // Manually enable Cherenkov and scintillation if set in input since only
    // optical photon processes are imported from Geant4
    // (TODO: these should be set by GeantImporter based on available
    // processes.)
    if (std::holds_alternative<inp::OpticalOffloadGenerator>(
            si.problem.generator))
    {
        auto& ophys = si.problem.physics;
        ophys.cherenkov = bool(si.geant_setup.cherenkov);
        ophys.scintillation = bool(si.geant_setup.scintillation);
        if (!(ophys.cherenkov || ophys.scintillation))
        {
            CELER_LOG(error) << "Optical offload generator should not be used "
                                "without scintillation or Cherenkov physics";
        }
    }

    // Set up optical core params and save geometry
    OpticalStandaloneLoaded result;
    result.problem = setup::problem(si.problem, imported);
    result.geant_geo = geant_setup.geo_params();

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
