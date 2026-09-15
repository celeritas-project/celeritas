//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/StandaloneInput.cc
//---------------------------------------------------------------------------//
#include "StandaloneInput.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <variant>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Openmp.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/inp/Import.hh"
#include "celeritas/inp/Problem.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/JsonEventReader.hh"
#include "celeritas/io/RootEventReader.hh"

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

    CELER_ASSUME(
        std::holds_alternative<std::string>(si.problem.model.geometry));
    std::shared_ptr<GeantGeoParams> ggp;

    // Import physics data from Geant4 or ROOT: see Import.hh
    ImportData imported;
    std::visit(Overload{
                   [&](inp::PhysicsFromFile const& pff) {
                       // Load model directly (when loading ROOT physics)
                       ggp = GeantGeoParams::from_gdml(
                           std::get<std::string>(si.problem.model.geometry));

                       setup::physics_from(pff, imported);
                   },
                   [&](inp::PhysicsFromGeant& pfg) {
                       // Take file name from problem and physics options from
                       // the arguments, and set up Geant4
                       GeantSetup geant_setup(
                           std::get<std::string>(si.problem.model.geometry),
                           si.geant_setup);

                       // Keep the geant4 geometry and set it as global
                       ggp = geant_setup.geo_params();
                       CELER_ASSERT(ggp);

                       // Adjust Geant4 data selection based on physics options
                       GeantImportDataSelection::Flags selection
                           = GeantImportDataSelection::em_basic;
                       if (si.geant_setup.muon || si.geant_setup.mucf_physics)
                       {
                           selection |= GeantImportDataSelection::em_ex;
                       }
                       if (si.geant_setup.optical)
                       {
                           selection |= GeantImportDataSelection::optical;
                       }
                       pfg.data_selection.particles = selection;
                       pfg.data_selection.processes = selection;
                       setup::physics_from(pfg, imported);
                   },
               },
               si.physics_import);

    // Replace model input: load geometry, surfaces, regions from Geant4 world
    // pointer
    si.problem.model = ggp->make_model_input();

    // Load from external Geant4 data files
    setup::physics_from(inp::PhysicsFromGeantFiles{}, imported);

    // Copy optical physics from import data
    // (TODO: will be replaced)
    si.problem.physics.optical = imported.optical_physics;

    auto& ctl = si.problem.control;

    // Load number of events, needed to construct core params before loading
    // events
    ctl.capacity.events = std::visit(
        Overload{
            [](inp::CorePrimaryGenerator const& pg) { return pg.num_events; },
            [](inp::SampleFileEvents const& sfe) { return sfe.num_events; },
            [](inp::ReadFileEvents const& rfe) {
                if (ends_with(rfe.event_file, ".jsonl"))
                {
                    return JsonEventReader{rfe.event_file, nullptr}.num_events();
                }
                else if (ends_with(rfe.event_file, ".root"))
                {
                    return RootEventReader{rfe.event_file, nullptr}.num_events();
                }
                return EventReader{rfe.event_file, nullptr}.num_events();
            },
        },
        si.events.generator);
    CELER_ASSERT(ctl.capacity.events > 0);

    // Set the number of streams
    ctl.num_streams
        = (CELERITAS_OPENMP == CELERITAS_OPENMP_EVENT && !si.events.merge)
              ? openmp_max_threads()
              : 1;
    ctl.num_streams = std::min(ctl.num_streams, *ctl.capacity.events);

    StandaloneLoaded result;

    // Set up core params
    result.problem = setup::problem(si.problem, imported);

    // Save geometry if loaded
    result.geant_geo = ggp;

    // Load events
    result.events = events(si.events, result.problem.core_params->particle());
    CELER_ENSURE(ctl.num_streams <= result.events.size());

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
    if (std::holds_alternative<inp::OpticalOffloadGenerator>(
            si.problem.generator))
    {
        // Also have to import Cherenkov/scintillation, which apply
        // to EM particles
        pfg.data_selection.particles |= GeantImportDataSelection::em_basic;
    }
    setup::physics_from(pfg, imported);

    // Copy optical physics from import data
    si.problem.physics = imported.optical_physics;

    if (std::holds_alternative<inp::OpticalOffloadGenerator>(
            si.problem.generator))
    {
        // Check that G4 cherenkov/scintillation were created
        auto& gen = si.problem.physics.gen;
        if (!(gen.cherenkov || gen.scintillation))
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
