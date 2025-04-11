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
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/inp/Import.hh"
#include "celeritas/inp/Model.hh"
#include "celeritas/inp/Problem.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/io/ImportData.hh"

#include "Events.hh"
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
    auto* problem = std::get_if<inp::Problem>(&si.problem);
    if (!problem)
    {
        // TODO: load from serialized JSON/ROOT file
        CELER_NOT_IMPLEMENTED("importing input data from an external file");
    }

    // Set up Geant4
    std::optional<GeantSetup> geant_setup;
    if (si.geant_setup)
    {
        // Take file name from problem and physics options from the arguments,
        // and set up Geant4
        CELER_ASSUME(
            std::holds_alternative<std::string>(problem->model.geometry));
        geant_setup.emplace(std::get<std::string>(problem->model.geometry),
                            *si.geant_setup);

        // Replace world geometry with Geant4 world pointer
        problem->model.geometry = geant_setup->world();
    }

    // Import physics data
    ImportData imported = std::visit(
        Overload{
            [](inp::FileImport const& fi) {
                CELER_VALIDATE(!fi.input.empty(),
                               << "no file import specified");
                // Import physics data from ROOT file
                return RootImporter(fi.input)();
            },
            [&geo = problem->model.geometry](inp::GeantImport const& gi) {
                // For standalone, no processes should need to be ignored
                CELER_ASSERT(gi.ignore_processes.empty());
                CELER_ASSUME(
                    std::holds_alternative<G4VPhysicalVolume const*>(geo));

                // Don't capture the setup; leave Geant4 alive for now
                GeantImporter import{std::get<G4VPhysicalVolume const*>(geo)};
                return import(gi.data_selection);
            },
        },
        si.physics_import);

    if (si.geant_data)
    {
        CELER_NOT_IMPLEMENTED("loading data directly into celeritas::inp");
    }

    if (si.update)
    {
        CELER_NOT_IMPLEMENTED("updating input problem from external file");
    }

    StandaloneLoaded result;

    // Set up core params
    result.problem = setup::problem(*problem, imported);

    // Load events
    result.events = events(si.events, result.problem.core_params->particle());

    auto const& ctl = problem->control;
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
