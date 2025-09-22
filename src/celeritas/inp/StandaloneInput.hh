//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/StandaloneInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "celeritas/ext/GeantPhysicsOptions.hh"

#include "Events.hh"
#include "Import.hh"
#include "Problem.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Set up a Geant4 run manager and problem.
 *
 * \note We should change celer-g4 so it just uses \c GeantSetup as an outer
 * wrapper, rather than trying to be a Geant4 example. Or maybe just delete
 * GeantSetup.
 *
 * \note Most of the "physics options" will be deleted. Only a few options
 * specific to Geant4, such as \c gamma_general , will be left.
 *
 * \todo Add run manager type, number of threads
 */
using GeantSetup = GeantPhysicsOptions;

//---------------------------------------------------------------------------//
/*!
 * Celeritas setup for standalone apps.
 *
 * The order of initialization and loading (see \c celeritas::setup::Problem )
 * follows the member declarations:
 *
 * - System attributes (GPU activation etc.) are set first
 * - Problem info is loaded
 * - Geant4 is initialized (if not using full ROOT data)
 * - Geant4 data is loaded (also if not using full ROOT)
 * - External Geant4 data files (such as EM LOW) are loaded
 * - Optional control/diagnostic overrides are loaded
 * - Events are loaded
 *
 * The input \c Problem can be an embedded struct or a path to a file to
 * import.
 *
 * \internal
 *
 * \todo Replace problem with a variant (either problem or file to load from)?
 * \note geant_setup is always required for real problems
 * \todo \c physics_import will be a `std::optional<GeantImport>` after all the
 *   \c ImportData is merged into \c Problem .
 * \todo Add \c PhysicsFromGeantFiles after physics_import
 * \todo Add an option to override control/diagnostics?
 */
struct StandaloneInput
{
    //! System attributes
    System system;
    //! Base problem options and input data
    Problem problem;
    //! Set up Geant4 (if all the data isn't already loaded into Problem)
    std::optional<GeantSetup> geant_setup;

    //! Whether using Geant4 or loading from ROOT
    std::variant<PhysicsFromGeant, PhysicsFromFile> physics_import;

    //! Primary particles
    Events events;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
