//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/StandaloneInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>

#include "Events.hh"
#include "Import.hh"
#include "Problem.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//! Physics list selection (TODO: move)
enum class PhysicsListSelection
{
    ftfp_bert,
    celer_ftfp_bert,  //!< FTFP BERT with Celeritas EM standard physics
    celer_em,  //!< Celeritas EM standard physics only
    size_,
};

//---------------------------------------------------------------------------//
/*!
 * Set up a Geant4 run manager and problem.
 *
 * \note We should change celer-g4 so it just uses \c GeantSetup as an outer
 * wrapper, rather than trying to be a Geant4 example.
 *
 * \todo The physics list (namely, whether to use hadronic physics or not) will
 * be combined into the "physics" problem input options.
 * \todo Should we have an option allow \c Problem::physics to be empty and add
 */
struct GeantSetup
{
    PhysicsListSelection physics_list{PhysicsListSelection::size_};

    //! TODO: most of these will be moved to Problem::Physics;
    //! some options (e.g., gamma_general) will not be applicable to Celeritas
    GeantPhysicsOptions physics_setup;
};

//---------------------------------------------------------------------------//
/*!
 * Celeritas setup for standalone apps.
 *
 * The order of initialization and loading follows the member declarations:
 * - System attributes (GPU activation etc.) are set first
 * - Problem info is loaded
 * - Geant4 is initialized (if not using full ROOT data)
 * - Geant4 data is loaded (also if not using full ROOT)
 * - External Geant4 data files (such as EM LOW) are loaded
 * - Optional tuning/diagnostic overrides are loaded
 * - Events are loaded
 *
 * The input \c Problem can be an embedded struct or a path to a file to
 * import.
 *
 * \todo physics_import will be a `std::optional<GeantImport>` after all the
 * \c ImportData is merged into \c Problem .
 */
struct StandaloneInput
{
    //! System attributes
    System system;
    //! Base problem options and input data
    std::variant<FileImport, Problem> problem;
    //! Set up Geant4 (if all the data isn't serialized)
    std::optional<GeantSetup> geant_setup;

    //! Whether using Geant4 or loading from ROOT
    std::variant<GeantImport, FileImport> physics_import;
    //! If using Geant4 or overriding or sparse input?
    std::optional<GeantDataImport> geant_data;
    //! If loading from an existing input, option to update data
    std::optional<UpdateImport> update;

    //! Primary particles
    Events events;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
