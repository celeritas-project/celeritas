//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/FrameworkInput.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <optional>

#include "Import.hh"
#include "System.hh"

namespace celeritas
{
namespace inp
{
struct Problem;

//---------------------------------------------------------------------------//
/*!
 * Describe how to import data into celeritas via an \c Input data structure.
 *
 * The order of initialization and loading follows the member declarations:
 * - System attributes (GPU activation etc.) are set
 * - Geant4 data is loaded
 * - External Geant4 data files (such as EM LOW) are loaded
 * - Optional control/diagnostic overrides are loaded
 * - Optional framework-defined adjustments are applied
 */
struct FrameworkInput
{
    //! Base system configuration
    System system;

    //! Configure what data to load from Geant4
    GeantImport geant;
    //! Load external data files
    GeantDataImport geant_data;
    //! Optionally add diagnostics and control parameters from an external file
    std::optional<UpdateImport> update;

    //! User application/framework-defined adjustments
    std::function<void(Problem&)> adjuster;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
