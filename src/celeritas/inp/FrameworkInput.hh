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
 * - Geant4 data is imported
 * - External Geant4 data files (such as EM LOW) are loaded
 * - Optional framework-defined adjustments are applied
 *
 * \todo Add an input option for kill_offload/disable
 */
struct FrameworkInput
{
    //! Base system configuration
    System system;

    //! Configure what data to load from Geant4
    PhysicsFromGeant physics_import;

    //! User application/framework-defined adjustments
    std::function<void(Problem&)> adjust;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
