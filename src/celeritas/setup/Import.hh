//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Import.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/inp/Import.hh"

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Configure a Celeritas problem from input data.
 *
 * This implementation detail is how \c celeritas::inp data is used to
 * construct all the main Celeritas objects.
 *
 * \todo This will change to load data into a Problem, not ImportData.
 * Currently we need to fill tables and base what gets loaded on the existing
 * processes.
 */
namespace setup
{
//---------------------------------------------------------------------------//
// Load from a ROOT file
void physics_from(inp::PhysicsFromFile const&, ImportData& data);

//---------------------------------------------------------------------------//
// Load from Geant4 data files, filling in model data
void physics_from(inp::PhysicsFromGeant const&, ImportData& data);

//---------------------------------------------------------------------------//
// Load from Geant4 data files, filling in model data
void physics_from(inp::PhysicsFromGeantFiles const&, ImportData& data);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
