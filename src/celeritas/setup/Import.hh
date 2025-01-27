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
//---------------------------------------------------------------------------//
/*!
 * Configure Celeritas problems from input data.
 *
 * This implementation detail is how \c celeritas::inp data is used to
 * construct all the main Celeritas objects.
 */
namespace setup
{
//---------------------------------------------------------------------------//
// Load from a file
void import(inp::FileImport const& file, inp::Problem& problem);
// Load from Geant4 in memory
void import(inp::GeantImport const& file, inp::Problem& problem);
// Load from Geant4 data files
void import(inp::GeantDataImport const& file, inp::Problem& problem);
// Update from another file
void import(inp::UpdateImport const& file, inp::Problem& problem);

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
