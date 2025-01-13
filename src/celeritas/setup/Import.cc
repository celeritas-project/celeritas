//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Import.cc
//---------------------------------------------------------------------------//
#include "Import.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
/*!
 * Load from a file.
 */
void import(inp::FileImport const&, inp::Problem&)
{
    CELER_NOT_IMPLEMENTED("import");
}

//---------------------------------------------------------------------------//
/*!
 * Load from Geant4 in memory.
 */
void import(inp::GeantImport const&, inp::Problem&)
{
    CELER_NOT_IMPLEMENTED("import");
}

//---------------------------------------------------------------------------//
/*!
 * Load from Geant4 data files.
 */
void import(inp::GeantDataImport const&, inp::Problem&)
{
    CELER_NOT_IMPLEMENTED("import");
}

//---------------------------------------------------------------------------//
/*!
 * Update from another file.
 */
void import(inp::UpdateImport const&, inp::Problem&)
{
    CELER_NOT_IMPLEMENTED("import");
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
