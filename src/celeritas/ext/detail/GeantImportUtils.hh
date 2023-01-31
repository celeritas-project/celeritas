//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantImportUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/io/ImportPhysicsTable.hh"

class G4PhysicsTable;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
ImportPhysicsTable
import_table(G4PhysicsTable const& g4table, ImportTableType table_type);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
