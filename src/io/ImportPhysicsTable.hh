//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportProcessType.hh"
#include "ImportTableType.hh"
#include "ImportProcess.hh"
#include "ImportModel.hh"
#include "physics/base/ParticleMd.hh"
#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store physics tables.
 *
 * The geant-exporter app stores Geant4 physics tables into a ROOT file, while
 * the RootImporter class is responsible for loading said data into memory.
 */
struct ImportPhysicsTable
{
    ImportProcessType                process_type;
    ImportTableType                  table_type;
    ImportProcess                    process;
    ImportModel                      model;
    PDGNumber                        particle;
    std::vector<ImportPhysicsVector> physics_vectors;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
