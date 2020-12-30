//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Geant4 equivalent enum for Physics vector types.
 * [See Geant4's G4PhysicsVectorType.hh]
 */
enum class ImportPhysicsVectorType
{
    base,
    linear,
    log,
    ln,
    free,
    ordered_free,
    low_energy_free
};

//---------------------------------------------------------------------------//
/*!
 * Store imported physics vector data [see Geant4's G4PhysicsVector.hh].
 *
 * MeV and cm units are defined in
 * \c GeantPhysicsTableWriter::fill_physics_vectors(...)
 */
struct ImportPhysicsVector
{
    ImportPhysicsVectorType vector_type;
    std::vector<real_type>  energy;   // [MeV] (Geant4's binVector)
    std::vector<real_type>  value;    // [1/cm or MeV] (Geant4's dataVector)
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

const char* to_cstring(ImportPhysicsVectorType value);

//---------------------------------------------------------------------------//
} // namespace celeritas
