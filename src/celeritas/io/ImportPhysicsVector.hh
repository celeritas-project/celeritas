//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportPhysicsVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Geant4 equivalent enum for Physics vector types.
 *
 * [See Geant4's G4PhysicsVectorType.hh]
 */
enum class ImportPhysicsVectorType
{
    unknown,
    linear,  //!< Uniform and linear in x
    log,  //!< Uniform and logarithmic in x
    free,  //!< Nonuniform in x
};

//---------------------------------------------------------------------------//
/*!
 * Store imported physics vector data [see Geant4's G4PhysicsVector.hh].
 *
 * Each vector's x axis is structured according to the vector_type. X is
 * usually energy, but (as in the case of "inverse range") can be distance or
 * any other arbitrary value.
 */
struct ImportPhysicsVector
{
    ImportPhysicsVectorType vector_type;
    std::vector<double> x;  //!< Geant4 binVector
    std::vector<double> y;  //!< Geant4 dataVector
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(ImportPhysicsVectorType value);

//---------------------------------------------------------------------------//
}  // namespace celeritas
