//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportPhysicsVectorType.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store imported physics vector data [see Geant4's G4PhysicsVector.hh].
 *
 * MeV and cm units are defined in
 * \c GeantPhysicsTableWriter::fill_physics_vectors(...)
 */
struct ImportPhysicsVector
{
    enum class DataType
    {
        xs,
        energy_loss
    };

    ImportPhysicsVectorType vector_type;
    DataType                data_type;
    std::vector<real_type>  energy;   // [MeV] (Geant4's binVector)
    std::vector<real_type>  xs_eloss; // [1/cm or MeV] (Geant4's dataVector)
};

//---------------------------------------------------------------------------//
} // namespace celeritas
