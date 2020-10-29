//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImportPhysicsVectorType.hh
//! \brief Geant4 PhysicsVector type enumerator
//---------------------------------------------------------------------------//
#pragma once

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
} // namespace celeritas
