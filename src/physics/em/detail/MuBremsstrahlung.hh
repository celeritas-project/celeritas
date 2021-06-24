//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlung.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
template<MemSpace M>
struct ModelInteractRefs;
    
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MuBremsstrahlungInteractorPointers
{
    //! Model ID
    ModelId model_id;
    //! ID of a gamma
    ParticleId gamma_id;
    //! ID of a muon
    ParticleId mu_minus_id;
    //! ID of a muon
    ParticleId mu_plus_id;
    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && gamma_id && mu_minus_id && mu_plus_id; 
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Muon Bremsstrahlung interaction
void mu_bremsstrahlung_interact(
    const MuBremsstrahlungInteractorPointers&  device_pointers,
    const ModelInteractRefs<MemSpace::device>& interaction);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

