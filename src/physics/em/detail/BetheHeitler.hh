//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"

namespace celeritas
{
struct ModelInteractPointers;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating a BetheHeitlerInteractor.
 */
struct BetheHeitlerPointers
{
    //! Model ID
    ModelId model_id;

    //! Inverse of electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleId electron_id;
    //! ID of an positron
    ParticleId positron_id;
    //! ID of a gamma
    ParticleId gamma_id;

    //! Check whether the view is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && inv_electron_mass > 0 && electron_id && positron_id
               && gamma_id;
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Bethe-Heitler interaction
void bethe_heitler_interact(const BetheHeitlerPointers&  device_pointers,
                            const ModelInteractPointers& interaction);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
