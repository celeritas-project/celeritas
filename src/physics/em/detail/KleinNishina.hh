//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishina.hh
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
 * Device data for creating a KleinNishinaInteractor.
 */
struct KleinNishinaPointers
{
    //! Model ID
    ModelId model_id;

    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleDefId electron_id;
    //! ID of a gamma
    ParticleDefId gamma_id;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return model_id && inv_electron_mass > 0 && electron_id && gamma_id;
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the KN interaction
void klein_nishina_interact(const KleinNishinaPointers&  device_pointers,
                            const ModelInteractPointers& interaction);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
