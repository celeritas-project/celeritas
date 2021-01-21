//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGG.hh
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
 * Device data for creating an EPlusGGInteractor.
 */
struct EPlusGGPointers
{
    //! Model ID
    ModelId model_id;

    //! electron mass [MevMass]
    real_type electron_mass;
    //! ID of an positron
    ParticleDefId positron_id;
    //! ID of a gamma
    ParticleDefId gamma_id;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && electron_mass > 0 && positron_id && gamma_id;
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the EPlusGG interaction
void eplusgg_interact(const EPlusGGPointers&       eplusgg,
                      const ModelInteractPointers& model);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
