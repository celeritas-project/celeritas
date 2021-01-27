//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePE.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/em/LivermorePEInterface.hh"

namespace celeritas
{
struct ModelInteractPointers;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating a LivermorePEInteractor.
 */
struct LivermorePEPointers
{
    //! Model ID
    ModelId model_id;

    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleId electron_id;
    //! ID of a gamma
    ParticleId gamma_id;
    //! Livermore EPICS2014 photoelectric data
    LivermorePEParamsPointers data;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return model_id && inv_electron_mass > 0 && electron_id && gamma_id
               && data;
    }
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Livermore photoelectric interaction
void livermore_pe_interact(const LivermorePEPointers&   device_pointers,
                           const ModelInteractPointers& interaction);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
