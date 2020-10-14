//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerInteractorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "Material.mock.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating a BetheHeitlerInteractor.
 */
struct BetheHeitlerInteractorPointers
{
    //! Inverse of electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleDefId electron_id;
    //! ID of an electron
    ParticleDefId positron_id;
    //! ID of a gamma
    ParticleDefId gamma_id;

    //! Check whether the view is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return (inv_electron_mass > 0) && electron_id && positron_id
               && gamma_id;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
