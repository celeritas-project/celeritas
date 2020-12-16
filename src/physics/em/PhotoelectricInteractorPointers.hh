//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricInteractorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating a PhotoelectricInteractor.
 */
struct PhotoelectricInteractorPointers
{
    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;
    //! ID of an electron
    ParticleDefId electron_id;
    //! ID of a gamma
    ParticleDefId gamma_id;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return electron_id && gamma_id;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
