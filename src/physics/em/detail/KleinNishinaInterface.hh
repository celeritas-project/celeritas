//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishinaInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
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
    ParticleId electron_id;
    //! ID of a gamma
    ParticleId gamma_id;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return model_id && inv_electron_mass > 0 && electron_id && gamma_id;
    }
};

using KleinNishinaDeviceRef = KleinNishinaPointers;
using KleinNishinaHostRef   = KleinNishinaPointers;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
