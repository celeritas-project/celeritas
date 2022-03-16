//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerData.hh
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
 * Device data for creating a BetheHeitlerInteractor.
 */
struct BetheHeitlerData
{
    //! Model ID
    ModelId model_id;

    //! Electron mass [MevMass]
    real_type electron_mass;
    //! ID of an electron
    ParticleId electron_id;
    //! ID of an positron
    ParticleId positron_id;
    //! ID of a gamma
    ParticleId gamma_id;

    //! LPM flag
    bool enable_lpm;

    //! Include a dielectric suppression effect in LPM functions
    static CELER_CONSTEXPR_FUNCTION bool dielectric_suppression()
    {
        return false;
    }

    //! Check whether the view is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && electron_mass > 0 && electron_id && positron_id
               && gamma_id;
    }
};

using BetheHeitlerHostRef   = BetheHeitlerData;
using BetheHeitlerDeviceRef = BetheHeitlerData;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
