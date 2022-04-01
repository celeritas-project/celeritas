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
#include "physics/base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Particle IDs used in Bethe-Heitler.
 */
struct BetheHeitlerIds
{
    //! Model ID
    ModelId model;
    //! ID of an electron
    ParticleId electron;
    //! ID of an positron
    ParticleId positron;
    //! ID of a gamma
    ParticleId gamma;

    //! Check whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return model && electron && positron && gamma;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a BetheHeitlerInteractor.
 */
struct BetheHeitlerData
{
    //! Model/particle IDs
    BetheHeitlerIds ids;
    //! Electron mass [MevMass]
    real_type electron_mass{0};
    //! LPM flag
    bool enable_lpm{false};

    //! Include a dielectric suppression effect in LPM functions
    static CELER_CONSTEXPR_FUNCTION bool dielectric_suppression()
    {
        return false;
    }

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > 0;
    }
};

using BetheHeitlerHostRef   = BetheHeitlerData;
using BetheHeitlerDeviceRef = BetheHeitlerData;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
