//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/EPlusGGData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating an EPlusGGInteractor.
 */
struct EPlusGGData
{
    using Mass = units::MevMass;

    struct
    {
        //! Model ID
        ActionId action;
        //! ID of an positron
        ParticleId positron;
        //! ID of a gamma
        ParticleId gamma;
    } ids;

    //! Electron mass
    units::MevMass electron_mass;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids.action && ids.positron && ids.gamma
               && electron_mass > zero_quantity();
    }
};

using EPlusGGHostRef = EPlusGGData;
using EPlusGGDeviceRef = EPlusGGData;

//---------------------------------------------------------------------------//
}  // namespace celeritas
