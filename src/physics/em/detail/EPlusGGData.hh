//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"
#include "sim/Types.hh"

namespace celeritas
{
namespace detail
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

    //! electron mass [MevMass]
    real_type electron_mass;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids.action && ids.positron && ids.gamma && electron_mass > 0;
    }
};

using EPlusGGHostRef   = EPlusGGData;
using EPlusGGDeviceRef = EPlusGGData;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
