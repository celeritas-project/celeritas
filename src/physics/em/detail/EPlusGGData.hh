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

    //! Model ID
    ModelId model_id;

    //! electron mass [MevMass]
    real_type electron_mass;
    //! ID of an positron
    ParticleId positron_id;
    //! ID of a gamma
    ParticleId gamma_id;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && electron_mass > 0 && positron_id && gamma_id;
    }
};

using EPlusGGHostRef   = EPlusGGData;
using EPlusGGDeviceRef = EPlusGGData;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
