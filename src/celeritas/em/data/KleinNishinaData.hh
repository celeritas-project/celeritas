//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/KleinNishinaData.hh
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
 * Model and particles IDs.
 */
struct KleinNishinaIds
{
    ActionId action;
    ParticleId electron;
    ParticleId gamma;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return action && electron && gamma;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a KleinNishinaInteractor.
 */
struct KleinNishinaData
{
    using Mass = units::MevMass;

    //! Model and particle identifiers
    KleinNishinaIds ids;

    //! 1 / electron mass [1 / MevMass]
    real_type inv_electron_mass;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && inv_electron_mass > 0;
    }
};

using KleinNishinaDeviceRef = KleinNishinaData;
using KleinNishinaHostRef = KleinNishinaData;

//---------------------------------------------------------------------------//
}  // namespace celeritas
