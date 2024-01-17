//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/MuBremsstrahlungData.hh
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
 * Particle IDs used in muon brems.
 */
struct MuBremsstrahlungIds
{
    ActionId action;
    ParticleId gamma;
    ParticleId mu_minus;
    ParticleId mu_plus;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return action && gamma && mu_minus && mu_plus;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MuBremsstrahlungData
{
    //! Model/particle IDs
    MuBremsstrahlungIds ids;
    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{1e3};
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{1e7};
    }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity();
    }
};

using MuBremsstrahlungHostRef = MuBremsstrahlungData;
using MuBremsstrahlungDeviceRef = MuBremsstrahlungData;
using MuBremsstrahlungRef = MuBremsstrahlungData;

//---------------------------------------------------------------------------//
}  // namespace celeritas
