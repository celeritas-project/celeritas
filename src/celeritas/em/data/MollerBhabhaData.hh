//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/MollerBhabhaData.hh
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
 * Model and particles IDs for Moller Bhabha.
 */
struct MollerBhabhaIds
{
    ActionId action;
    ParticleId electron;
    ParticleId positron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return action && electron && positron;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MollerBhabhaData
{
    //! Model and particle IDs
    MollerBhabhaIds ids;

    //! Electron mass * c^2 [MeV]
    units::MevMass electron_mass;

    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{100e6};
    }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity();
    }
};

using MollerBhabhaHostRef = MollerBhabhaData;
using MollerBhabhaDeviceRef = MollerBhabhaData;

//---------------------------------------------------------------------------//
}  // namespace celeritas
