//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/data/MuDecayData.hh
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
 * Data storage for muon decay.
 */
struct MuDecayIds
{
    ParticleId mu_minus;
    ParticleId mu_plus;
    ParticleId electron;
    ParticleId positron;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return mu_minus && mu_plus && electron && positron;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a muon decay interactor.
 */
struct MuDecayData
{
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    //!@}

    //! Particle identifiers
    MuDecayIds ids;

    //! Muon/anti-muon mass [MeV]
    Mass muon_mass;

    //! Electron/positron mass [MeV]
    Mass electron_mass;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && muon_mass > zero_quantity()
               && electron_mass > zero_quantity();
    }
};

using MuDecayDeviceRef = MuDecayData;
using MuDecayHostRef = MuDecayData;

//---------------------------------------------------------------------------//
}  // namespace celeritas
