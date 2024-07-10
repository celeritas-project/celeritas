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
 * Device data for creating an interactor.
 */
struct MuBremsstrahlungData
{
    //! Particle IDs
    ParticleId gamma;
    ParticleId mu_minus;
    ParticleId mu_plus;

    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return gamma && mu_minus && mu_plus && electron_mass > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
