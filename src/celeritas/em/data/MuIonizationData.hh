//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/MuIonizationData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * IDs used in muon ionization.
 */
struct MuIonizationIds
{
    ActionId action;
    ParticleId electron;
    ParticleId mu_minus;
    ParticleId mu_plus;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return action && electron && mu_minus && mu_plus;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
