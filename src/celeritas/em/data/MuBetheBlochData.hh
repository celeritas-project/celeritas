//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/MuBetheBlochData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "MuIonizationData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct MuBetheBlochData
{
    //! Model/particle IDs
    MuIonizationIds ids;
    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{0.2};
    }

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{1e8};
    }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
