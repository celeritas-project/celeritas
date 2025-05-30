//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

    //! ID of an positron
    ParticleId positron;
    //! ID of a gamma
    ParticleId gamma;
    //! Electron mass
    units::MevMass electron_mass;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return positron && gamma && electron_mass > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
