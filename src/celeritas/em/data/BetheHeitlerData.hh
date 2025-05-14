//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/BetheHeitlerData.hh
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
 * Particle IDs used in Bethe-Heitler.
 */
struct BetheHeitlerIds
{
    //! ID of an electron
    ParticleId electron;
    //! ID of an positron
    ParticleId positron;
    //! ID of a gamma
    ParticleId gamma;

    //! Check whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return electron && positron && gamma;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating a BetheHeitlerInteractor.
 */
struct BetheHeitlerData
{
    //! Model/particle IDs
    BetheHeitlerIds ids;
    //! Electron mass [MevMass]
    units::MevMass electron_mass;
    //! LPM flag
    bool enable_lpm{false};

    //! Include a dielectric suppression effect in LPM functions
    static CELER_CONSTEXPR_FUNCTION bool dielectric_suppression()
    {
        return false;
    }

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
