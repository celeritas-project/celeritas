//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/MuHadIonizationData.hh
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
 * Data for the Bragg and ICRU73QO ionization models.
 *
 * The ICRU73QO and Bragg models apply to negatively and positively charged
 * muons and hadrons, respectively.
 */
struct MuHadIonizationData
{
    //! Particle IDs
    ParticleId inc_particle;  //!< Muon or hadron, depending on model
    ParticleId electron;

    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    //! Model-dependent kinetic energy limit [MeV]
    units::MevEnergy lowest_kin_energy;

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return inc_particle && electron && electron_mass > zero_quantity()
               && lowest_kin_energy > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
