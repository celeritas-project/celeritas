//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "base/Macros.hh"
#include "base/Types.hh"

#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/UniformRealDistribution.hh"

#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"

#include "MollerBhabhaInteractorPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform Moller (e-e-) and Bhabha (e+e-) scattering.
 *
 * This is a model for both Moller and Bhabha scattering processes.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MollerBhabhaModel class, as documented in section 10.1 of the Geant4
 * Physics Reference (release 10.6).
 */
class MollerBhabhaInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MollerBhabhaInteractor(const MollerBhabhaInteractorPointers& shared,
                           const ParticleTrackView&              particle,
                           const Real3&                          inc_direction,
                           const bool&                           is_electron,
                           SecondaryAllocatorView& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    //// COMMON PROPERTIES ////

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {
        return units::MevEnergy{0.001}; // Must be at least 1 keV
    }

    //! Maximum incident energy for this model to be valid
    //! 100 TeV is the current EM limit
    //! This model's limit is worth double checking
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {
        return units::MevEnergy{100E6};
    }

  private:
    // Shared constant physics properties
    const MollerBhabhaInteractorPointers& shared_;
    // Incident energy
    const units::MevEnergy inc_energy_;
    // Incident momentum
    const units::MevMomentum inc_momentum_;
    // Incident direction
    const Real3& inc_direction_;
    // Incident particle type flag (electron or positron)
    bool inc_particle_is_electron_;

    // Allocate space for one or more secondary particles
    SecondaryAllocatorView& allocate_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MollerBhabhaInteractor.i.hh"
