//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MuBetheBlochInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuBetheBlochData.hh"
#include "celeritas/em/distribution/MuBBEnergyDistribution.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

#include "detail/IoniFinalStateHelper.hh"
#include "detail/PhysicsConstants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform the discrete part of the muon ionization process.
 *
 * This simulates the production of delta rays by incident mu- and mu+ with
 * energies above 200 keV.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuBetheBlochModel and as documented in the Geant4 Physics Reference Manual
 * (Release 11.1) section 11.1.
 */
class MuBetheBlochInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct with shared and state data
    inline CELER_FUNCTION
    MuBetheBlochInteractor(MuBetheBlochData const& shared,
                           ParticleTrackView const& particle,
                           CutoffView const& cutoffs,
                           Real3 const& inc_direction,
                           StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Allocate space for the secondary particle
    StackAllocator<Secondary>& allocate_;
    // Incident direction
    Real3 const& inc_direction_;
    // Incident particle energy [MeV]
    Energy inc_energy_;
    // Incident particle momentum [MeV / c]
    Momentum inc_momentum_;
    // Muon mass
    Mass inc_mass_;
    // Electron mass
    Mass electron_mass_;
    // Secondary electron ID
    ParticleId electron_id_;
    // Secondary electron cutoff for current material [MeV]
    Energy electron_cutoff_;
    // Maximum energy of the secondary electron [MeV]
    Energy max_secondary_energy_;
    // Secondary electron energy distribution
    MuBBEnergyDistribution sample_energy_;

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION Energy calc_max_secondary_energy() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MuBetheBlochInteractor::MuBetheBlochInteractor(
    MuBetheBlochData const& shared,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate)
    : allocate_(allocate)
    , inc_direction_(inc_direction)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_mass_(particle.mass())
    , electron_mass_(shared.electron_mass)
    , electron_id_(shared.electron)
    , electron_cutoff_(cutoffs.energy(electron_id_))
    , max_secondary_energy_(this->calc_max_secondary_energy())
    , sample_energy_(inc_energy_,
                     inc_mass_,
                     particle.beta_sq(),
                     electron_mass_,
                     electron_cutoff_,
                     max_secondary_energy_)
{
    CELER_EXPECT(particle.particle_id() == shared.mu_minus
                 || particle.particle_id() == shared.mu_plus);
    CELER_EXPECT(inc_energy_ > electron_cutoff_);
    CELER_EXPECT(inc_energy_ >= detail::mu_bethe_bloch_lower_limit());
}

//---------------------------------------------------------------------------//
/*!
 * Simulate discrete muon ionization.
 */
template<class Engine>
CELER_FUNCTION Interaction MuBetheBlochInteractor::operator()(Engine& rng)
{
    if (electron_cutoff_ > max_secondary_energy_)
    {
        // No interaction if the maximum secondary energy is below the cutoff
        return Interaction::from_unchanged();
    }

    // Allocate secondary electron
    Secondary* secondary = allocate_(1);
    if (secondary == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Sample the delta ray energy to construct the final sampler
    detail::IoniFinalStateHelper sample_interaction(inc_energy_,
                                                    inc_direction_,
                                                    inc_momentum_,
                                                    inc_mass_,
                                                    sample_energy_(rng),
                                                    electron_mass_,
                                                    electron_id_,
                                                    secondary);

    // Update kinematics of the final state and return this interaction
    return sample_interaction(rng);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate maximum kinetic energy of the secondary electron.
 */
CELER_FUNCTION auto MuBetheBlochInteractor::calc_max_secondary_energy() const
    -> Energy
{
    real_type mass_ratio = value_as<Mass>(electron_mass_)
                           / value_as<Mass>(inc_mass_);
    real_type tau = value_as<Energy>(inc_energy_) / value_as<Mass>(inc_mass_);
    return Energy{2 * value_as<Mass>(electron_mass_) * tau * (tau + 2)
                  / (1 + 2 * (tau + 1) * mass_ratio + ipow<2>(mass_ratio))};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
