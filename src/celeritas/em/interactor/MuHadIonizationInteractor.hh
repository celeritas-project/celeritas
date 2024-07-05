//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MuHadIonizationInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuHadIonizationData.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "detail/IoniFinalStateHelper.hh"
#include "detail/PhysicsConstants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform the discrete part of the muon or hadron ionization process.
 *
 * This simulates the production of delta rays by incident muons or hadrons
 * in the low energy region according to the Bragg and ICRU73QO models.
 *
 * \note This performs the same sampling routine as in Geant4's G4ICRU73QOModel
 * and G4BraggModel and as documented in the Geant4 Physics Reference Manual
 * (Release 11.1) section 12.2.1 and section 12.1. The sampling of the
 * secondary and the final state is identical in the ICRU73QO and Bragg models.
 */
class MuHadIonizationInteractor
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
    MuHadIonizationInteractor(MuHadIonizationData const& shared,
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
    // Muon mass [MeV / c^2]
    Mass inc_mass_;
    // Square of fractional speed of light for incident particle
    real_type beta_sq_;
    // Electron mass [MeV / c^2]
    Mass electron_mass_;
    // Secondary electron ID
    ParticleId electron_id_;
    // Minimum energy of the secondary electron [MeV]
    real_type min_secondary_energy_;
    // Maximum energy of the secondary electron [MeV]
    real_type max_secondary_energy_;

    //// HELPER FUNCTIONS ////

    inline CELER_FUNCTION Energy calc_max_secondary_energy() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
MuHadIonizationInteractor::MuHadIonizationInteractor(
    MuHadIonizationData const& shared,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate)
    : allocate_(allocate)
    , inc_direction_(inc_direction)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_mass_(particle.mass())
    , beta_sq_(particle.beta_sq())
    , electron_mass_(shared.electron_mass)
    , electron_id_(shared.electron)
    , min_secondary_energy_(min(
          value_as<Energy>(cutoffs.energy(electron_id_)),
          value_as<Energy>(shared.lowest_kin_energy) * value_as<Mass>(inc_mass_)
              / native_value_to<Mass>(constants::proton_mass).value()))
    , max_secondary_energy_(value_as<Energy>(this->calc_max_secondary_energy()))
{
    CELER_EXPECT(particle.particle_id() == shared.inc_particle);
    CELER_EXPECT(inc_energy_ > cutoffs.energy(electron_id_));
    CELER_EXPECT(inc_energy_ < detail::mu_bethe_bloch_lower_limit());
}

//---------------------------------------------------------------------------//
/*!
 * Simulate discrete muon ionization.
 */
template<class Engine>
CELER_FUNCTION Interaction MuHadIonizationInteractor::operator()(Engine& rng)
{
    if (min_secondary_energy_ >= max_secondary_energy_)
    {
        // No interaction if the maximum secondary energy is below the limit
        return Interaction::from_unchanged();
    }

    // Allocate secondary electron
    Secondary* secondary = allocate_(1);
    if (secondary == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    real_type energy;
    real_type target;
    do
    {
        energy = min_secondary_energy_ * max_secondary_energy_
                 / UniformRealDistribution(min_secondary_energy_,
                                           max_secondary_energy_)(rng);
        target = 1 - beta_sq_ * energy / max_secondary_energy_;
    } while (!BernoulliDistribution(target)(rng));

    // Sample the delta ray energy to construct the final sampler
    detail::IoniFinalStateHelper sample_interaction(inc_energy_,
                                                    inc_direction_,
                                                    inc_momentum_,
                                                    inc_mass_,
                                                    Energy{energy},
                                                    electron_mass_,
                                                    electron_id_,
                                                    secondary);

    // Update kinematics of the final state and return this interaction
    return sample_interaction(rng);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate maximum kinematically allowed kinetic energy of the secondary.
 *
 * TODO: Duplicated in \c MuBetheBlochInteractor.
 */
CELER_FUNCTION auto
MuHadIonizationInteractor::calc_max_secondary_energy() const -> Energy
{
    real_type mass_ratio = value_as<Mass>(electron_mass_)
                           / value_as<Mass>(inc_mass_);
    real_type tau = value_as<Energy>(inc_energy_) / value_as<Mass>(inc_mass_);
    return Energy{2 * value_as<Mass>(electron_mass_) * tau * (tau + 2)
                  / (1 + 2 * (tau + 1) * mass_ratio + ipow<2>(mass_ratio))};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
