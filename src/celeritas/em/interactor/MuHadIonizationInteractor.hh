//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/random/distribution/InverseSquareDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuHadIonizationData.hh"
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
 * Perform the discrete part of the muon or hadron ionization process.
 *
 * This simulates the production of delta rays by incident muons or hadrons.
 * The same basic sampling routine is used by multiple models, but the energy
 * of the secondary is sampled from a distribution unique to the model.
 *
 * \note This performs the same sampling routine as in Geant4's \c
 * G4BetheBlochModel, \c G4MuBetheBlochModel, \c G4BraggModel, and \c
 * G4ICRU73QOModel, as documented in the Geant4 Physics Reference Manual
 * release 11.2 sections 11.1 and 12.1.5.
 */
template<class EnergySampler>
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
    // Electron mass [MeV / c^2]
    Mass electron_mass_;
    // Secondary electron ID
    ParticleId electron_id_;
    // Maximum energy of the secondary electron [MeV]
    real_type max_secondary_energy_;
    // Secondary electron energy distribution
    EnergySampler sample_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
template<class ES>
CELER_FUNCTION MuHadIonizationInteractor<ES>::MuHadIonizationInteractor(
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
    , electron_mass_(shared.electron_mass)
    , electron_id_(shared.electron)
    , sample_energy_(particle, cutoffs.energy(electron_id_), electron_mass_)
{
    CELER_EXPECT(inc_energy_ > sample_energy_.min_secondary_energy());
}

//---------------------------------------------------------------------------//
/*!
 * Simulate discrete muon ionization.
 */
template<class ES>
template<class Engine>
CELER_FUNCTION Interaction MuHadIonizationInteractor<ES>::operator()(Engine& rng)
{
    if (sample_energy_.min_secondary_energy()
        >= sample_energy_.max_secondary_energy())
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

    // Update kinematics of the final state and return the interaction
    return detail::IoniFinalStateHelper(inc_energy_,
                                        inc_direction_,
                                        inc_momentum_,
                                        inc_mass_,
                                        sample_energy_(rng),
                                        electron_mass_,
                                        electron_id_,
                                        secondary)(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
