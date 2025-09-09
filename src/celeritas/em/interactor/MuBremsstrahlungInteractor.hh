//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MuBremsstrahlungInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/ReciprocalDistribution.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuBremsstrahlungData.hh"
#include "celeritas/em/distribution/MuAngularDistribution.hh"
#include "celeritas/em/xs/MuBremsDiffXsCalculator.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

#include "detail/BremFinalStateHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform muon bremsstrahlung interaction.
 *
 * This is a model for the Bremsstrahlung process for muons. Given an incident
 * muon, the class computes the change to the incident muon direction and
 * energy, and it adds a single secondary gamma to the secondary stack.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuBremsstrahlungModel class, as documented in section 11.2
 * of the Geant4 Physics Reference (release 10.6).
 */
class MuBremsstrahlungInteractor
{
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MuBremsstrahlungInteractor(MuBremsstrahlungData const& shared,
                               ParticleTrackView const& particle,
                               Real3 const& inc_direction,
                               CutoffView const& cutoffs,
                               StackAllocator<Secondary>& allocate,
                               MaterialView const& material,
                               ElementComponentId elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant physics properties
    MuBremsstrahlungData const& shared_;
    // Incident direction
    Real3 const& inc_direction_;
    // Allocate space for one or more secondary particles
    StackAllocator<Secondary>& allocate_;
    // Element properties
    ElementView const element_;
    // Incident particle
    ParticleTrackView const& particle_;
    // Differential cross section calculator
    MuBremsDiffXsCalculator calc_dcs_;
    // Distribution to sample energy
    ReciprocalDistribution<> sample_energy_;
    // Envelope distribution for rejection sampling of gamma energy
    real_type envelope_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MuBremsstrahlungInteractor::MuBremsstrahlungInteractor(
    MuBremsstrahlungData const& shared,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    CutoffView const& cutoffs,
    StackAllocator<Secondary>& allocate,
    MaterialView const& material,
    ElementComponentId elcomp_id)
    : shared_(shared)
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , element_(material.element_record(elcomp_id))
    , particle_(particle)
    , calc_dcs_(
          element_, particle.energy(), particle.mass(), shared.electron_mass)
    , sample_energy_{value_as<Energy>(cutoffs.energy(shared.gamma)),
                     value_as<Energy>(particle_.energy())}
{
    CELER_EXPECT(particle.particle_id() == shared_.mu_minus
                 || particle.particle_id() == shared_.mu_plus);
    CELER_EXPECT(particle_.energy() > cutoffs.energy(shared.gamma));

    // Calculate rejection envelope: *assume* the highest cross section
    // is at its lowest value
    real_type gamma_cutoff = value_as<Energy>(cutoffs.energy(shared.gamma));
    envelope_ = gamma_cutoff * calc_dcs_(Energy{gamma_cutoff});
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the muon bremsstrahlung model.
 */
template<class Engine>
CELER_FUNCTION Interaction MuBremsstrahlungInteractor::operator()(Engine& rng)
{
    // Allocate space for gamma
    Secondary* secondary = allocate_(1);
    if (secondary == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Sample the energy transfer
    real_type gamma_energy;
    do
    {
        gamma_energy = sample_energy_(rng);
    } while (RejectionSampler{gamma_energy * calc_dcs_(Energy{gamma_energy}),
                              envelope_}(rng));

    MuAngularDistribution sample_costheta(
        particle_.energy(), particle_.mass(), Energy{gamma_energy});

    // Update kinematics of the final state and return this interaction
    return detail::BremFinalStateHelper(particle_.energy(),
                                        inc_direction_,
                                        particle_.momentum(),
                                        shared_.gamma,
                                        Energy{gamma_energy},
                                        sample_costheta(rng),
                                        secondary)(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
