//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/RelativisticBremInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

#include "detail/BremFinalStateHelper.hh"
#include "detail/PhysicsConstants.hh"
#include "detail/RBEnergySampler.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform a high-energy Bremsstrahlung interaction.
 *
 * This is a relativistic Bremsstrahlung model for high-energy (> 1 GeV)
 * electrons and positrons.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4eBremsstrahlungRelModel class, as documented in section 10.2.2 of the
 * Geant4 Physics Reference (release 10.7).
 */
class RelativisticBremInteractor
{
  public:
    //!@{
    //! Type aliases
    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    using ElementData = RelBremElementData;
    using ItemIdT = celeritas::ItemId<unsigned int>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    RelativisticBremInteractor(RelativisticBremRef const& shared,
                               ParticleTrackView const& particle,
                               Real3 const& direction,
                               CutoffView const& cutoffs,
                               StackAllocator<Secondary>& allocate,
                               MaterialView const& material,
                               ElementComponentId const& elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant physics properties
    RelativisticBremRef const& shared_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle momentum
    const Momentum inc_momentum_;
    // Incident direction
    Real3 const& inc_direction_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;

    //// HELPER CLASSES ////

    // A helper to sample the photon energy from the relativistic model
    RBEnergySampler rb_energy_sampler_;
    // A helper to update the final state of the primary and the secondary
    BremFinalStateHelper final_state_interaction_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
RelativisticBremInteractor::RelativisticBremInteractor(
    RelativisticBremRef const& shared,
    ParticleTrackView const& particle,
    Real3 const& direction,
    CutoffView const& cutoffs,
    StackAllocator<Secondary>& allocate,
    MaterialView const& material,
    ElementComponentId const& elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(direction)
    , gamma_cutoff_(cutoffs.energy(shared.ids.gamma))
    , allocate_(allocate)
    , rb_energy_sampler_(shared, particle, cutoffs, material, elcomp_id)
    , final_state_interaction_(inc_energy_,
                               inc_direction_,
                               inc_momentum_,
                               shared.electron_mass,
                               shared.ids.gamma)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.electron
                 || particle.particle_id() == shared_.ids.positron);
    CELER_EXPECT(gamma_cutoff_ > zero_quantity());

    // Valid energy region of the relativistic e-/e+ Bremsstrahlung model
    CELER_EXPECT(inc_energy_ > seltzer_berger_limit());
}

//---------------------------------------------------------------------------//
/*!
 * Sample the production of photons and update final states
 */
template<class Engine>
CELER_FUNCTION Interaction RelativisticBremInteractor::operator()(Engine& rng)
{
    // Allocate space for the brems photon
    Secondary* secondaries = this->allocate_(1);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for the secondary
        return Interaction::from_failure();
    }

    // Sample the bremsstrahlung photon energy
    Energy gamma_energy = rb_energy_sampler_(rng);

    // Update kinematics of the final state and return this interaction
    return final_state_interaction_(rng, gamma_energy, secondaries);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
