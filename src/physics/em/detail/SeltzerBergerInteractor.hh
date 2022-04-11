//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBergerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/StackAllocator.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "physics/material/Types.hh"

#include "BremFinalStateHelper.hh"
#include "PhysicsConstants.hh"
#include "SBEnergySampler.hh"
#include "SeltzerBergerData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Seltzer-Berger model for electron and positron bremsstrahlung processes.
 *
 * Given an incoming electron or positron of sufficient energy (as per
 * CutOffView), this class provides the energy loss of these particles due to
 * radiation of photons in the field of a nucleus. This model improves accuracy
 * using cross sections based on interpolation of published tables from Seltzer
 * and Berger given in Nucl. Instr. and Meth. in Phys. Research B, 12(1):95–134
 * (1985) and Atomic Data and Nuclear Data Tables, 35():345 (1986). The cross
 * sections are obtained from SBEnergyDistribution and are appropriately scaled
 * in the case of positrons via SBPositronXsCorrector.
 *
 * \note This interactor performs an analogous sampling as in Geant4's
 * G4SeltzerBergerModel, documented in 10.2.1 of the Geant Physics Reference
 * (release 10.6). The implementation is based on Geant4 10.4.3.
 */
class SeltzerBergerInteractor
{
  public:
    //!@{
    //! Type aliases
    using Energy   = units::MevEnergy;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct sampler from device/shared and state data
    inline CELER_FUNCTION
    SeltzerBergerInteractor(const SeltzerBergerRef&    shared,
                            const ParticleTrackView&   particle,
                            const Real3&               inc_direction,
                            const CutoffView&          cutoffs,
                            StackAllocator<Secondary>& allocate,
                            const MaterialView&        material,
                            const ElementComponentId&  elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////
    // Device (host CPU or GPU device) references
    const SeltzerBergerRef& shared_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Momentum inc_momentum_;
    // Incident particle direction
    const Real3& inc_direction_;
    // Incident particle flag for selecting XS correction factor
    const bool inc_particle_is_electron_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Element in which interaction occurs
    const ElementComponentId elcomp_id_;

    //// HELPER CLASSES ////
    // A helper to sample the bremsstrahlung photon energy
    SBEnergySampler sb_energy_sampler_;
    // A helper to update the final state of the primary and the secondary
    BremFinalStateHelper final_state_interaction_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared/device and state data.
 *
 * The incident particle must be within the model's valid energy range. this
 * must be handled in code *before* the interactor is constructed.
 */
CELER_FUNCTION SeltzerBergerInteractor::SeltzerBergerInteractor(
    const SeltzerBergerRef&    shared,
    const ParticleTrackView&   particle,
    const Real3&               inc_direction,
    const CutoffView&          cutoffs,
    StackAllocator<Secondary>& allocate,
    const MaterialView&        material,
    const ElementComponentId&  elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(inc_direction)
    , inc_particle_is_electron_(particle.particle_id() == shared_.ids.electron)
    , gamma_cutoff_(cutoffs.energy(shared.ids.gamma))
    , allocate_(allocate)
    , elcomp_id_(elcomp_id)
    , sb_energy_sampler_(shared.differential_xs,
                         particle,
                         gamma_cutoff_,
                         material,
                         elcomp_id,
                         shared.electron_mass,
                         inc_particle_is_electron_)
    , final_state_interaction_(inc_energy_,
                               inc_direction_,
                               inc_momentum_,
                               shared.electron_mass,
                               shared.ids.gamma)
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.electron
                 || particle.particle_id() == shared_.ids.positron);
    CELER_EXPECT(gamma_cutoff_ > zero_quantity());
}

//---------------------------------------------------------------------------//
/*!
 * Bremsstrahlung using the Seltzer-Berger model.
 *
 * See section 10.2.1 of the Geant physics reference 10.6.
 */
template<class Engine>
CELER_FUNCTION Interaction SeltzerBergerInteractor::operator()(Engine& rng)
{
    // Check if secondary can be produced. If not, this interaction cannot
    // happen and the incident particle must undergo an energy loss process
    // instead.
    if (gamma_cutoff_ > inc_energy_)
    {
        return Interaction::from_unchanged(inc_energy_, inc_direction_);
    }

    // Allocate space for the brems photon
    Secondary* secondaries = this->allocate_(1);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for the secondary
        return Interaction::from_failure();
    }

    // Sample the bremsstrahlung photon energy
    Energy gamma_energy = sb_energy_sampler_(rng);

    // Update kinematics of the final state and return this interaction
    return final_state_interaction_(rng, gamma_energy, secondaries);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
