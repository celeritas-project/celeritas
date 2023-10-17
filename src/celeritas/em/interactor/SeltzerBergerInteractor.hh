//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/SeltzerBergerInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/SeltzerBergerData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

#include "detail/BremFinalStateHelper.hh"
#include "detail/PhysicsConstants.hh"
#include "detail/SBEnergySampler.hh"

namespace celeritas
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
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    //! Construct sampler from device/shared and state data
    inline CELER_FUNCTION
    SeltzerBergerInteractor(SeltzerBergerRef const& shared,
                            ParticleTrackView const& particle,
                            Real3 const& inc_direction,
                            CutoffView const& cutoffs,
                            StackAllocator<Secondary>& allocate,
                            MaterialView const& material,
                            ElementComponentId const& elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////
    // Device (host CPU or GPU device) references
    SeltzerBergerRef const& shared_;
    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Momentum inc_momentum_;
    // Incident particle direction
    Real3 const& inc_direction_;
    // Incident particle flag for selecting XS correction factor
    bool const inc_particle_is_electron_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Element in which interaction occurs
    const ElementComponentId elcomp_id_;

    //// HELPER CLASSES ////
    // A helper to sample the bremsstrahlung photon energy
    detail::SBEnergySampler sample_photon_energy_;
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
    SeltzerBergerRef const& shared,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    CutoffView const& cutoffs,
    StackAllocator<Secondary>& allocate,
    MaterialView const& material,
    ElementComponentId const& elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(inc_direction)
    , inc_particle_is_electron_(particle.particle_id() == shared_.ids.electron)
    , gamma_cutoff_(cutoffs.energy(shared.ids.gamma))
    , allocate_(allocate)
    , elcomp_id_(elcomp_id)
    , sample_photon_energy_(shared.differential_xs,
                            particle.energy(),
                            gamma_cutoff_,
                            material,
                            elcomp_id,
                            shared.electron_mass,
                            inc_particle_is_electron_)
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
    // TODO: reject this interaction before executing the kernel by using
    // correct material-dependent lower bounds for the interaction
    if (gamma_cutoff_ > inc_energy_)
    {
        return Interaction::from_unchanged(inc_energy_, inc_direction_);
    }

    // Allocate space for the brems photon
    Secondary* secondaries = allocate_(1);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for the secondary
        return Interaction::from_failure();
    }

    // Sample the bremsstrahlung photon energy to construct the final sampler
    detail::BremFinalStateHelper sample_interaction(inc_energy_,
                                                    inc_direction_,
                                                    inc_momentum_,
                                                    shared_.electron_mass,
                                                    shared_.ids.gamma,
                                                    sample_photon_energy_(rng),
                                                    secondaries);

    // Update kinematics of the final state and return this interaction
    return sample_interaction(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
