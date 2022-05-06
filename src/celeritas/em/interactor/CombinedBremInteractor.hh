//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/CombinedBremInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/CombinedBremData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

#include "detail/BremFinalStateHelper.hh"
#include "detail/PhysicsConstants.hh"
#include "detail/RBEnergySampler.hh"
#include "detail/SBEnergySampler.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply either Seltzer-Berger or Relativistic depending on energy.
 *
 * This is a combined bremsstrahlung interactor consisted of the Seltzer-Berger
 * interactor at the low energy (< 1 GeV) and the relativistic bremsstrahlung
 * interactor at the high energy for the e-/e+ bremsstrahlung process.
 *
 * \todo: see if there's any occupancy/performance difference by defining the
 * samplers *inside* the conditional on "is_relativistic".
 */
class CombinedBremInteractor
{
    //!@{
    //! Type aliases
    using Energy      = units::MevEnergy;
    using Momentum    = units::MevMomentum;
    using ElementData = detail::RelBremElementData;
    using ItemIdT     = celeritas::ItemId<unsigned int>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    CombinedBremInteractor(const CombinedBremRef&     shared,
                           const ParticleTrackView&   particle,
                           const Real3&               direction,
                           const CutoffView&          cutoffs,
                           StackAllocator<Secondary>& allocate,
                           const MaterialView&        material,
                           const ElementComponentId&  elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Incident particle energy
    const Energy inc_energy_;
    // Incident particle direction
    const Momentum inc_momentum_;
    // Incident particle direction
    const Real3& inc_direction_;
    // Production cutoff for gammas
    const Energy gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Element in which interaction occurs
    const ElementComponentId elcomp_id_;
    // Incident particle flag for selecting XS correction factor
    const bool is_electron_;
    // Flag for selecting the relativistic bremsstrahlung model
    const bool is_relativistic_;

    //// HELPER CLASSES ////

    // A helper to Sample the photon energy from the relativistic model
    RBEnergySampler rb_energy_sampler_;
    // A helper to sample the photon energy from the SeltzerBerger model
    SBEnergySampler sb_energy_sampler_;
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
CombinedBremInteractor::CombinedBremInteractor(
    const CombinedBremRef&     shared,
    const ParticleTrackView&   particle,
    const Real3&               direction,
    const CutoffView&          cutoffs,
    StackAllocator<Secondary>& allocate,
    const MaterialView&        material,
    const ElementComponentId&  elcomp_id)
    : inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(direction)
    , gamma_cutoff_(cutoffs.energy(shared.rb_data.ids.gamma))
    , allocate_(allocate)
    , elcomp_id_(elcomp_id)
    , is_electron_(particle.particle_id() == shared.rb_data.ids.electron)
    , is_relativistic_(particle.energy() > seltzer_berger_limit())
    , rb_energy_sampler_(shared.rb_data, particle, cutoffs, material, elcomp_id)
    , sb_energy_sampler_(shared.sb_differential_xs,
                         particle,
                         gamma_cutoff_,
                         material,
                         elcomp_id,
                         shared.rb_data.electron_mass,
                         is_electron_)
    , final_state_interaction_(inc_energy_,
                               inc_direction_,
                               inc_momentum_,
                               shared.rb_data.electron_mass,
                               shared.rb_data.ids.gamma)
{
    CELER_EXPECT(is_electron_
                 || particle.particle_id() == shared.rb_data.ids.positron);
    CELER_EXPECT(gamma_cutoff_.value() > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the production of bremsstrahlung photons using a combined model.
 */
template<class Engine>
CELER_FUNCTION Interaction CombinedBremInteractor::operator()(Engine& rng)
{
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
    Energy gamma_energy = (is_relativistic_) ? rb_energy_sampler_(rng)
                                             : sb_energy_sampler_(rng);

    // Update kinematics of the final state and return this interaction
    return final_state_interaction_(rng, gamma_energy, secondaries);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
