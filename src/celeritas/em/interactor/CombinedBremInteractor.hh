//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    using ElementData = RelBremElementData;
    using ItemIdT = celeritas::ItemId<unsigned int>;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    CombinedBremInteractor(CombinedBremRef const& shared,
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

    // SB and relativistic data
    CombinedBremRef const& shared_;
    // Incident particle energy
    Energy const inc_energy_;
    // Incident particle direction
    Momentum const inc_momentum_;
    // Incident particle direction
    Real3 const& inc_direction_;
    // Energy cutoffs
    CutoffView const& cutoffs_;
    // Production cutoff for gammas
    Energy const gamma_cutoff_;
    // Allocate space for a secondary particle
    StackAllocator<Secondary>& allocate_;
    // Material properties
    MaterialView const& material_;
    // Element in which interaction occurs
    ElementComponentId const elcomp_id_;
    // Incident particle flag for selecting XS correction factor
    bool const is_electron_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
CombinedBremInteractor::CombinedBremInteractor(
    CombinedBremRef const& shared,
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
    , cutoffs_(cutoffs)
    , gamma_cutoff_(cutoffs.energy(shared.rb_data.ids.gamma))
    , allocate_(allocate)
    , material_(material)
    , elcomp_id_(elcomp_id)
    , is_electron_(particle.particle_id() == shared.rb_data.ids.electron)
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
    // TODO: reject this interaction before launching the kernel by using
    // correct material-dependent lower bounds for the interaction
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
    Energy gamma_energy;
    if (inc_energy_ > detail::seltzer_berger_limit())
    {
        detail::RBEnergySampler sample_energy{
            shared_.rb_data, inc_energy_, cutoffs_, material_, elcomp_id_};
        gamma_energy = sample_energy(rng);
    }
    else
    {
        detail::SBEnergySampler sample_energy{shared_.sb_differential_xs,
                                              inc_energy_,
                                              gamma_cutoff_,
                                              material_,
                                              elcomp_id_,
                                              shared_.rb_data.electron_mass,
                                              is_electron_};
        gamma_energy = sample_energy(rng);
    }

    // Sample the bremsstrahlung photon energy to construct the final sampler
    detail::BremFinalStateHelper sample_interaction(
        inc_energy_,
        inc_direction_,
        inc_momentum_,
        shared_.rb_data.electron_mass,
        shared_.rb_data.ids.gamma,
        gamma_energy,
        secondaries);

    // Update kinematics of the final state and return this interaction
    return sample_interaction(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
