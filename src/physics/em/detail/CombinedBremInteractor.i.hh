//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CombinedBremInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Algorithms.hh"
#include "base/Constants.hh"
#include "random/distributions/GenerateCanonical.hh"

#include "PhysicsConstants.hh"
#include "SBEnergyDistHelper.hh"
#include "SBEnergyDistribution.hh"
#include "SBPositronXsCorrector.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
CombinedBremInteractor::CombinedBremInteractor(
    const CombinedBremNativeRef& shared,
    const ParticleTrackView&     particle,
    const Real3&                 direction,
    const CutoffView&            cutoffs,
    StackAllocator<Secondary>&   allocate,
    const MaterialView&          material,
    const ElementComponentId&    elcomp_id)
    : shared_(shared)
    , inc_energy_(particle.energy())
    , inc_momentum_(particle.momentum())
    , inc_direction_(direction)
    , gamma_cutoff_(cutoffs.energy(shared.rb_data.ids.gamma))
    , allocate_(allocate)
    , material_(material)
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
                               shared_.rb_data.electron_mass,
                               shared_.rb_data.ids.gamma)
{
    CELER_EXPECT(is_electron_
                 || particle.particle_id() == shared_.rb_data.ids.positron);
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
