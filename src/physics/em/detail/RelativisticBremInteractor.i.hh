//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Algorithms.hh"
#include "PhysicsConstants.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
RelativisticBremInteractor::RelativisticBremInteractor(
    const RelativisticBremNativeRef& shared,
    const ParticleTrackView&         particle,
    const Real3&                     direction,
    const CutoffView&                cutoffs,
    StackAllocator<Secondary>&       allocate,
    const MaterialView&              material,
    const ElementComponentId&        elcomp_id)
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
} // namespace detail
} // namespace celeritas
