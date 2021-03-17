//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaInteractor.i.hh
//---------------------------------------------------------------------------//
#include "base/Range.hh"
#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "base/Algorithms.hh"
#include "random/distributions/UniformRealDistribution.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "MollerEnergyDistribution.hh"
#include "BhabhaEnergyDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident particle must be within the model's valid energy range. this
 * must be handled in code *before* the interactor is constructed.
 */
CELER_FUNCTION MollerBhabhaInteractor::MollerBhabhaInteractor(
    const MollerBhabhaPointers& shared,
    const ParticleTrackView&    particle,
    const Real3&                inc_direction,
    StackAllocator<Secondary>&  allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_momentum_(particle.momentum().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , inc_particle_is_electron_(particle.particle_id() == shared_.electron_id)
{
    CELER_EXPECT(particle.particle_id() == shared_.electron_id
                 || particle.particle_id() == shared_.positron_id);
    CELER_EXPECT(shared_.cutoff_energy < inc_energy_);
}

//---------------------------------------------------------------------------//
/*!
 * Sample e-e- or e+e- scattering using Moller or Bhabha models, depending on
 * the incident particle.
 *
 * See section 10.1.4 of the Geant4 physics reference manual (release 10.6).
 */
template<class Engine>
CELER_FUNCTION Interaction MollerBhabhaInteractor::operator()(Engine& rng)
{
    // Allocate memory for the produced electron
    Secondary* electron_secondary = this->allocate_(1);

    if (electron_secondary == nullptr)
    {
        // Fail to allocate space for a secondary
        return Interaction::from_failure();
    }

    // Set up energy threshold for secondary production
    real_type min_sampled_energy;
    if (shared_.cutoff_energy < shared_.min_valid_energy)
    {
        // Use model's low energy boundary
        min_sampled_energy = shared_.min_valid_energy;
    }
    else
    {
        // Use suggested cutoff value
        min_sampled_energy = shared_.cutoff_energy;
    }

    // TODO: do we assert here? Do we return Interaction::from_failure()? Do we
    // CELER_EXPECT this condition in the constructor?
    // Avoid an infinite sampling loop
    CELER_ASSERT(min_sampled_energy < 0.5 * inc_energy_);

    // Sample energy transfer fraction
    real_type epsilon;
    if (inc_particle_is_electron_)
    {
        MollerEnergyDistribution sample_moller(
            shared_.electron_mass_c_sq, min_sampled_energy, inc_energy_);
        epsilon = sample_moller(rng);
    }
    else
    {
        BhabhaEnergyDistribution sample_bhabha(
            shared_.electron_mass_c_sq, min_sampled_energy, inc_energy_);
        epsilon = sample_bhabha(rng);
    }

    // Sampled secondary kinetic energy
    const real_type secondary_energy = epsilon * inc_energy_;

    // Same equation as in ParticleTrackView::momentum_sq()
    const real_type secondary_momentum
        = std::sqrt(secondary_energy
                    * (secondary_energy + 2.0 * shared_.electron_mass_c_sq));

    const real_type total_energy = inc_energy_ + shared_.electron_mass_c_sq;

    // Calculate theta from energy-momentum conservation
    real_type secondary_cos_theta
        = secondary_energy * (total_energy + shared_.electron_mass_c_sq)
          / (secondary_momentum * inc_momentum_);

    secondary_cos_theta = min(secondary_cos_theta, 1.0);
    CELER_ASSERT(secondary_cos_theta >= -1.0 && secondary_cos_theta <= 1.0);

    // Sample phi isotropically
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

    // Create cartesian direction from the sampled theta and phi
    Real3 secondary_direction = rotate(
        from_spherical(secondary_cos_theta, sample_phi(rng)), inc_direction_);

    // Calculate incident particle final direction
    Real3 inc_exiting_direction;
    for (int i : range(3))
    {
        real_type inc_momentum_ijk       = inc_momentum_ * inc_direction_[i];
        real_type secondary_momentum_ijk = secondary_momentum
                                           * secondary_direction[i];
        inc_exiting_direction[i] = inc_momentum_ijk - secondary_momentum_ijk;
    }
    normalize_direction(&inc_exiting_direction);

    // Construct interaction for change to primary (incident) particle
    const real_type inc_exiting_energy = inc_energy_ - secondary_energy;
    Interaction     result;
    result.action      = Action::scattered;
    result.energy      = units::MevEnergy{inc_exiting_energy};
    result.secondaries = {electron_secondary, 1};
    result.direction   = inc_exiting_direction;

    // Assign values to the secondary particle
    electron_secondary[0].particle_id = shared_.electron_id;
    electron_secondary[0].energy      = units::MevEnergy{secondary_energy};
    electron_secondary[0].direction   = secondary_direction;

    return result;
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
