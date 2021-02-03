//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabhaInteractor.i.hh
//---------------------------------------------------------------------------//
#include "base/Range.hh"
#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "base/Algorithms.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/UniformRealDistribution.hh"

#include <iostream>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident particle must be above the energy threshold: this should be
 * handled in code *before* the interactor is constructed.
 */
CELER_FUNCTION MollerBhabhaInteractor::MollerBhabhaInteractor(
    const MollerBhabhaPointers& shared,
    const ParticleTrackView&    particle,
    const Real3&                inc_direction,
    SecondaryAllocatorView&     allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_momentum_(particle.momentum().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , inc_particle_is_electron_(particle.particle_id() == shared_.electron_id)
{
    CELER_EXPECT(particle.particle_id() == shared_.electron_id
                 || particle.particle_id() == shared_.positron_id);
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

    real_type epsilon;

    if (inc_particle_is_electron_)
    {
        epsilon = this->sample_moller(rng);
    }
    else
    {
        epsilon = this->sample_bhabha(rng);
    }

    // Calculate secondary kinetic energy
    real_type secondary_energy = epsilon * inc_energy_.value();

    // Calculate secondary momentum
    // Same equation as in ParticleTrackView::momentum_sq()
    real_type secondary_momentum
        = sqrt(secondary_energy
               * (secondary_energy + 2.0 * shared_.electron_mass_c_sq));

    // Calculate theta from energy-momentum conservation
    const real_type total_energy = inc_energy_.value()
                                   + shared_.electron_mass_c_sq;
    real_type secondary_cos_theta
        = secondary_energy * (total_energy + shared_.electron_mass_c_sq)
          / (secondary_momentum * inc_momentum_.value());

    // Geant says: if (secondary_cos_theta > 1) { secondary_cos_theta = 1; }
    secondary_cos_theta = min(secondary_cos_theta, 1.0);
    CELER_ASSERT(secondary_cos_theta >= -1.0 && secondary_cos_theta <= 1.0);

    real_type secondary_sin_theta
        = sqrt((1.0 - secondary_cos_theta) * (1.0 + secondary_cos_theta));

    // Sample phi isotropically
    UniformRealDistribution<real_type> random_phi(0, 2 * constants::pi);
    real_type                          secondary_phi = random_phi(rng);

    // Create cartesian direction from the sampled theta and phi
    Real3 secondary_direction = {secondary_sin_theta * std::cos(secondary_phi),
                                 secondary_sin_theta * std::sin(secondary_phi),
                                 secondary_cos_theta};

    // Calculate incident particle final vector momentum
    Real3 inc_vec_momentum;
    Real3 secondary_vec_momentum;
    Real3 inc_exiting_vec_momentum;
    for (int i : range(3))
    {
        inc_vec_momentum[i]       = inc_momentum_.value() * inc_direction_[i];
        secondary_vec_momentum[i] = secondary_momentum * secondary_direction[i];
        inc_exiting_vec_momentum[i] = inc_vec_momentum[i]
                                      - secondary_vec_momentum[i];
    }

    real_type inc_exiting_momentum
        = std::sqrt(std::pow(inc_exiting_vec_momentum[0], 2)
                    + std::pow(inc_exiting_vec_momentum[1], 2)
                    + std::pow(inc_exiting_vec_momentum[2], 2));

    // Calculate incident particle final direction
    Real3 inc_exiting_direction;
    for (int i : range(3))
    {
        inc_exiting_direction[i] = inc_exiting_vec_momentum[i]
                                   / inc_exiting_momentum;
    }

    // Construct interaction for change to primary (incident) particle
    real_type   inc_exiting_energy = inc_energy_.value() - secondary_energy;
    Interaction result;
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
/*!
 * Sample Moller (e-e-) scattering.
 */
template<class Engine>
CELER_FUNCTION real_type MollerBhabhaInteractor::sample_moller(Engine& rng)
{
    // Max / min transferable energy fraction to the free electron
    const real_type max_energy_fraction = 0.5;
    const real_type min_energy_fraction = shared_.min_valid_energy_.value()
                                          / inc_energy_.value();

    // Set up sampling parameters
    const real_type total_energy = inc_energy_.value()
                                   + shared_.electron_mass_c_sq;
    const real_type gamma    = total_energy / shared_.electron_mass_c_sq;
    const real_type gamma_sq = ipow<2>(gamma);
    real_type       epsilon;
    real_type       z;
    real_type       rejection_function_g;
    real_type       random[2];

    real_type two_gamma_term = (2.0 * gamma - 1.0) / gamma_sq;
    real_type y              = 1.0 - max_energy_fraction;
    rejection_function_g     = 1.0 - two_gamma_term * max_energy_fraction
                           + max_energy_fraction * max_energy_fraction
                                 * (1.0 - two_gamma_term
                                    + (1.0 - two_gamma_term * y) / (y * y));

    do
    {
        random[0] = generate_canonical(rng);
        random[1] = generate_canonical(rng);

        epsilon = min_energy_fraction * max_energy_fraction
                  / (min_energy_fraction * (1.0 - random[0])
                     + max_energy_fraction * random[0]);
        y = 1.0 - epsilon;
        z = 1.0 - two_gamma_term * epsilon
            + epsilon * epsilon
                  * (1.0 - two_gamma_term
                     + (1.0 - two_gamma_term * y) / (y * y));
    } while (rejection_function_g * random[1] > z);

    return epsilon;
}

//---------------------------------------------------------------------------//
/*!
 * Sample Bhabha (e+e-) scattering.
 */
template<class Engine>
CELER_FUNCTION real_type MollerBhabhaInteractor::sample_bhabha(Engine& rng)
{
    // Max / min transferable energy fraction to the free electron
    const real_type max_energy_fraction = 1.0;
    const real_type min_energy_fraction = shared_.min_valid_energy_.value()
                                          / inc_energy_.value();

    // Set up sampling parameters
    const real_type total_energy = inc_energy_.value()
                                   + shared_.electron_mass_c_sq;
    const real_type gamma    = total_energy / shared_.electron_mass_c_sq;
    const real_type gamma_sq = ipow<2>(gamma);
    const real_type beta_sq  = 1.0 - (1.0 / gamma_sq);
    real_type       epsilon;
    real_type       z;
    real_type       rejection_function_g;
    real_type       random[2];

    real_type y    = 1.0 / (1.0 + gamma);
    real_type y2   = y * y;
    real_type y12  = 1.0 - 2.0 * y;
    real_type b1   = 2.0 - y2;
    real_type b2   = y12 * (3.0 + y2);
    real_type y122 = y12 * y12;
    real_type b4   = y122 * y12;
    real_type b3   = b4 + y122;

    y = max_energy_fraction * max_energy_fraction;

    rejection_function_g = 1.0
                           + (y * y * b4
                              - min_energy_fraction * min_energy_fraction
                                    * min_energy_fraction * b3
                              + y * b2 - min_energy_fraction * b1)
                                 * beta_sq;
    do
    {
        random[0] = generate_canonical(rng);
        random[1] = generate_canonical(rng);

        epsilon = min_energy_fraction * max_energy_fraction
                  / (min_energy_fraction * (1.0 - random[0])
                     + max_energy_fraction * random[0]);
        y = epsilon * epsilon;
        z = 1.0
            + (y * y * b4 - epsilon * y * b3 + y * b2 - epsilon * b1) * beta_sq;

    } while (rejection_function_g * random[1] > z);

    return epsilon;
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
