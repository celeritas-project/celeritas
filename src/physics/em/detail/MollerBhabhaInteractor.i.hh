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
#include "random/distributions/BernoulliDistribution.hh"

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

    // Sampled secondary kinetic energy
    const real_type secondary_energy = epsilon * inc_energy_.value();

    // Same equation as in ParticleTrackView::momentum_sq()
    const real_type secondary_momentum
        = std::sqrt(secondary_energy
                    * (secondary_energy + 2.0 * shared_.electron_mass_c_sq));

    const real_type total_energy = inc_energy_.value()
                                   + shared_.electron_mass_c_sq;

    // Calculate theta from energy-momentum conservation
    real_type secondary_cos_theta
        = secondary_energy * (total_energy + shared_.electron_mass_c_sq)
          / (secondary_momentum * inc_momentum_.value());

    secondary_cos_theta = min(secondary_cos_theta, 1.0);
    CELER_ASSERT(secondary_cos_theta >= -1.0 && secondary_cos_theta <= 1.0);

    // Sample phi isotropically
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

    // Create cartesian direction from the sampled theta and phi
    Real3 secondary_direction
        = from_spherical(secondary_cos_theta, sample_phi(rng));

    // Calculate incident particle final direction
    Real3 inc_exiting_direction;
    for (int i : range(3))
    {
        real_type inc_momentum_ijk = inc_momentum_.value() * inc_direction_[i];
        real_type secondary_momentum_ijk = secondary_momentum
                                           * secondary_direction[i];
        inc_exiting_direction[i] = inc_momentum_ijk - secondary_momentum_ijk;
    }
    normalize_direction(&inc_exiting_direction);

    // Construct interaction for change to primary (incident) particle
    const real_type inc_exiting_energy = inc_energy_.value() - secondary_energy;
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
    const real_type gamma          = total_energy / shared_.electron_mass_c_sq;
    const real_type gamma_sq       = ipow<2>(gamma);
    const real_type two_gamma_term = (2.0 * gamma - 1.0) / gamma_sq;

    // Lambda for f(epsilon) and g(epsilon), which are equivalent
    auto calc_f_g = [two_gamma_term](real_type epsilon) {
        const real_type complement_frac = 1.0 - epsilon;
        return 1.0 - two_gamma_term * epsilon
               + ipow<2>(epsilon)
                     * (1.0 - two_gamma_term
                        + (1.0 - two_gamma_term * complement_frac)
                              / ipow<2>(complement_frac));
    };

    const real_type rejection_g = calc_f_g(max_energy_fraction);

    UniformRealDistribution<> sample_inverse_epsilon(1 / max_energy_fraction,
                                                     1 / min_energy_fraction);

    // Sample epsilon
    real_type prob_f;
    real_type epsilon;
    do
    {
        epsilon = 1 / sample_inverse_epsilon(rng);
        prob_f  = calc_f_g(epsilon);

    } while (BernoulliDistribution(prob_f / rejection_g)(rng));

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
    const real_type gamma        = total_energy / shared_.electron_mass_c_sq;
    const real_type gamma_sq     = ipow<2>(gamma);
    const real_type beta_sq      = 1.0 - (1.0 / gamma_sq);
    const real_type y            = 1.0 / (1.0 + gamma);
    const real_type y_sq         = ipow<2>(y);
    const real_type one_minus_2y = 1.0 - 2.0 * y;
    const real_type b1           = 2.0 - y_sq;
    const real_type b2           = one_minus_2y * (3.0 + y_sq);
    const real_type b4           = ipow<3>(one_minus_2y);
    const real_type b3           = ipow<2>(one_minus_2y) + b4;

    // Lambda for f(epsilon) and g(epsilon)
    auto calc_f_g = [=](real_type epsilon_min, real_type epsilon_max) {
        return 1.0
               + (ipow<4>(epsilon_max) * b4 - ipow<3>(epsilon_min) * b3
                  + ipow<2>(epsilon_max) * b2 - epsilon_min * b1)
                     * beta_sq;
    };

    const real_type rejection_g
        = calc_f_g(min_energy_fraction, max_energy_fraction);

    UniformRealDistribution<> sample_inverse_epsilon(1 / max_energy_fraction,
                                                     1 / min_energy_fraction);

    // Sample epsilon
    real_type prob_f;
    real_type epsilon;
    do
    {
        // real_type random = generate_canonical(rng);
        epsilon = 1 / sample_inverse_epsilon(rng);
        prob_f  = calc_f_g(epsilon, epsilon);

    } while (BernoulliDistribution(prob_f / rejection_g)(rng));

    return epsilon;
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
