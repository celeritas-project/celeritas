//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlungInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "base/Macros.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/MaterialView.hh"
#include "random/distributions/UniformRealDistribution.hh"
#include "random/distributions/ReciprocalDistribution.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "MuBremsstrahlungData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Muon bremsstrahlung.
 *
 * This is a model for the Bremsstrahlung process for muons. Given an incident
 * muon, the class computes the change to the incident muon direction and
 * energy, and it adds a single secondary gamma to the secondary stack.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4MuBremsstrahlungModel class, as documented in section 11.2
 * of the Geant4 Physics Reference (release 10.6).
 */
class MuBremsstrahlungInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MuBremsstrahlungInteractor(const MuBremsstrahlungData& shared,
                               const ParticleTrackView&    particle,
                               const Real3&                inc_direction,
                               const MaterialView&         material,
                               ElementComponentId          elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(real_type gamma_energy,
                                                     Engine&   rng) const;

    inline CELER_FUNCTION real_type
    differential_cross_section(real_type gamma_energy) const;

    // Shared constant physics properties
    const MuBremsstrahlungData& shared_;
    // Incident direction
    const Real3& inc_direction_;
    // Element properties
    const ElementView element_;
    // Incident particle
    const ParticleTrackView& particle_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MuBremsstrahlungInteractor::MuBremsstrahlungInteractor(
    const MuBremsstrahlungData& shared,
    const ParticleTrackView&    particle,
    const Real3&                inc_direction,
    const MaterialView&         material,
    ElementComponentId          elcomp_id)
    : shared_(shared)
    , inc_direction_(inc_direction)
    , element_(material.element_view(elcomp_id))
    , particle_(particle)
{
    CELER_EXPECT(particle_.energy() >= shared_.min_incident_energy()
                 && particle_.energy() <= shared_.max_incident_energy());
    CELER_EXPECT(particle.particle_id() == shared_.mu_minus_id
                 || particle.particle_id() == shared_.mu_plus_id);
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the Muon Bremsstrahlung model.
 */
template<class Engine>
CELER_FUNCTION Interaction MuBremsstrahlungInteractor::operator()(Engine& rng)
{
    const real_type min_inc_kinetic_energy = min(
        particle_.energy().value(), shared_.min_incident_energy().value());
    const real_type func_1
        = min_inc_kinetic_energy
          * this->differential_cross_section(min_inc_kinetic_energy);

    ReciprocalDistribution<real_type> sample_epsilon(
        min_inc_kinetic_energy, particle_.energy().value());

    real_type epsilon;
    do
    {
        epsilon = sample_epsilon(rng);
    } while (!BernoulliDistribution(
        epsilon * this->differential_cross_section(epsilon) / func_1)(rng));

    // Sample secondary direction.
    UniformRealDistribution<real_type> phi(0, 2 * constants::pi);

    real_type cost  = this->sample_cos_theta(epsilon, rng);
    Real3 gamma_dir = rotate(from_spherical(cost, phi(rng)), inc_direction_);

    Real3 inc_direction;
    for (int i : range(3))
    {
        inc_direction[i] = particle_.momentum().value() * inc_direction_[i]
                           - epsilon * gamma_dir[i];
    }
    normalize_direction(&inc_direction);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action    = Action::scattered;
    result.energy    = units::MevEnergy{particle_.energy().value() - epsilon};
    result.direction = inc_direction;

    // Save outgoing secondary data
    result.secondary.particle_id = shared_.gamma_id;
    result.secondary.energy      = units::MevEnergy{epsilon};
    result.secondary.direction   = gamma_dir;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample cosine of the angle between incident and secondary particles.
 */
template<class Engine>
CELER_FUNCTION real_type MuBremsstrahlungInteractor::sample_cos_theta(
    real_type gamma_energy, Engine& rng) const
{
    const real_type gamma = particle_.lorentz_factor();
    const real_type r_max
        = gamma * constants::pi * real_type(0.5)
          * min(real_type(1.0),
                gamma * particle_.mass().value() / gamma_energy - 1);
    const real_type r_max_sq = ipow<2>(r_max);

    const real_type a = generate_canonical(rng) * r_max_sq / (1 + r_max_sq);

    return std::cos(std::sqrt(a / (1 - a)) / gamma);
}

//---------------------------------------------------------------------------//

CELER_FUNCTION real_type MuBremsstrahlungInteractor::differential_cross_section(
    real_type gamma_energy) const
{
    real_type dxsection = 0;

    if (gamma_energy >= particle_.energy().value())
    {
        return dxsection;
    }

    const int       atomic_number    = element_.atomic_number();
    const real_type atomic_mass      = element_.atomic_mass().value();
    const real_type sqrt_e           = std::sqrt(constants::euler);
    const real_type inc_total_energy = particle_.mass().value()
                                       + particle_.energy().value();
    const real_type rel_energy_transfer = gamma_energy / inc_total_energy;
    const real_type inc_mass_sq         = ipow<2>(particle_.mass().value());
    const real_type delta = real_type(0.5) * inc_mass_sq * rel_energy_transfer
                            / (inc_total_energy - gamma_energy);

    real_type       d_n_prime, b, b1;
    const real_type d_n = real_type(1.54)
                          * std::pow(atomic_mass, real_type(0.27));

    if (atomic_number == 1)
    {
        d_n_prime = d_n;
        b         = real_type(202.4);
        b1        = 446;
    }
    else
    {
        d_n_prime = std::pow(d_n, 1 - real_type(1) / atomic_number);
        b         = 183;
        b1        = 1429;
    }

    const real_type inv_cbrt_z = 1 / element_.cbrt_z();
    const real_type electron_m = shared_.electron_mass.value();

    const real_type phi_n = clamp_to_nonneg(std::log(
        b * inv_cbrt_z
        * (particle_.mass().value() + delta * (d_n_prime * sqrt_e - 2))
        / (d_n_prime * (electron_m + delta * sqrt_e * b * inv_cbrt_z))));

    real_type       phi_e = 0;
    const real_type epsilon_max_prime
        = inc_total_energy
          / (1
             + real_type(0.5) * inc_mass_sq / (electron_m * inc_total_energy));

    if (gamma_energy < epsilon_max_prime)
    {
        const real_type inv_cbrt_z_sq = ipow<2>(inv_cbrt_z);
        phi_e                         = clamp_to_nonneg(std::log(
            b1 * inv_cbrt_z_sq * particle_.mass().value()
            / ((1
                + delta * particle_.mass().value()
                      / (ipow<2>(electron_m) * sqrt_e))
               * (electron_m + delta * sqrt_e * b1 * inv_cbrt_z_sq))));
    }

    dxsection = 16 * constants::alpha_fine_structure * constants::na_avogadro
                * ipow<2>(electron_m) * ipow<2>(constants::r_electron)
                * atomic_number * (atomic_number * phi_n + phi_e)
                * (1
                   - rel_energy_transfer
                         * (1 - real_type(0.75) * rel_energy_transfer))
                / (3 * inc_mass_sq * gamma_energy * atomic_mass);
    return dxsection;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
