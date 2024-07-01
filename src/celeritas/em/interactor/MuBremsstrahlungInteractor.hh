//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/MuBremsstrahlungInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MuBremsstrahlungData.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsUtils.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/ReciprocalDistribution.hh"

namespace celeritas
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
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    using Momentum = units::MevMomentum;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    MuBremsstrahlungInteractor(MuBremsstrahlungData const& shared,
                               ParticleTrackView const& particle,
                               Real3 const& inc_direction,
                               StackAllocator<Secondary>& allocate,
                               MaterialView const& material,
                               ElementComponentId elcomp_id);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    template<class Engine>
    inline CELER_FUNCTION real_type sample_cos_theta(real_type gamma_energy,
                                                     Engine& rng) const;

    inline CELER_FUNCTION real_type
    differential_cross_section(real_type gamma_energy) const;

    // Shared constant physics properties
    MuBremsstrahlungData const& shared_;
    // Incident direction
    Real3 const& inc_direction_;
    // Allocate space for one or more secondary particles
    StackAllocator<Secondary>& allocate_;
    // Element properties
    ElementView const element_;
    // Incident particle
    ParticleTrackView const& particle_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION MuBremsstrahlungInteractor::MuBremsstrahlungInteractor(
    MuBremsstrahlungData const& shared,
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate,
    MaterialView const& material,
    ElementComponentId elcomp_id)
    : shared_(shared)
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , element_(material.make_element_view(elcomp_id))
    , particle_(particle)
{
    CELER_EXPECT(particle_.energy() >= shared_.min_incident_energy()
                 && particle_.energy() <= shared_.max_incident_energy());
    CELER_EXPECT(particle.particle_id() == shared_.ids.mu_minus
                 || particle.particle_id() == shared_.ids.mu_plus);
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the Muon Bremsstrahlung model.
 */
template<class Engine>
CELER_FUNCTION Interaction MuBremsstrahlungInteractor::operator()(Engine& rng)
{
    // Allocate space for gamma
    Secondary* secondary = allocate_(1);
    if (secondary == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    real_type const min_inc_kinetic_energy
        = min(value_as<Energy>(particle_.energy()),
              value_as<Energy>(shared_.min_incident_energy()));
    real_type const func_1
        = min_inc_kinetic_energy
          * this->differential_cross_section(min_inc_kinetic_energy);

    ReciprocalDistribution<real_type> sample_epsilon(
        min_inc_kinetic_energy, value_as<Energy>(particle_.energy()));

    real_type epsilon;
    do
    {
        epsilon = sample_epsilon(rng);
    } while (!BernoulliDistribution(
        epsilon * this->differential_cross_section(epsilon) / func_1)(rng));

    // Save outgoing secondary data
    secondary->particle_id = shared_.ids.gamma;
    secondary->energy = Energy{epsilon};
    secondary->direction = ExitingDirectionSampler{
        this->sample_cos_theta(epsilon, rng), inc_direction_}(rng);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.energy
        = units::MevEnergy{value_as<Energy>(particle_.energy()) - epsilon};
    result.direction = calc_exiting_direction(
        {value_as<Momentum>(particle_.momentum()), inc_direction_},
        {epsilon, secondary->direction});
    result.secondaries = {secondary, 1};

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
    real_type const gamma = particle_.lorentz_factor();
    real_type const r_max
        = gamma * constants::pi * real_type(0.5)
          * min(real_type(1.0),
                gamma * value_as<Mass>(particle_.mass()) / gamma_energy - 1);
    real_type const r_max_sq = ipow<2>(r_max);

    real_type const a = generate_canonical(rng) * r_max_sq / (1 + r_max_sq);

    return std::cos(std::sqrt(a / (1 - a)) / gamma);
}

//---------------------------------------------------------------------------//

CELER_FUNCTION real_type MuBremsstrahlungInteractor::differential_cross_section(
    real_type gamma_energy) const
{
    real_type dxsection = 0;

    if (gamma_energy >= value_as<Energy>(particle_.energy()))
    {
        return dxsection;
    }

    int const atomic_number = element_.atomic_number().unchecked_get();
    real_type const atomic_mass
        = value_as<units::AmuMass>(element_.atomic_mass());
    real_type const sqrt_e = std::sqrt(constants::euler);
    real_type const inc_total_energy = value_as<Mass>(particle_.mass())
                                       + value_as<Energy>(particle_.energy());
    real_type const rel_energy_transfer = gamma_energy / inc_total_energy;
    real_type const inc_mass_sq = ipow<2>(value_as<Mass>(particle_.mass()));
    real_type const delta = real_type(0.5) * inc_mass_sq * rel_energy_transfer
                            / (inc_total_energy - gamma_energy);

    // TODO: precalculate these data per element
    real_type d_n_prime, b, b1;
    real_type const d_n = real_type(1.54)
                          * std::pow(atomic_mass, real_type(0.27));

    if (atomic_number == 1)
    {
        d_n_prime = d_n;
        b = real_type(202.4);
        b1 = 446;
    }
    else
    {
        d_n_prime = std::pow(d_n, 1 - real_type(1) / atomic_number);
        b = 183;
        b1 = 1429;
    }

    real_type const inv_cbrt_z = 1 / element_.cbrt_z();
    real_type const electron_m = value_as<Mass>(shared_.electron_mass);

    real_type const phi_n = clamp_to_nonneg(std::log(
        b * inv_cbrt_z
        * (value_as<Mass>(particle_.mass()) + delta * (d_n_prime * sqrt_e - 2))
        / (d_n_prime * (electron_m + delta * sqrt_e * b * inv_cbrt_z))));

    real_type phi_e = 0;
    real_type const epsilon_max_prime
        = inc_total_energy
          / (1
             + real_type(0.5) * inc_mass_sq / (electron_m * inc_total_energy));

    if (gamma_energy < epsilon_max_prime)
    {
        real_type const inv_cbrt_z_sq = ipow<2>(inv_cbrt_z);
        phi_e = clamp_to_nonneg(std::log(
            b1 * inv_cbrt_z_sq * value_as<Mass>(particle_.mass())
            / ((1
                + delta * value_as<Mass>(particle_.mass())
                      / (ipow<2>(electron_m) * sqrt_e))
               * (electron_m + delta * sqrt_e * b1 * inv_cbrt_z_sq))));
    }

    dxsection = 16 * constants::alpha_fine_structure * constants::na_avogadro
                * ipow<2>(electron_m * constants::r_electron) * atomic_number
                * (atomic_number * phi_n + phi_e)
                * (1
                   - rel_energy_transfer
                         * (1 - real_type(0.75) * rel_energy_transfer))
                / (3 * inc_mass_sq * gamma_energy * atomic_mass);
    return dxsection;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
